# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
import bisect
import copy
from enum import Enum
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path
import itertools
import logging

import matplotlib.pyplot as plt

import h5py
import hdf5plugin
from nexusformat.nexus import NeXusError, NXdata, NXsubentry
import numpy as np
import dask.array as da

from argsui.args import no_arg_field
from argsui import (
    arg_field,
    ArgType,
    ConfigFileArgs,
    InteractiveArgs,
    FilePathType,
)

from ..lib.nxs import NexusFile, NxAxis, NxAxes, create_field, FieldOptions
from ..lib.chunker import count_chunks_to_cover
from ..lib.bounds import Shape, Chunk
from ..lib.image import imshow_sparse
from ..lib.spectrum import plot_spectrum
from ..lib.utils import simplify_chunks

from icecream import install as icecream_install

icecream_install()
logger = logging.getLogger(__name__)


class MissingArgumentError(Exception):
    def __init__(self, argument_name: str, group: str, axis: str | None = None):
        suffix = f" on axis '{axis}'." if axis is not None else "."
        super().__init__(f"Missing '{argument_name}' for '{group}'{suffix}")


class InvalidArgumentError(Exception):
    def __init__(
        self,
        argument_name: str,
        group: str,
        axis: str | None = None,
        error: str | None = None,
    ):
        suffix = error if error is not None else "."
        suffix = f" on axis '{axis}': {suffix}" if axis is not None else suffix

        super().__init__(f"Invalid '{argument_name}' for '{group}'{suffix}")


class ActionType(Enum):
    Error = "error"
    Leave = "leave"
    Loop = "loop"
    Sum = "sum"
    Max = "max"


class SliceType(Enum):
    Error = "error"
    Centred = "centred"
    Range = "range"
    Value = "value"
    All = "all"


class GroupType(Enum):
    Error = "error"
    View = "view"
    Summary = "summary"


@dataclass
class ProcessArgs(ConfigFileArgs, InteractiveArgs):
    in_path: Path = arg_field(
        "-i",
        "--input",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The nxs file to process.",
        default=None,
        type=FilePathType(must_exist=True),
    )

    out_dir: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The file into which to place the requested slices.",
        default=None,
        type=FilePathType(must_exist=False),
    )

    check: bool = arg_field(
        "--check",
        action="store_true",
        help="If present will check that all the arguments are well formed and prints out a summary.",
    )

    default_group_type: GroupType = arg_field(
        "--default-type",
        arg_type=ArgType.EXPLICIT_ONLY,
        default=GroupType.Error,
        help="The default type for groups: --type (view | summary | error)",
    )
    default_paths: list[str] = arg_field(
        "--default-paths",
        nargs="+",
        default_factory=list,
        help="The default paths to the data groups: --default-path DATA_PATH+",
        metavar="DATA_PATH",
    )
    default_action: ActionType = arg_field(
        "--default-action",
        default=ActionType.Error,
        help="The default action to apply to axes: --default-action (leave | loop | sum | max | error)",
    )
    default_slice: list[str] = arg_field(
        "--default-slice",
        nargs="+",
        default_factory=lambda: ["error"],
        help="The default slice to apply axes: --default-slice (centred CENTRE WIDTH | range LOWER UPPER | all | error)",
        metavar="slice-type VAR1 VAR2",
    )

    group_type: list[list[str]] = arg_field(
        "--type",
        arg_type=ArgType.EXPLICIT_ONLY,
        nargs=2,
        action="append",
        default_factory=list,
        help="The type of the group: --type GROUP (view | summary)",
        metavar=("GROUP", "group-type"),
    )
    paths: list[list[str]] = arg_field(
        "--paths",
        nargs="+",
        action="append",
        default_factory=list,
        help="The paths to the data groups to use for this group: --path GROUP DATA_PATH+",
        metavar="GROUP DATA_PATH+",
    )
    action: list[list[str]] = arg_field(
        "--action",
        nargs=3,
        action="append",
        default_factory=list,
        help="The action to apply to this axis of the group: --action GROUP AXIS (leave | loop | sum | max)",
        metavar=("GROUP", "AXIS", "action-type"),
    )
    slice: list[list[str]] = arg_field(
        "--slice",
        nargs="+",
        action="append",
        default_factory=list,
        help="The slice to this axis of the group: --slice GROUP AXIS (centred CENTRE WIDTH | range LOWER UPPER | value VALUE | all)",
        metavar="GROUP AXIS slice-type VARS+",
    )

    plot_spectrum: bool = arg_field(
        "--no-plot-spectrum",
        "--no-plot-1d",
        arg_type=ArgType.EXPLICIT_ONLY,
        action="store_false",
        help="Do not plot images for 1D views and summaries.",
    )

    plot_image: bool = arg_field(
        "--no-plot-image",
        "--no-plot-2d",
        arg_type=ArgType.EXPLICIT_ONLY,
        action="store_false",
        help="Do not plot images for 2D views and summaries.",
    )

    field_options: FieldOptions = no_arg_field(
        default=FieldOptions(
            compression=hdf5plugin.Blosc(),
            compression_opts=None,
            max_bytes_per_chunk=-1,
            shuffle=True,
        ),
    )


@dataclass
class AxisParams:
    action_type: ActionType = ActionType.Error
    slice_type: SliceType = SliceType.Error
    slice_var1: float = np.nan
    slice_var2: float = np.nan
    index: list[int] = field(default_factory=list)

    def apply_default(self, default: "AxisParams") -> None:
        if self.action_type is ActionType.Error:
            self.action_type = default.action_type

        if self.slice_type is SliceType.Error:
            self.slice_type = default.slice_type
            self.slice_var1 = default.slice_var1
            self.slice_var2 = default.slice_var2

    def validate(self, group: str, axis: str) -> None:
        if self.action_type is ActionType.Error:
            raise MissingArgumentError(argument_name="action", group=group, axis=axis)
        match self.slice_type:
            case SliceType.Error:
                raise MissingArgumentError(
                    argument_name="slice", group=group, axis=axis
                )
            case SliceType.Centred | SliceType.Range:
                if np.isnan(self.slice_var1) or np.isnan(self.slice_var2):
                    raise InvalidArgumentError(
                        argument_name="slice",
                        group=group,
                        axis=axis,
                        error=f"The {self.slice_type} values are not valid: {self.slice_var1}, {self.slice_var2}.",
                    )
            case SliceType.Value:
                if np.isnan(self.slice_var1) or not np.isnan(self.slice_var2):
                    raise InvalidArgumentError(
                        argument_name="slice",
                        group=group,
                        axis=axis,
                        error=f"The {self.slice_type} values are not valid: {self.slice_var1}, {self.slice_var2}.",
                    )
            case SliceType.All:
                if not np.isnan(self.slice_var1) or not np.isnan(self.slice_var2):
                    raise InvalidArgumentError(
                        argument_name="slice",
                        group=group,
                        axis=axis,
                        error=f"The {self.slice_type} values are not valid: {self.slice_var1}, {self.slice_var2}.",
                    )
            case _:
                raise InvalidArgumentError(
                    argument_name="slice",
                    group=group,
                    axis=axis,
                    error=f"Unknown slice type {self.slice_type}.",
                )
        if len(self.index) == 0:
            raise ValueError(f"The indices were not specified for '{axis}'")
        if len(self.index) > 1:
            raise InvalidArgumentError(
                argument_name="axis",
                group=group,
                axis=axis,
                error="Operations are not supported on n-dimensional axis",
            )


@dataclass
class GroupParams:
    group_type: GroupType = GroupType.Error
    paths: list[str] = field(default_factory=list)
    axes: dict[str, AxisParams] = field(default_factory=dict)
    shape: Shape = tuple()
    slice_axes: list[str] = field(default_factory=list)

    def apply_default(self, default: "GroupParams", fle: h5py.File) -> None:
        if self.group_type is GroupType.Error:
            self.group_type = default.group_type

        if len(self.paths) == 0:
            self.paths = default.paths

        for axis in self.axes.values():
            axis.apply_default(default.axes["default"])

        axis_names = []
        axis_indices = []
        first_path: str | None = None
        first_shape: Shape | None = None
        first_default_axes: list[str] | None = None
        for path in self.paths:
            if path not in fle:
                raise ValueError(f"Expected {path} in data.")
            signal_name = fle[path].attrs["signal"]

            if first_shape is None or first_default_axes is None:
                first_shape = fle[path][signal_name].shape
                first_default_axes = fle[path].attrs["axes"]
                if len(first_shape) != len(first_default_axes):
                    raise NeXusError(
                        f"The default axes has a different number of dimensions to the signal on {path}"
                    )
            elif first_shape != fle[path][signal_name].shape:
                raise ValueError(
                    f"Signal shapes do not match: {path}: {fle[path][signal_name].shape} but {first_path}: {first_shape}"
                )
            elif any(
                first != current
                for first, current in zip(
                    first_default_axes, fle[path].attrs["axes"], strict=True
                )
            ):
                raise ValueError(
                    f"Default axes do not match: {path}: {fle[path].attrs['axes']} but {first_path}: {first_default_axes}"
                )

            for name in fle[path]:
                if f"{name}_indices" in fle[path].attrs:
                    if first_path is None:
                        axis_names.append(name)
                        axis_indices.append(fle[path].attrs[f"{name}_indices"])
                    elif name not in axis_names:
                        raise ValueError(
                            f"Found axis {name} in {path} but not in {first_path}"
                        )
            if first_path is None:
                first_path = path
            else:
                for name in axis_names:
                    if f"{name}_indices" not in fle[path].attrs:
                        raise ValueError(
                            "axis",
                            f"Did not find axis {name} in {path} but was present in {first_path}",
                        )

        if first_shape is not None:
            self.shape = first_shape

        dim_has_axis: list[str | None] = [None for _ in self.shape]
        for name, index in zip(axis_names, axis_indices, strict=True):
            if name not in self.axes:
                continue
            self.axes[name].index = index
            if len(index) == 1:
                dim_has_axis[index[0]] = name

        if first_default_axes is not None:
            for index, (present, name) in enumerate(
                zip(dim_has_axis, first_default_axes, strict=True)
            ):
                if present is None:
                    assert name not in self.axes
                    self.axes[name] = copy.copy(default.axes["default"])
                    self.axes[name].index = [index]
                    self.slice_axes.append(name)
                else:
                    self.slice_axes.append(present)

    def validate(self, group: str) -> None:
        if self.group_type is GroupType.Error:
            raise MissingArgumentError(
                argument_name="group type", group=group, axis=None
            )
        if len(self.paths) == 0:
            raise MissingArgumentError(argument_name="paths", group=group, axis=None)

        dim_has_axis = [None for _ in self.shape]
        for name, axis in self.axes.items():
            axis.validate(group, name)
            previous_axis_on_dim = dim_has_axis[axis.index[0]]
            if previous_axis_on_dim is None:
                dim_has_axis[axis.index[0]] = name
            else:
                raise InvalidArgumentError(
                    argument_name="axis",
                    group=group,
                    axis=f"{name} and {previous_axis_on_dim}",
                    error=f"Performing actions over multiple axis for the same dimension ({axis.index[0]}) is not supported.",
                )

    def calculate_chunk(self, fle: h5py.File) -> Chunk:
        chunk = Chunk()
        for ii in range(len(self.shape)):
            axis_name = self.slice_axes[ii]
            axis = self.axes[axis_name]
            slc: slice
            values = fle[self.paths[0]][axis_name][:]
            match axis.slice_type:
                case SliceType.Error:
                    raise RuntimeError("Error slice type")
                case SliceType.All:
                    slc = slice(0, len(values))
                case SliceType.Centred:
                    start_value = axis.slice_var1 - axis.slice_var2 / 2
                    stop_value = axis.slice_var1 + axis.slice_var2 / 2
                    start = bisect.bisect_left(values, start_value)
                    stop = bisect.bisect_left(values, stop_value)
                    slc = slice(start, stop)
                case SliceType.Range:
                    start_value = axis.slice_var1
                    stop_value = axis.slice_var2
                    start = bisect.bisect_left(values, start_value)
                    stop = bisect.bisect_left(values, stop_value)
                    slc = slice(start, stop)
                case SliceType.Value:
                    start_value = axis.slice_var1
                    start = bisect.bisect_left(values, start_value)
                    if start == len(values):
                        raise IndexError(
                            f"The value {start_value} is out of bounds of {axis_name}."
                        )
                    slc = slice(start, start + 1)
                case _:
                    raise RuntimeError(f"Unknown slice type: {axis.slice_type}")
            chunk.append(slc)

        return chunk

    def assemble_final_axes(self, fle: h5py.File, chunk: Chunk) -> NxAxes:
        path = self.paths[0]
        dimension_actions = [self.axes[name].action_type for name in self.slice_axes]

        axes = NxAxes(
            [[] for action in dimension_actions if action == ActionType.Leave]
        )
        for name in fle[path]:
            if f"{name}_indices" in fle[path].attrs:
                indices = fle[path].attrs[f"{name}_indices"]
                use_axes = all(
                    dimension_actions[ii] == ActionType.Leave for ii in indices
                )
                if not use_axes:
                    logger.info(
                        f"Did not put axis '{name}' into the output: One or more of its dimensions have been aggrigated."
                    )
                    continue

                offset = 0
                new_indices = []
                for ii, action in enumerate(dimension_actions):
                    if action != ActionType.Leave:
                        offset += 1
                    if ii in indices:
                        new_indices.append(ii - offset)

                dataset = fle[path][name]
                values = dataset[*[chunk[ii] for ii in indices]]
                primary_axis = new_indices[-1]

                unit = dataset.attrs.get("unit", None)
                chunks = dataset.chunks

                axes[primary_axis].append(
                    NxAxis.create(
                        values,
                        name=name,
                        indices=indices,
                        units=unit,
                        chunk_shape=chunks,
                    )
                )

        default_axes = [
            ax
            for ii, ax in enumerate(fle[path].attrs["axes"])
            if dimension_actions[ii] == ActionType.Leave
        ]

        for ii, name in enumerate(default_axes):
            dim_axes = axes[ii]
            for jj, dim_axis in enumerate(dim_axes):
                if dim_axis.name == name:
                    if jj > 0:
                        axes[ii].insert(0, axes[ii].pop(jj))
                    break
        assert axes.default_list() == default_axes
        return axes


def parse_slice(slice_arg: list[str]) -> tuple[SliceType, float, float]:
    if not (1 <= len(slice_arg) <= 3):
        raise TypeError("Expected 'slice' path to have between 1 and 3 arguments.")

    try:
        slice_type = SliceType(slice_arg[0])
    except ValueError:
        raise TypeError(
            f"Slice Type '{slice_arg[0]}' is not a valid slice type."
        ) from None

    slice_var1 = np.nan
    slice_var2 = np.nan

    match slice_type:
        case SliceType.Centred | SliceType.Range:
            if len(slice_arg) != 3:
                raise TypeError(
                    f"Slice ({' '.join(slice_arg)}) should have 3 arguments."
                )
            slice_var1 = float(slice_arg[1])
            slice_var2 = float(slice_arg[2])
        case SliceType.Value:
            if len(slice_arg) != 2:
                raise TypeError(
                    f"Slice ({' '.join(slice_arg)}) should have 2 arguments."
                )
            slice_var1 = float(slice_arg[1])
        case SliceType.All | SliceType.Error:
            if len(slice_arg) != 1:
                raise TypeError(
                    f"Slice ({' '.join(slice_arg)}) should have 1 argument."
                )
    return slice_type, slice_var1, slice_var2


def parse_default_args(args: ProcessArgs) -> GroupParams:
    default_params = GroupParams()
    default_params.group_type = args.default_group_type
    default_params.paths = args.default_paths
    default_params.axes["default"] = AxisParams()
    default_params.axes["default"].action_type = args.default_action
    if len(args.default_slice) == 0:
        slice_params = (SliceType.Error, np.nan, np.nan)
    else:
        slice_params = parse_slice(args.default_slice)
    default_params.axes["default"].slice_type = slice_params[0]
    default_params.axes["default"].slice_var1 = slice_params[1]
    default_params.axes["default"].slice_var2 = slice_params[2]

    return default_params


def parse_group_args(args: ProcessArgs) -> dict[str, GroupParams]:

    results: dict[str, GroupParams] = {}

    for group_type_arg in args.group_type:
        if len(group_type_arg) != 2:
            raise TypeError("Expected group type to have 2 args: GROUP group-type")
        if group_type_arg[0] not in results:
            results[group_type_arg[0]] = GroupParams()
        try:
            results[group_type_arg[0]].group_type = GroupType(group_type_arg[1])
        except ValueError:
            raise TypeError(
                f"Group Type '{group_type_arg[0]}' is not a valid group type."
            ) from None

    for path_arg in args.paths:
        if len(path_arg) < 2:
            raise TypeError("Expected path to have at least 2 args: GROUP path+.")
        group = path_arg[0]
        if group not in results:
            results[group] = GroupParams()
        results[group].paths = path_arg[1:]

    for action_arg in args.action:
        if len(action_arg) != 3:
            raise TypeError("Expected action to have 3 args: GROUP AXIS action-type.")
        group = action_arg[0]
        if group not in results:
            results[group] = GroupParams()
        axis = action_arg[1]
        if axis not in results[group].axes:
            results[group].axes[axis] = AxisParams()
        try:
            results[group].axes[axis].action_type = ActionType(action_arg[2])
        except ValueError:
            raise TypeError(
                f"Action Type '{action_arg[2]}' is not a valid action type."
            ) from None

    for slice_arg in args.slice:
        if len(slice_arg) < 3:
            raise TypeError("Expected slice to have at least 3 args: GROUP AXIS ...")
        group = slice_arg[0]
        if group not in results:
            results[group] = GroupParams()
        axis = slice_arg[1]
        slice_params = parse_slice(slice_arg[2:])
        if axis not in results[group].axes:
            results[group].axes[axis] = AxisParams()
        results[group].axes[axis].slice_type = slice_params[0]
        results[group].axes[axis].slice_var1 = slice_params[1]
        results[group].axes[axis].slice_var2 = slice_params[2]

    return results


def process(args: ProcessArgs, config: dict[str, Any]) -> None:

    default_params = parse_default_args(args)
    all_params = parse_group_args(args)

    with h5py.File(args.in_path, "r") as fle:
        for group, params in all_params.items():
            params.apply_default(default_params, fle)
            params.validate(group)

        for group, params in all_params.items():
            nxs = NexusFile(args.out_dir / f"{group}.nxs", mode="w")
            with nxs.as_context():
                chunk = params.calculate_chunk(fle)
                min_chunk_count = 0
                min_path: str | None = None
                for path in params.paths:
                    signal_name = fle[path].attrs["signal"]
                    chunks = fle[path][signal_name].chunks
                    chunk_count = np.prod(count_chunks_to_cover(chunk.shape, chunks))
                    if min_path is None or chunk_count < min_chunk_count:
                        min_path = path
                        min_chunk_count = chunk_count

                signal_name = fle[min_path].attrs["signal"]
                hdf_data = fle[min_path][signal_name]
                chunked_data = da.from_array(hdf_data, chunks=hdf_data.chunks)[*chunk]
                if 0 in chunked_data.shape:
                    raise IndexError("The provided slices have resulted in zero data")
                    chunked_data.reshape(tuple(max(1, ss) for ss in chunked_data.shape))

                actions = [
                    params.axes[axis_name].action_type
                    for axis_name in params.slice_axes
                ]

                processes_data = None
                if ActionType.Sum in actions and ActionType.Max in actions:
                    raise NotImplementedError(
                        "Summing and taking the Maximum cannot both be done to a single group."
                    )

                if ActionType.Sum in actions:
                    axes = tuple(
                        [ii for ii, act in enumerate(actions) if act == ActionType.Sum]
                    )
                    processes_data = da.sum(chunked_data, axis=axes)
                elif ActionType.Max in actions:
                    axes = tuple(
                        [ii for ii, act in enumerate(actions) if act == ActionType.Sum]
                    )
                    processes_data = da.max(chunked_data, axis=axes)
                else:
                    processes_data = chunked_data

                final_axes = params.assemble_final_axes(fle, chunk)

                loop_axes = [
                    ii for ii, act in enumerate(actions) if act == ActionType.Loop
                ]
                if len(loop_axes) > 0:
                    names = [params.slice_axes[ii] for ii in loop_axes]
                    values = [
                        fle[min_path][name][chunk[ii]]
                        for ii, name in zip(loop_axes, names, strict=True)
                    ]
                    for parts in itertools.product(
                        *[range(chunk.shape[ii]) for ii in loop_axes]
                    ):
                        ii = 0
                        slc = []
                        for jj in range(len(params.slice_axes)):
                            if jj in loop_axes:
                                slc.append(parts[ii])
                                ii += 1
                            else:
                                slc.append(slice(None))
                        assert ii == len(parts)

                        loop_data = processes_data[*slc]

                        loop_values = [values[ii][p] for ii, p in enumerate(parts)]

                        name = "-".join(
                            f"{name}_{value:.3g}"
                            for name, value in zip(names, loop_values, strict=True)
                        )
                        nxs.root[name] = NXsubentry(
                            NXdata(
                                signal=create_field(
                                    dtype=loop_data.dtype,
                                    shape=loop_data.shape,
                                    compression=args.field_options.compression,
                                    compression_opts=args.field_options.compression_opts,
                                    chunks=simplify_chunks(loop_data.chunks),
                                    shuffle=args.field_options.shuffle,
                                    fillvalue=0,
                                    name=signal_name,
                                ),
                            ),
                        )
                        loop_data.store(nxs.root[f"{name}/data/{signal_name}"])
                        nxs.root[name].attrs["default"] = "data"
                        final_axes.add_to_group(nxs.root[f"{name}/data/"])
                else:
                    if processes_data.shape == ():
                        shape = (1,)
                        chunks = (1,)
                    else:
                        shape = processes_data.shape
                        chunks = simplify_chunks(processes_data.chunks)
                    chunks = tuple(
                        min(s, c) for s, c in zip(shape, chunks, strict=True)
                    )

                    nxs.root["data"] = NXdata(
                        signal=create_field(
                            dtype=processes_data.dtype,
                            shape=shape,
                            compression=args.field_options.compression,
                            compression_opts=args.field_options.compression_opts,
                            chunks=chunks,
                            shuffle=args.field_options.shuffle,
                            fillvalue=0,
                            name=signal_name,
                        ),
                    )
                    processes_data.store(nxs.root[f"data/{signal_name}"])
                    final_axes.add_to_group(nxs.root["data/"])
                    nxs.root.attrs["default"] = "data"

                    if params.group_type == GroupType.View:
                        indices = tuple(ii for ii, s in enumerate(shape) if s > 1)
                        new_shape = tuple(s for s in shape if s > 1)
                        shaped_data = processes_data.reshape(new_shape)
                        ndim = len(shaped_data.shape)
                        if ndim == 2 and args.plot_image:
                            fig, ax = plt.subplots()
                            imshow_sparse(
                                ax,
                                shaped_data.compute(),
                                final_axes[indices[0]][0].field.nxdata,
                                final_axes[indices[1]][0].field.nxdata,
                            )
                            ax.set_xlabel(final_axes[indices[0]][0].name)
                            ax.set_ylabel(final_axes[indices[1]][0].name)
                            fig.savefig(args.out_dir / f"{group}.2d.png")
                        elif ndim == 1 and args.plot_spectrum:
                            fig, ax = plt.subplots()
                            plot_spectrum(
                                ax,
                                shaped_data,
                                final_axes[indices[0]][0].field.nxdata,
                            )
                            ax.set_xlabel(final_axes[indices[0]][0].name)
                            ax.set_ylabel(signal_name)
                            fig.savefig(args.out_dir / f"{group}.1d.png")
