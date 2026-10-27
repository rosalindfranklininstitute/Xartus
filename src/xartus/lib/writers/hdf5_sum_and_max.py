# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from xartus.lib.exceptions import InvalidAxisError
from xartus.lib.dtypes import Bool1D, Intp1D
from xartus.lib.chunker import Chunker
from typing import override, Any, cast
import numpy.typing as npt
import numpy as np
import sparse

from .abstract_writer import AbstractWriter

from xartus.lib.bounds import Chunk, Shape
from xartus.lib.nxs import (
    NexusFile,
    create_nxgroup,
    create_nxfield,
    FieldOptions,
    NxAxes,
    NxAxis,
)
from xartus.lib.multi_coo import MultiCOO
from xartus.lib.data_source import Axis


class SumAndMax(AbstractWriter):
    def __init__(
        self,
        name: str,
        axes: list[str],
    ):
        self.name = name
        self.axes = axes

    @override
    def supports_filetype(self, typ) -> bool:
        return typ is NexusFile

    @override
    def init_chunker(
        self,
        data_shape: Shape,
        signal_dtype: npt.DTypeLike,
        max_bytes_per_chunk: int,
        axes: list[Axis],
    ) -> Chunker:

        axis_to_accumulate: Bool1D = np.full(shape=(len(data_shape),), fill_value=False)
        edges = []

        self.group_axes = NxAxes()
        self.group_axes.append(
            [
                NxAxis(
                    values=np.array(["sum", "max"]).astype("T"),
                    name="accumulator",
                    indices=[0],
                    units="",
                ),
            ],
        )
        for ax in axes:
            if ax.name in self.axes:
                axis_to_accumulate[ax.primary_axis] = True

        final_dim = np.sum(axis_to_accumulate == False)  # noqa: E712
        acc_shape: Intp1D = np.zeros((final_dim,), dtype=np.intp)
        for _ in range(final_dim):
            self.group_axes.append([])

        for ax in axes:
            if not axis_to_accumulate[ax.primary_axis]:
                edges.append(None)
                count = data_shape[ax.primary_axis]
                new_index = ax.primary_axis - np.sum(
                    axis_to_accumulate[0 : ax.primary_axis],
                )
                if acc_shape[new_index] == 0:
                    acc_shape[new_index] = count
                elif count != acc_shape[new_index]:
                    raise InvalidAxisError(
                        f"Found conflicting sizes for {new_index}. Initially set to {acc_shape[new_index]} now trying to set to {count}",
                    )

                nx_axis = NxAxis(
                    name=ax.name,
                    indices=[cast(int, new_index + 1)],
                    units=ax.units,
                    shape=(count,),
                    dtype=ax.dtype,
                )
                self.group_axes[new_index + 1].append(nx_axis)

        self.axes_inx = tuple(ii for ii, aa in enumerate(axis_to_accumulate) if aa)
        self.acc_shape = Shape(acc_shape.tolist())
        self.field_shape = (2, *self.acc_shape)
        self.data_shape = data_shape

        return Chunker.from_chunk_shape(self.field_shape, self.field_shape)

    @override
    def init_field(
        self,
        file: Any,
        data_shape: Shape,
        axes: NxAxes,
        signal_dtype: npt.DTypeLike,
        field_options: FieldOptions,
    ) -> None:
        assert isinstance(file, NexusFile)
        self.nxs = file
        if "accumulations" not in self.nxs.entry:
            data_group = create_nxgroup(
                self.nxs.entry, "accumulations", nx_class="NXsubentry"
            )
            data_group.attrs["default"] = self.name
        else:
            data_group = self.nxs.entry["accumulations"]

        acc_group = create_nxgroup(data_group, self.name, nx_class="NXdata")
        acc_group.attrs["signal"] = "signal"

        axis_dict = {ax.name: ax for dim_axes in axes for ax in dim_axes}
        for group_dim_axes in self.group_axes:
            for ax in group_dim_axes:
                if ax.name in axis_dict:
                    ax.values = axis_dict[ax.name].values

        create_nxfield(
            acc_group,
            "signal",
            dtype=signal_dtype,
            shape=(2, *self.acc_shape),
            compression=field_options.compression,
            compression_opts=field_options.compression_opts,
            chunks=None,
            shuffle=field_options.shuffle,
            fillvalue=0,
        )
        self.group_axes.add_to_group(acc_group)

        self.max_data = np.zeros(self.acc_shape)
        self.sum_data = np.zeros(self.acc_shape)
        self.has_data = False

    @override
    def parse_data(
        self,
        memory_chunk: Chunk,
        chunk_data: np.ndarray | MultiCOO,
    ) -> None:
        sub_chunk = Chunk(
            [
                memory_chunk[ii]
                for ii in range(len(memory_chunk.shape))
                if ii not in self.axes_inx
            ]
        )
        sub_data = (
            sparse.COO(
                coords=chunk_data.coords,
                data=chunk_data["signal"],
                sorted=True,
                has_duplicates=False,
                shape=self.data_shape,
            )[*memory_chunk]
            if isinstance(chunk_data, MultiCOO)
            else chunk_data
        )
        max_data = np.maximum(
            self.max_data[*sub_chunk], sub_data.max(axis=self.axes_inx)
        )
        sum_data = np.add(self.sum_data[*sub_chunk], sub_data.sum(axis=self.axes_inx))

        if isinstance(max_data, sparse.COO):
            self.max_data[*sub_chunk] = max_data.todense()
        else:
            self.max_data[*sub_chunk] = max_data

        if isinstance(sum_data, sparse.COO):
            self.sum_data[*sub_chunk] = sum_data.todense()
        else:
            self.sum_data[*sub_chunk] = sum_data

        self.has_data = True

    @override
    def finalise(self) -> None:
        if self.has_data:
            extra_slices = [slice(None) for _ in range(len(self.acc_shape))]
            acc_group = self.nxs.entry[f"accumulations/{self.name}"]
            signal_name = acc_group.attrs["signal"]
            acc_group[signal_name][0, *extra_slices] = self.sum_data
            acc_group[signal_name][1, *extra_slices] = self.max_data
