# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from typing import Any, Iterable, Generator, cast
from threading import Lock, local
import concurrent.futures as cfutures
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import numpy.typing as npt
import sparse
from h5py import h5s

from tqdm import tqdm

import hdf5plugin
from nexusformat.nexus import NXsubentry, NXdata

from argsui import (
    no_arg_field,
    arg_field,
    ArgType,
    ConfigFileArgs,
    InteractiveArgs,
    FilePathType,
)
from ..lib.data_source import (
    AbstractDataSource,
    AxisType,
    Axis,
    Signal,
)
from ..lib.exceptions import InvalidAxisError
from ..lib.multi_coo import (
    MultiCOO,
)
from ..lib.bounds import Chunk, Shape
from ..lib.chunker import Chunker, count_chunks_to_cover
from ..lib.nxs import (
    NexusFile,
    FieldOptions,
    NxAxis,
    create_field,
    NxAxes,
    create_group,
)
from ..lib.utils import format_bytes
from ..lib.dtypes import Intp1D, Bool1D


def _count_subentry_name() -> str:
    return "item_counts"


def _items(d: dict[str, Any]) -> list[tuple[str, Any]]:
    return list(d.items())


@dataclass
class ProcessArgs(
    ConfigFileArgs,
    InteractiveArgs,
):
    in_path: Path = arg_field(
        "-i",
        "--input",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The file to process.",
        default=None,
        type=FilePathType(must_exist=True),
    )

    out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The directory to place the requested images and spectra.",
        default=None,
        type=FilePathType(must_exist=False),
    )

    chunk_max_byte_count: int = arg_field(
        "--chunk-bytes",
        help="The maximum number of bytes of a chunk in the output file.",
        default=1024 * 1024 * 8,
    )  # 8Mb

    memory_max_byte_count: int = arg_field(
        "--memory-bytes",
        help="The maximum number of bytes to use as the memory buffer. Each thread uses this much memory.",
        default=1024 * 1024 * 1024 * 4,
    )

    data_source: AbstractDataSource = no_arg_field(default=None)

    field_options: FieldOptions = no_arg_field(
        default=FieldOptions(
            compression=hdf5plugin.Blosc(),
            compression_opts=None,
            max_bytes_per_chunk=-1,
            shuffle=True,
        ),
    )


class DataChunks:
    def __init__(
        self,
        names: Iterable[str],
        chunkers: Iterable[Chunker],
        definitions: Iterable[Signal],
    ):
        self.names: set[str] = set(names)
        self.chunkers: dict[str, Chunker] = dict(zip(names, chunkers, strict=True))
        self.definitions: dict[str, Signal] = dict(zip(names, definitions, strict=True))

    def __setitem__(self, name: str, data: tuple[Chunker, Signal]) -> None:
        self.names.add(name)
        self.chunkers[name] = data[0]
        self.definitions[name] = data[1]

    def __repr__(self) -> str:
        return ",\n".join(
            [f"{name} ({dtype}): {chunker}" for name, chunker, dtype in self.items()],
        )

    def chunker(self, name: str) -> Chunker:
        return self.chunkers[name]

    def signal(self, name: str) -> Signal:
        return self.definitions[name]

    def items(self) -> Generator[tuple[str, Chunker, Signal]]:
        for name in self.names:
            yield name, self.chunker(name), self.signal(name)


def choose_memory_buffer(
    args: ProcessArgs,
    max_item_count: int,
    density: float,
    data_chunks: DataChunks,
) -> tuple[Chunker, int, str]:
    memory_chunks = {
        name: Chunker.find_chunk_multiple(
            chunker.data_shape,
            chunker.chunk_shape,
            max_item_count / density,
            priorities=args.data_source.read_chunks(),
        )
        for name, chunker, _ in data_chunks.items()
    }

    min_read_count = np.pow(2, 32, dtype=np.int64)
    min_read_name = ""
    for name, chunker in memory_chunks.items():
        if name == _count_subentry_name():
            continue
        chunker.normalise()
        read_count = 0
        for chunk in chunker.chunks():
            read_count += args.data_source.chunk_read_count(chunk.shape)
        if read_count < min_read_count:
            min_read_count = read_count
            min_read_name = name
    assert len(min_read_name) > 0

    return memory_chunks[min_read_name], min_read_count, min_read_name


def choose_memory_buffer_and_data_chunks(
    args: ProcessArgs,
    full_shape: Shape,
    density: float,
) -> tuple[Chunker, int, int, DataChunks, str]:
    data_priorities = args.data_source.output_chunks()
    if len(data_priorities) == 0:
        raise ValueError("At least one dataset must be provided.")
    for name, priorities in data_priorities.items():
        if len(name.strip()) == 0:
            raise ValueError(
                "An invalid name was returned for a data set. Names must not be empty.",
            )
        if len(priorities) != len(full_shape):
            raise ValueError(
                f"An invalid set of priorities was returned for dataset {name}: there should be {len(full_shape)} items, but only {len(priorities)} were provided.",
            )
    signal_definition = args.data_source.signal_definition()
    signal_item_width = np.dtype(signal_definition.dtype).itemsize
    data_max_items = {
        name: int(args.field_options.max_bytes_per_chunk / signal_item_width)
        for name in data_priorities
    }

    data_chunks = DataChunks([], [], [])

    for name, priorities in data_priorities.items():
        chunker = Chunker.from_max_item_count(
            data_shape=full_shape,
            priorities=priorities,
            items_per_chunk=data_max_items[name],
        )
        chunker.normalise()
        data_chunks[name] = (
            chunker,
            signal_definition,
        )

    axis_definitions = args.data_source.axis_definitions()
    if any(ax.axis_type == AxisType.BINNED for ax in axis_definitions):
        counts_item_width = np.dtype(np.uint16).itemsize
        data_max_items[_count_subentry_name()] = int(
            args.field_options.max_bytes_per_chunk / counts_item_width,
        )
        chunker = Chunker.from_max_item_count(
            data_shape=full_shape,
            priorities=tuple(1 for _ in full_shape),
            items_per_chunk=data_max_items[_count_subentry_name()],
        )
        chunker.normalise()
        data_chunks[_count_subentry_name()] = (
            chunker,
            Signal("items_per_bin", np.uint16, "items"),
        )

    size_per_item = signal_item_width + np.sum(
        [
            np.dtype(ax.dtype).itemsize
            for ax in axis_definitions
            if ax.axis_type == AxisType.BINNED
        ],
    )
    memory_max_item_count = int(args.memory_max_byte_count / size_per_item)
    memory_chunks, total_read_count, min_read_name = choose_memory_buffer(
        args,
        memory_max_item_count,
        density,
        data_chunks,
    )
    for name, chunker, dtype in data_chunks.items():
        chunker = Chunker.from_max_item_count(
            data_shape=chunker.data_shape,
            priorities=chunker.priorities,
            items_per_chunk=data_max_items[name],
            min_chunk_count=memory_chunks.chunk_count,
        )
        chunker.normalise()
        data_chunks[name] = (
            chunker,
            dtype,
        )

    return memory_chunks, total_read_count, size_per_item, data_chunks, min_read_name


def provision_subentries(
    nxs: NexusFile,
    args: ProcessArgs,
    data_chunks: DataChunks,
    default_name: str,
) -> None:
    nxs.root.attrs["default"] = default_name

    for name, chunker, signal in data_chunks.items():
        nxs.root[name] = NXsubentry(
            NXdata(
                signal=create_field(
                    name=signal.name,
                    dtype=signal.dtype,
                    shape=chunker.data_shape,
                    compression=args.field_options.compression,
                    compression_opts=args.field_options.compression_opts,
                    chunks=chunker.chunk_shape,
                    shuffle=args.field_options.shuffle,
                    fillvalue=0,
                ),
            ),
        )
        nxs.root[name].attrs["default"] = "data"


def provision_data_axis(
    nxs: NexusFile,
    args: ProcessArgs,
    full_shape: Shape,
    data_chunks: DataChunks,
) -> tuple[dict[str, Axis], bool]:
    axis_definitions = args.data_source.axis_definitions()

    any_binned_axis = False
    for entry_name, chunker, _ in data_chunks.items():
        group_axes = NxAxes()
        for _ in full_shape:
            group_axes.append([])
        for axis in axis_definitions:
            match axis.axis_type:
                case AxisType.EXACT:
                    values = args.data_source.exact_axis_values(axis)
                    if len(values) != full_shape[axis.primary_axis]:
                        raise InvalidAxisError(
                            f"Expected {full_shape[axis.primary_axis]} values for {axis.name} but received {len(values)}.",
                        )
                    if values.dtype != axis.dtype:
                        raise TypeError(
                            f"Expected {axis.dtype} values for {axis.name} but found {values.dtype}"
                        )

                    nx_axis = NxAxis.create(
                        values=values,
                        name=axis.name,
                        indices=[axis.primary_axis],
                        unit=axis.units,
                    )
                    group_axes[axis.primary_axis].append(nx_axis)
                case AxisType.BINNED:
                    any_binned_axis = True
                    values = args.data_source.binned_axis_edges(axis)[1:]
                    if len(values) != full_shape[axis.primary_axis]:
                        raise InvalidAxisError(
                            f"Expected {full_shape[axis.primary_axis] + 1} edges for {axis.name} but received {len(values) + 1}.",
                        )
                    if values.dtype != axis.dtype:
                        raise TypeError(
                            f"Expected {axis.dtype} values for {axis.name} but found {values.dtype}"
                        )

                    nx_axis = NxAxis.create(
                        values=values,
                        name=axis.name,
                        indices=[axis.primary_axis],
                        unit=axis.units,
                    )
                    if entry_name != _count_subentry_name():
                        nx_axis_exact = NxAxis.create_empty(
                            name=f"{axis.name}_exact",
                            indices=list(range(len(chunker.data_shape) + 1)),
                            unit=axis.units,
                            shape=chunker.data_shape,
                            compression=args.field_options.compression,
                            compression_opts=args.field_options.compression_opts,
                            chunks=chunker.chunk_shape,
                            dtype=axis.dtype,
                            fillvalue=0
                            if np.issubdtype(axis.dtype, np.integer)
                            else np.nan,
                        )
                        group_axes[axis.primary_axis].extend([nx_axis, nx_axis_exact])
                    else:
                        group_axes[axis.primary_axis].append(nx_axis)
                case _:
                    raise InvalidAxisError(f"Unknown Axis type: {axis.axis_type}")
        group_axes.add_to_group(nxs.root[entry_name]["data"])

    return {ax.name: ax for ax in axis_definitions}, any_binned_axis


@dataclass
class Accumulation:
    name: str
    axis: tuple[int, ...]
    axis_edges: list[None | np.ndarray]
    shape: Shape

    contains_binned_axes: bool = field(init=False)
    max_data: np.ndarray = field(init=False)
    sum_data: np.ndarray = field(init=False)
    ndim: int = field(init=False)
    has_data: bool = field(init=False)

    def __post_init__(self):

        self.contains_binned_axes = False
        for edge in self.axis_edges:
            if edge is not None:
                self.contains_binned_axes = True
                break

        self.max_data = np.zeros(self.shape)
        self.sum_data = np.zeros(self.shape)
        self.ndim = len(self.shape)
        self.has_data = False

    def add(self, data: np.ndarray | sparse.COO, chunk: Chunk) -> None:
        sub_chunk = Chunk([chunk[ii] for ii in range(data.ndim) if ii not in self.axis])

        sub_data = data[*chunk] if isinstance(data, sparse.COO) else data

        max_data = np.maximum(self.max_data[*sub_chunk], sub_data.max(axis=self.axis))
        sum_data = np.add(self.sum_data[*sub_chunk], sub_data.sum(axis=self.axis))

        if isinstance(max_data, sparse.COO):
            self.max_data[*sub_chunk] = max_data.todense()
        else:
            self.max_data[*sub_chunk] = max_data

        if isinstance(sum_data, sparse.COO):
            self.sum_data[*sub_chunk] = sum_data.todense()
        else:
            self.sum_data[*sub_chunk] = sum_data

        self.has_data = True


def provision_accumulation_subentries(
    nxs: NexusFile,
    args: ProcessArgs,
    shape: Shape,
    axis_definitions: dict[str, Axis],
) -> tuple[dict[str, Accumulation], dict[str, Accumulation]]:
    accumulations = args.data_source.output_accumulations()
    signal = args.data_source.signal_definition()

    final_accumulations: dict[str, Accumulation] = {}
    count_accumulations: dict[str, Accumulation] = {}

    for ac_name, axes in accumulations.items():
        axis_to_accumulate: Bool1D = np.full(shape=(len(shape),), fill_value=False)
        edges = []

        group_axes = NxAxes()
        group_axes.append(
            [
                NxAxis.create(
                    values=["sum", "max"],
                    name="accumulator",
                    indices=[0],
                    unit="",
                ),
            ],
        )
        has_binned_axis = False

        for ax_name, ax in axis_definitions.items():
            if ax_name in axes:
                axis_to_accumulate[ax.primary_axis] = True

        final_dim = np.sum(axis_to_accumulate == False)  # noqa: E712
        acc_shape: Intp1D = np.zeros((final_dim,), dtype=np.intp)
        for _ in range(final_dim):
            group_axes.append([])

        for ax in axis_definitions.values():
            if not axis_to_accumulate[ax.primary_axis]:
                match ax.axis_type:
                    case AxisType.EXACT:
                        values = args.data_source.exact_axis_values(ax)
                        edges.append(None)
                    case AxisType.BINNED:
                        values = args.data_source.binned_axis_edges(ax)[1:]
                        edges.append(values)
                        has_binned_axis = True
                    case _:
                        raise InvalidAxisError(f"Unknown Axis type: {ax.axis_type}")
                count = len(values)
                new_index = ax.primary_axis - np.sum(
                    axis_to_accumulate[0 : ax.primary_axis],
                )
                if acc_shape[new_index] == 0:
                    acc_shape[new_index] = count
                elif count != acc_shape[new_index]:
                    raise InvalidAxisError(
                        f"Found conflicting sizes for {new_index}. Initially set to {acc_shape[new_index]} now trying to set to {count}",
                    )

                nx_axis = NxAxis.create(
                    values=values,
                    name=ax.name,
                    indices=[cast(int, new_index + 1)],
                    unit=ax.units,
                )
                group_axes[new_index + 1].append(nx_axis)

        final_axis = tuple(ii for ii, aa in enumerate(axis_to_accumulate) if aa)
        final_accumulations[ac_name] = Accumulation(
            name=ac_name,
            axis=final_axis,
            axis_edges=edges,
            shape=Shape(acc_shape.tolist()),
        )

        nxs.root[ac_name] = NXsubentry(
            NXdata(
                signal=create_field(
                    dtype=signal.dtype,
                    shape=(2, *acc_shape),
                    compression=args.field_options.compression,
                    compression_opts=args.field_options.compression_opts,
                    chunks=None,
                    shuffle=args.field_options.shuffle,
                    fillvalue=0,
                ),
            ),
        )
        group_axes.add_to_group(nxs.root[ac_name]["data"])
        if has_binned_axis:
            counts_name = f"{_count_subentry_name()}_{ac_name}"
            count_accumulations[counts_name] = Accumulation(
                name=counts_name,
                axis=final_axis,
                axis_edges=edges,
                shape=Shape(acc_shape.tolist()),
            )
            nxs.root[counts_name] = NXsubentry(
                NXdata(
                    signal=create_field(
                        dtype=np.uint16,
                        shape=(2, *acc_shape),
                        compression=args.field_options.compression,
                        compression_opts=args.field_options.compression_opts,
                        chunks=None,
                        shuffle=args.field_options.shuffle,
                        fillvalue=0,
                    ),
                ),
            )
            group_axes.add_to_group(nxs.root[counts_name]["data"])

    return final_accumulations, count_accumulations


def write_data(
    nxs: NexusFile,
    args: ProcessArgs,
    memory_chunk: Chunk,
    full_shape: Shape,
    binned_axes: list[Axis],
    chunk_data: np.ndarray | MultiCOO,
    data_chunks: DataChunks,
) -> tuple[np.ndarray, None] | tuple[sparse.COO, sparse.COO]:
    if isinstance(chunk_data, np.ndarray):
        if len(binned_axes) != 0:
            raise TypeError(
                "Received a binned axis, with dense data. The data should be sparse. ",
            )

        for data_entry in data_chunks.names:
            assert data_entry != _count_subentry_name()
            signal_name = data_chunks.signal(data_entry).name
            nxs.root[data_entry].data[signal_name][*memory_chunk] = chunk_data
        return chunk_data, None

    if chunk_data.coords.shape[1] == 0:
        null = sparse.COO(coords=[], data=[], shape=full_shape)
        return null, null

    chunk_data.sort(full_shape)
    counts = chunk_data.acc_duplicates(
        full_shape,
        count=True,
        accumulators={"signal": np.add},
        default_accumulator=np.maximum,
    )
    assert counts is not None

    for name, value in chunk_data.values.items():
        if value.shape != counts.shape:
            raise ValueError(
                f"Expected all signals to have shape {counts.shape} but '{name}' has shape {value.shape}"
            )

    signal_data = sparse.COO(
        coords=chunk_data.coords,
        data=chunk_data["signal"],
        shape=full_shape,
        sorted=True,
        has_duplicates=False,
        prune=False,
    )

    count_data = sparse.COO(
        coords=chunk_data.coords,
        data=counts,
        shape=full_shape,
        sorted=True,
        has_duplicates=False,
        prune=False,
    )

    data_chunk_values = [
        (entry, chunker, dtype) for entry, chunker, dtype in data_chunks.items()
    ]

    axis_names = [axis.name for axis in binned_axes]
    axis_used = dict.fromkeys(axis_names, False)
    for data_entry, _, signal in tqdm(data_chunk_values, desc="Processing data chunks"):
        for name in chunk_data.values:
            if name != "signal" and name not in axis_names:
                raise IndexError(f"Values provided for {name}, but axis is not binned.")

            if name == "signal":
                ds = nxs.root[data_entry].data[signal.name]
            elif data_entry == _count_subentry_name():
                continue
            else:
                ds = nxs.root[data_entry].data[f"{name}_exact"]
                axis_used[name] = True

            f_space = ds.id.get_space()
            f_space.select_elements(chunk_data.coords.T, h5s.SELECT_SET)
            m_space = h5s.create_simple(chunk_data[name].shape)

            if data_entry == _count_subentry_name():
                ds.id.write(m_space, f_space, counts)
            else:
                ds.id.write(m_space, f_space, chunk_data[name])

            f_space.close()
            m_space.close()

    for k, v in axis_used.items():
        if not v:
            raise ValueError(f"Expected values for binned axis {k}")

    return signal_data, count_data


def accumulate_data(
    accumulations: dict[str, Accumulation],
    count_accumulations: dict[str, Accumulation],
    memory_chunk: Chunk,
    data: np.ndarray | sparse.COO,
    counts: None | sparse.COO,
) -> None:

    total = len(accumulations) + len(count_accumulations) if counts is not None else 0

    with tqdm(total=total, desc="Accumulating", leave=False) as progress:
        for accumulation in accumulations.values():
            accumulation.add(data, memory_chunk)
            progress.update()
        if counts is not None:
            for accumulation in count_accumulations.values():
                accumulation.add(counts, memory_chunk)
                progress.update()


def add_items_to_group(items: dict[str, Any], root) -> None:
    for key, value in items.items():
        if isinstance(value, dict):
            if key not in root:
                root[key] = create_group()
            add_items_to_group(value, root[key])
        else:
            root.attrs[key] = value


def process(args: ProcessArgs, config: dict[str, Any] = {}) -> None:
    assert args.in_path.exists(), f"The input file {args.in_path} was not found"

    if args.field_options.max_bytes_per_chunk <= 0:
        args.field_options = FieldOptions(
            compression=args.field_options.compression,
            compression_opts=args.field_options.compression_opts,
            max_bytes_per_chunk=args.chunk_max_byte_count,
            shuffle=args.field_options.shuffle,
        )
    else:
        assert args.field_options.max_bytes_per_chunk == args.chunk_max_byte_count

    nxs = NexusFile(args.out_path, mode="w")
    with nxs.as_context():
        add_items_to_group(args.data_source.instrument_metadata(), nxs.instrument)
        add_items_to_group(args.data_source.experiment_metadata(), nxs.experiment)

        full_shape, is_dense, density = args.data_source.shape()

        if any(s <= 0 for s in full_shape):
            raise ValueError(
                f"Invalid data shape {full_shape}. Each dimension must have at least 1 element."
            )

        memory_chunks, total_read_count, size_per_item, data_chunks, default_name = (
            choose_memory_buffer_and_data_chunks(args, full_shape, density)
        )

        provision_subentries(nxs, args, data_chunks, default_name)

        axis_definitions, any_binned_axis = provision_data_axis(
            nxs,
            args,
            full_shape,
            data_chunks,
        )

        accumulations, count_accumulations = provision_accumulation_subentries(
            nxs,
            args,
            full_shape,
            axis_definitions,
        )

        binned_axis = [
            ax for ax in axis_definitions.values() if ax.axis_type == AxisType.BINNED
        ]

        print(f"Processing file {args.in_path}")
        print(f" Writing results to {args.out_path}")
        print(
            f" Giving a final data shape of {full_shape} (Raw {format_bytes(np.prod(full_shape) * size_per_item)})",
        )

        print(
            f"Using a memory chunk shape {memory_chunks.chunk_shape} and count {memory_chunks.chunk_count}.",
        )
        print(
            f" Dense usage: {np.prod(memory_chunks.chunk_shape)} items ({format_bytes(np.prod(memory_chunks.chunk_shape) * size_per_item)}).",
        )
        if len(binned_axis) > 0:
            print(
                f" Binned usage: {int(np.prod(memory_chunks.chunk_shape) * density)} items ({format_bytes(np.prod(memory_chunks.chunk_shape) * size_per_item * density)}), worst case density {density:.2f}.",
            )
        print(f" Memory limit set at {format_bytes(args.memory_max_byte_count)}.")
        print("With data blocks:")
        print(
            f"maximum chunk size ({format_bytes(args.field_options.max_bytes_per_chunk)})",
        )
        for name, chunker, signal in data_chunks.items():
            width = np.dtype(signal.dtype).itemsize
            print(
                f"    {name: >10}: chunk shape {chunker.chunk_shape} and total count {chunker.chunk_count} and memory count {count_chunks_to_cover(memory_chunks.chunk_shape, chunker.chunk_shape)}.",
            )
            print(
                f"    {' ' * 10}: chunk size: {np.prod(chunker.chunk_shape)} items ({format_bytes(np.prod(chunker.chunk_shape) * width)}).",
            )

        with tqdm(desc="Overall reads", total=total_read_count) as overall_reads_timer:
            data_source_lock = Lock()
            accumulation_lock = Lock()
            nexus_file_lock = Lock()

            def process_chunk(memory_chunk: Chunk) -> None:
                local_store = local()
                with data_source_lock:
                    local_store.chunk_data = args.data_source.fill_chunk(
                        memory_chunk,
                        overall_reads_timer.update,
                    )

                with nexus_file_lock:
                    local_store.written_signal, local_store.written_count = write_data(
                        nxs,
                        args,
                        memory_chunk,
                        full_shape,
                        binned_axis,
                        local_store.chunk_data,
                        data_chunks,
                    )
                    del local_store.chunk_data

                with accumulation_lock:
                    accumulate_data(
                        accumulations,
                        count_accumulations,
                        memory_chunk,
                        local_store.written_signal,
                        local_store.written_count,
                    )

            outer_chunks = list(memory_chunks.chunks())
            with cfutures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(process_chunk, memory_chunk)
                    for memory_chunk in outer_chunks
                ]

                for memory_chunk in tqdm(
                    cfutures.as_completed(futures),
                    total=len(outer_chunks),
                    desc="Memory chunks",
                    leave=True,
                ):
                    try:
                        memory_chunk.result()
                    except:
                        for f in futures:
                            f.cancel()
                        raise

        for name, accumulation in tqdm(
            [*_items(accumulations), *_items(count_accumulations)],
            desc="Writing accumulations",
        ):
            if accumulation.has_data:
                extra_slices = [slice(None) for _ in range(accumulation.ndim)]
                nxs.root[name].data.signal[0, *extra_slices] = accumulation.max_data
                nxs.root[name].data.signal[1, *extra_slices] = accumulation.sum_data
