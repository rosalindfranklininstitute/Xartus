# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
import argparse
from typing import Any, Callable
from threading import Lock, local
import concurrent.futures as cfutures
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import logging

from tqdm import tqdm

import hdf5plugin

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
    Axis,
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
    NxAxes,
    create_nxgroup,
)
from ..lib.utils import format_bytes
from ..lib.writers.hdf5_signal import HDF5ChunkedSignalWriter

logger = logging.getLogger(__name__)


@dataclass
class ProcessArgs(
    ConfigFileArgs,
    InteractiveArgs,
):
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

    show_progress: bool = arg_field(
        action=argparse.BooleanOptionalAction,
        help="If true (if not specified) tqdm is used to show the progress. Otherwise no progress is shown.",
        default=True,
    )

    dry_run: bool = arg_field(
        action="store_true",
        help="If true (if specified) no conversion will be done, but all the metadata will be read and displayed.",
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


def choose_memory_buffer(
    read_priorities: Shape | list[Shape] | None,
    max_item_count: int,
    density: float,
    data_chunks: list[Chunker],
    chunk_read_count: Callable[[Shape], int],
) -> tuple[Chunker, int, int]:
    memory_chunks = [
        Chunker.find_chunk_multiple(
            chunker.data_shape,
            chunker.chunk_shape,
            max_item_count / density,
            priorities=read_priorities,
        )
        for chunker in data_chunks
    ]

    min_read_count = np.pow(2, 32, dtype=np.int64)
    min_read_inx = -1
    for inx, chunker in enumerate(memory_chunks):
        read_count = 0
        for chunk in chunker.chunks():
            read_count += chunk_read_count(chunk.shape)
        if (
            read_count == min_read_count and inx < min_read_inx
        ) or read_count < min_read_count:
            min_read_count = read_count
            min_read_inx = inx
    assert min_read_inx >= 0

    return memory_chunks[min_read_inx], min_read_count, min_read_inx


def fill_axes(
    data_source: AbstractDataSource,
    axis_definitions: list[Axis],
    full_shape: Shape,
) -> NxAxes:

    group_axes = NxAxes()
    for _ in full_shape:
        group_axes.append([])
    for axis in axis_definitions:
        values = data_source.axis_values(axis)
        if len(values) != full_shape[axis.primary_axis]:
            raise InvalidAxisError(
                f"Expected {full_shape[axis.primary_axis]} values for {axis.name} but received {len(values)}.",
            )
        if values.dtype != axis.dtype:
            raise TypeError(
                f"Expected {axis.dtype} values for {axis.name} but found {values.dtype}"
            )

        nx_axis = NxAxis(
            values=values,
            name=axis.name,
            indices=[axis.primary_axis],
            units=axis.units,
        )
        group_axes[axis.primary_axis].append(nx_axis)

    return group_axes


def prepare_data(
    full_shape: Shape,
    chunk_data: np.ndarray | MultiCOO,
) -> np.ndarray | MultiCOO:
    if isinstance(chunk_data, np.ndarray):
        return chunk_data

    if chunk_data.coords.shape[1] == 0:
        return MultiCOO(
            coords=np.array([], dtype=np.int32),
            values={"signal": np.array([]), "counts": np.array([])},
        )

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

    return MultiCOO(
        coords=chunk_data.coords,
        values={"signal": chunk_data["signal"], "counts": counts},
    )


def add_items_to_group(items: dict[str, Any], root, as_attributes=True) -> None:
    for key, value in items.items():
        if isinstance(value, dict):
            if key not in root:
                create_nxgroup(root, key)
            add_items_to_group(value, root[key])
        else:
            if as_attributes:
                root.attrs[key] = value
            else:
                root.create_dataset(key, data=value)


def process(args: ProcessArgs, config: dict[str, Any] = {}) -> None:

    if args.field_options.max_bytes_per_chunk <= 0:
        args.field_options = FieldOptions(
            compression=args.field_options.compression,
            compression_opts=args.field_options.compression_opts,
            max_bytes_per_chunk=args.chunk_max_byte_count,
            shuffle=args.field_options.shuffle,
        )
    else:
        assert args.field_options.max_bytes_per_chunk == args.chunk_max_byte_count

    full_shape, density = args.data_source.shape()

    if any(s <= 0 for s in full_shape):
        raise ValueError(
            f"Invalid data shape {full_shape}. Each dimension must have at least 1 element."
        )

    signal_definition = args.data_source.signal_definition()
    signal_writers: list[HDF5ChunkedSignalWriter] = [
        HDF5ChunkedSignalWriter(
            signal_definition.name, name, priorities, signal_definition.units
        )
        for name, priorities in args.data_source.output_chunk_priorities().items()
    ]
    signal_writers.sort(key=lambda w: w.chunking_name)

    extra_writers = args.data_source.output_accumulations()
    all_writers = [*signal_writers, *extra_writers]

    size_per_signal_item = np.dtype(signal_definition.dtype).itemsize
    max_items_per_chunk = int(
        args.field_options.max_bytes_per_chunk / size_per_signal_item
    )

    axis_definitions = args.data_source.axis_definitions()
    writer_shapes = [
        writer.init_shape_and_priorities(
            full_shape,
            axis_definitions,
        )
        for writer in all_writers
    ]

    memory_max_item_count = int(args.memory_max_byte_count / size_per_signal_item)

    all_chunkers: list[Chunker] = [
        Chunker.from_max_item_count(
            data_shape=shape,
            priorities=priorities,
            items_per_chunk=max_items_per_chunk,
        )
        for shape, priorities in writer_shapes
    ]

    memory_chunker, total_read_count, default_inx = choose_memory_buffer(
        args.data_source.read_chunks(),
        memory_max_item_count,
        density,
        all_chunkers[0 : len(signal_writers)],
        args.data_source.chunk_read_count,
    )
    signal_writers.insert(0, signal_writers.pop(default_inx))
    all_chunkers.insert(0, all_chunkers.pop(default_inx))

    logger.info(
        f" Giving a final data shape of {full_shape} (Raw {format_bytes(np.prod(full_shape) * size_per_signal_item)})",
    )

    logger.info(
        f"Using a memory chunk shape {memory_chunker.chunk_shape} and count {memory_chunker.chunk_count}.",
    )
    logger.info(
        f" Dense usage: {np.prod(memory_chunker.chunk_shape)} items ({format_bytes(np.prod(memory_chunker.chunk_shape) * size_per_signal_item)}).",
    )
    logger.info(
        f" Sparse usage: {int(np.prod(memory_chunker.chunk_shape) * density)} items ({format_bytes(np.prod(memory_chunker.chunk_shape) * size_per_signal_item * density)}), worst case density {density:.2f}.",
    )
    logger.info(f" Memory limit set at {format_bytes(args.memory_max_byte_count)}.")
    logger.info("With data blocks:")
    logger.info(
        f"maximum chunk size ({format_bytes(args.field_options.max_bytes_per_chunk)})",
    )
    logger.info("Signal data:")
    for ii, (writer, chunker) in enumerate(zip(all_writers, all_chunkers, strict=True)):
        if ii == 0:
            logger.info("Signal data:")
        elif ii == len(signal_writers):
            logger.info("Additional data:")
        size_per_item = np.dtype(writer.dtype(signal_definition.dtype)).itemsize
        logger.info(
            f"    {writer.name(): >10}: chunk shape {chunker.chunk_shape} and total count {chunker.chunk_count} and memory count {count_chunks_to_cover(memory_chunker.chunk_shape, chunker.chunk_shape)}.",
        )
        logger.info(
            f"    {' ' * 10}: chunk size: {np.prod(chunker.chunk_shape)} items ({format_bytes(np.prod(chunker.chunk_shape) * size_per_item)}).",
        )

    if args.dry_run:
        logger.info("Dry Run: exiting.")
        return

    logger.info(f"Writing results to {args.out_path}")

    nxs = NexusFile(args.out_path, mode="w")
    with nxs.as_context():
        add_items_to_group(args.data_source.instrument_metadata(), nxs.instrument)
        add_items_to_group(args.data_source.experiment_metadata(), nxs.experiment)
        add_items_to_group(
            args.data_source.user_metadata(), nxs.user, as_attributes=False
        )

        axes = fill_axes(args.data_source, axis_definitions, full_shape)

        nxs.entry.attrs["default"] = "data"

        for writer, chunker in zip(all_writers, all_chunkers, strict=True):
            writer.init_field(
                nxs,
                full_shape,
                axes,
                signal_definition.dtype,
                args.field_options,
                chunker,
            )

        with tqdm(
            desc="Overall reads", total=total_read_count, disable=not args.show_progress
        ) as overall_reads_timer:
            data_source_lock = Lock()
            accumulation_lock = Lock()

            def process_chunk(memory_chunk: Chunk) -> None:
                local_store = local()
                with data_source_lock:
                    local_store.chunk_data = prepare_data(
                        full_shape,
                        args.data_source.fill_chunk(
                            memory_chunk,
                            overall_reads_timer.update,
                        ),
                    )

                with accumulation_lock:
                    for writer in tqdm(
                        all_writers,
                        desc="Parsing data",
                        disable=not args.show_progress,
                    ):
                        writer.parse_data(
                            memory_chunk,
                            local_store.chunk_data,
                        )

            outer_chunks = list(memory_chunker.chunks())
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
                    disable=not args.show_progress,
                ):
                    try:
                        memory_chunk.result()
                    except:
                        for f in futures:
                            f.cancel()
                        raise
            for writer in tqdm(
                all_writers,
                desc="Finalising data",
                disable=not args.show_progress,
            ):
                writer.finalise()
