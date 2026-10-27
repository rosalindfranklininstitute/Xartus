# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from xartus.lib.chunker import Chunker
from typing import override, Any
import numpy.typing as npt
import numpy as np
from h5py import h5s

from .abstract_writer import AbstractWriter

from xartus.lib.bounds import Chunk, Shape
from xartus.lib.nxs import (
    NexusFile,
    create_nxgroup,
    create_nxfield,
    FieldOptions,
    NxAxes,
)
from xartus.lib.multi_coo import MultiCOO
from xartus.lib.data_source import Axis


class HDF5ChunkedSignalWriter(AbstractWriter):
    def __init__(
        self,
        signal_name: str,
        chunking_name: str,
        chunk_priorities: tuple[int, ...],
        units: str | None = None,
    ):
        self.signal_name = signal_name
        self.chunking_name = chunking_name
        self.chunk_priorities = chunk_priorities
        self.units = units

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
        signal_item_width = np.dtype(signal_dtype).itemsize
        max_items_per_chunk = int(max_bytes_per_chunk / signal_item_width)
        self.chunker = Chunker.from_max_item_count(
            data_shape=data_shape,
            priorities=self.chunk_priorities,
            items_per_chunk=max_items_per_chunk,
        )

        return self.chunker

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
        if "data" not in self.nxs.entry:
            data_group = create_nxgroup(self.nxs.entry, "data", nx_class="NXsubentry")
            data_group.attrs["default"] = self.chunking_name
        else:
            data_group = self.nxs.entry["data"]

        signal_group = create_nxgroup(data_group, self.chunking_name, nx_class="NXdata")
        signal_group.attrs["signal"] = self.signal_name
        create_nxfield(
            signal_group,
            name=self.signal_name,
            dtype=signal_dtype,
            shape=data_shape,
            compression=field_options.compression,
            compression_opts=field_options.compression_opts,
            chunks=self.chunker.chunk_shape,
            shuffle=field_options.shuffle,
        )
        axes.add_to_group(signal_group)

    @override
    def parse_data(
        self,
        memory_chunk: Chunk,
        chunk_data: np.ndarray | MultiCOO,
    ) -> None:
        if isinstance(chunk_data, np.ndarray):
            self.nxs.entry["data"][self.chunking_name][self.signal_name][
                *memory_chunk
            ] = chunk_data
        elif len(chunk_data.coords) == 0:
            return
        else:
            ds = self.nxs.entry[f"data/{self.chunking_name}/{self.signal_name}"]
            f_space = ds.id.get_space()
            f_space.select_elements(chunk_data.coords.T, h5s.SELECT_SET)
            m_space = h5s.create_simple((len(chunk_data["signal"]),))
            ds.id.write(m_space, f_space, chunk_data["signal"])

            f_space.close()
            m_space.close()
        return

    @override
    def finalise(self) -> None:
        pass
