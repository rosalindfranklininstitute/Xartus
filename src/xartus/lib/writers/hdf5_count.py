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


class Count(AbstractWriter):
    def __init__(
        self,
        name: str,
        chunk_priorities: tuple[int, ...] | None = None,
        dtype: npt.DTypeLike = np.int32,
    ):
        self.group_name = name
        self.chunk_priorities = chunk_priorities
        self.count_dtype = dtype

    @override
    def name(self) -> str:
        return self.group_name

    @override
    def supports_filetype(self, typ) -> bool:
        return typ is NexusFile

    @override
    def init_shape_and_priorities(
        self,
        data_shape: Shape,
        axes: list[Axis],
    ) -> tuple[Shape, tuple[int, ...]]:
        if self.chunk_priorities is None:
            self.chunk_priorities = tuple(1 for _ in data_shape)
        return data_shape, self.chunk_priorities

    @override
    def dtype(self, signal_dtype: npt.DTypeLike) -> npt.DTypeLike:
        return self.count_dtype

    @override
    def init_field(
        self,
        file: Any,
        data_shape: Shape,
        axes: NxAxes,
        signal_dtype: npt.DTypeLike,
        field_options: FieldOptions,
        chunker: Chunker,
    ) -> None:
        assert isinstance(file, NexusFile)
        self.nxs = file
        if "qa" not in self.nxs.entry:
            data_group = create_nxgroup(self.nxs.entry, "qa", nx_class="NXsubentry")
            data_group.attrs["default"] = self.group_name
        else:
            data_group = self.nxs.entry["count"]

        signal_group = create_nxgroup(data_group, self.group_name, nx_class="NXdata")
        signal_group.attrs["signal"] = "counts"
        create_nxfield(
            signal_group,
            name="counts",
            dtype=self.count_dtype,
            shape=data_shape,
            compression=field_options.compression,
            compression_opts=field_options.compression_opts,
            chunks=chunker.chunk_shape,
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
            self.nxs.entry["qa"][self.group_name]["counts"][*memory_chunk] = chunk_data
        elif len(chunk_data.coords) == 0:
            return
        else:
            ds = self.nxs.entry[f"qa/{self.group_name}/counts"]
            f_space = ds.id.get_space()
            f_space.select_elements(chunk_data.coords.T, h5s.SELECT_SET)
            m_space = h5s.create_simple((len(chunk_data["counts"]),))
            ds.id.write(m_space, f_space, chunk_data["counts"])

            f_space.close()
            m_space.close()
        return

    @override
    def finalise(self) -> None:
        pass
