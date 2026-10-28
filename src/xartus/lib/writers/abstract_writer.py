# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from xartus.lib.chunker import Chunker
from xartus.lib.bounds import Chunk, Shape
from xartus.lib.nxs import (
    FieldOptions,
    NxAxes,
)
from xartus.lib.multi_coo import MultiCOO
from xartus.lib.data_source import Axis


class AbstractWriter(ABC):
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the writer. This is used by the converyer in log messages.
        """

    @abstractmethod
    def supports_filetype(self, typ: type) -> bool:
        """
        Returns the file types that this wrier supports.
        """

    @abstractmethod
    def init_shape_and_priorities(
        self,
        data_shape: Shape,
        axes: list[Axis],
    ) -> tuple[Shape, tuple[int, ...]]:
        """
        This should return the shape and chunking priorities of the data that this writer will write.
        """

    def dtype(
        self,
        signal_dtype: npt.DTypeLike,
    ) -> npt.DTypeLike:
        """
        This should return the dtype that the data will be written with.
        """
        return signal_dtype

    @abstractmethod
    def init_field(
        self,
        file: Any,
        data_shape: Shape,
        axes: NxAxes,
        signal_dtype: npt.DTypeLike,
        field_options: FieldOptions,
        chunker: Chunker,
    ) -> None:
        """
        This should initialise the appropriate field in the given file.
        """

    @abstractmethod
    def parse_data(
        self,
        memory_chunk: Chunk,
        chunk_data: np.ndarray | MultiCOO,
    ) -> None:
        """
        Takes in the processed data.
        Should write the data or prepare it for writing in finalise()
        """

    @abstractmethod
    def finalise(self) -> None:
        """
        Should finalise this data writer.
        Flushing any buffers to file, and closing any open parts of the data.
        Should not close the high level file.
        """
