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
    def supports_filetype(self, typ: type) -> bool:
        """
        Returns the file types that this wrier supports.
        """

    @abstractmethod
    def init_chunker(
        self,
        data_shape: Shape,
        signal_dtype: npt.DTypeLike,
        max_bytes_per_chunk: int,
        axes: list[Axis],
    ) -> Chunker:
        """
        This method should initialise the chunker for this writer and return it.
        """

    @abstractmethod
    def init_field(
        self,
        file: Any,
        data_shape: Shape,
        axes: NxAxes,
        signal_dtype: npt.DTypeLike,
        field_options: FieldOptions,
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
