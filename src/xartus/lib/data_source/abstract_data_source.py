# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

from contextlib import AbstractContextManager
from typing import Any, Callable
from abc import abstractmethod

import numpy as np

from xartus.lib.bounds import Chunk, Shape
from xartus.lib.multi_coo import MultiCOO
from xartus.lib.writers import AbstractWriter

from .definitions import DataShape, Signal, Axis


class AbstractDataSource(AbstractContextManager):
    """
    Provides an interface that represents raw data.

    If this is fulfilled, then the class can be added to the :obj:`~xartus.api.data_converter.ProcessArgs` and :obj:`~xartus.api.data_converter.process` will read data from this class and write it to the specified NeXus file.

    """

    @abstractmethod
    def __enter__(self):
        """
        Called to open the data source.
        """

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Called to close the data source.
        """

    @abstractmethod
    def instrument_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the instrument metadata.
        This writes attributes to a NeXus NXinstrument.
        Any properties are accepted, but using NXinstrument properties improves self describing.
        """

    @abstractmethod
    def user_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the user metadata.
        This writes fields to a NeXus NXuser.
        Any properties are accepted, but using NXuser properties improves self describing.
        """

    @abstractmethod
    def experiment_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the experiment metadata.
        This writes attributes to a NeXus NXparameters.
        Any properties are accepted, but using NXparameters properties improves self describing.
        """

    @abstractmethod
    def shape(self) -> DataShape:
        """
        Return the shape of the data.
        """

    @abstractmethod
    def signal_definition(self) -> Signal:
        """
        Returns the name, data type, and (optionally) unit for the data.
        """

    @abstractmethod
    def output_chunk_priorities(self) -> dict[str, Shape]:
        """
        Returns the names and chunking priorities of the desired output array.
        For example simple image data (x,y, spectra) with shape (32,32,184000)
        might produce:
        'images':   (1,1,2) -> (32,32,1)
        'spectra':  (2,2,1) -> (1,1,184000)
        """

    def read_chunks(self) -> Shape | list[Shape] | None:
        """
        Returns a list of chunking priorities that can be used for reading the raw data.
        This is used in conjunction with chunk_read_count to select the most efficient reading scheme, and to track progress.
        If this returns None, then the priorities from the output_chunks are used.

        This is not an abstract method and defaults to returning None.
        """
        return None

    @abstractmethod
    def chunk_read_count(self, memory_chunk: Shape) -> int:
        """
        Returns the number of read operations needed to fill the provided memory chunk.
        """

    @abstractmethod
    def axis_definitions(self) -> list[Axis]:
        """
        Returns the axis that should be used when storing the data.
        Each axis has a name, a data type, the dimension it applies to, and (optionally) the unit.
        Every dimension must have an axis.
        Multiple dimensions can be defined for a single dimension. The first in the list for each dimesnsion will be used as the default.
        """

    @abstractmethod
    def axis_values(self, axis: Axis) -> np.ndarray:
        """
        Returns the values for the specified axis.
        """

    @abstractmethod
    def output_accumulations(self) -> list[AbstractWriter]:
        """
        Returns a list of the extra data you want in the file.
        These are implementations of Abstract writers.
        For example, with simple image data (x,y, spectra):
        return [
        SumAndMax("total_images", axes= ["mz"]), # Accumulate over the spectra
        SumAndMax("total_spectra", axes=["x","y"]), # Accumulate over the images
        ]

        """

    @abstractmethod
    def fill_chunk(
        self,
        memory_chunk: Chunk,
        update: Callable[[int], None],
    ) -> np.ndarray | MultiCOO:
        """
        Read data from the source in the region specified by
        memory_chunk and return that data.

        Args:
            memory_chunk:   The bounds of the data to read.
            update:         A callback to update progress.
                            The total of the progress counter is
                            sum([chunk_read_count(mc) for mc in all_memory_chunks])
        Returns:
            The data from the source.
            -> return_data.shape == self.shape()

        """
