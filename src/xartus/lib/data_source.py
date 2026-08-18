# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

from contextlib import AbstractContextManager
from typing import Any, Callable, NamedTuple
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


from .bounds import Chunk, Shape
from .multi_coo import MultiCOO


@dataclass
class Signal:
    name: str
    dtype: npt.DTypeLike
    units: str | None = None


class AxisType(Enum):
    EXACT = 1
    BINNED = 2


@dataclass
class Axis:
    name: str
    primary_axis: int
    axis_type: AxisType
    dtype: npt.DTypeLike
    units: str | None = None


class UnknownAxisError(Exception):
    def __init__(self, name: str, density: AxisType | None = None):
        match density:
            case None:
                super().__init__(f"Unknown axis: {name}")
            case AxisType.EXACT:
                super().__init__(f"Unknown exact axis: {name}")
            case AxisType.BINNED:
                super().__init__(f"Unknown binned axis: {name}")


class DataShape(NamedTuple):
    shape: Shape
    is_sparse: bool
    worst_case_density: float


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
        """

    @abstractmethod
    def experiment_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the experiment metadata.
        """

    @abstractmethod
    def shape(self) -> DataShape:
        """
        Return the shape of the data.
        """

    @abstractmethod
    def signal_definition(self) -> Signal:
        """
        Returns the type for data.
        """

    @abstractmethod
    def output_chunks(self) -> dict[str, Shape]:
        """
        Returns the names and chunking priorities of the desired output array.
        For example simple image data (x,y, spectra) with shape (32,32,184000)
        might produce:
        'images':   (1,1,2) -> (32,32,1)
        'spectra':  (2,2,1) -> (1,1,184000)
        """

    def read_chunks(self) -> list[Shape] | None:
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
        For example simple image data (x,y, spectra):

        axis(0) : Axis('x', 0, [], EXACT, 'um')

        axis(1) : Axis('y', 1, [], EXACT, 'um')

        If is it exact:

        axis(2) : Axis('mz', 2, [], EXACT, 'mz')

        if it is only peaks:

        axis(2) : Axis('mz', 2, [0,1], BINNED, 'mz')
        """

    @abstractmethod
    def exact_axis_values(self, axis: Axis) -> np.ndarray:
        """
        Returns the values for the specified exact axis.
        """

    @abstractmethod
    def binned_axis_edges(self, axis: Axis) -> np.ndarray:
        """
        Returns the bin edges used to histogram the given binned axis.
        This is used for generating the output accumulations across this axis, if required.
        """

    @abstractmethod
    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        """
        Returns the names and lists of axis that should be
        accumulated (summed and max).
        For example simple image data (x,y, spectra):
        might produce:
        'total_images':     ('mz') # Accumulate over the spectra
        'total_spectra':    ('x','y') # Accumulate over the images
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
