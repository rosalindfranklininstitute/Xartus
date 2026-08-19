# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from pathlib import Path
from contextlib import AbstractContextManager

import math
from typing import Iterator, Any, Iterable
from bisect import bisect_right, bisect_left
import json

import numpy as np

from .bounds import Shape
from .dtypes import Number1D, Number


def format_bytes(n: Number, digits: int = 2) -> str:
    """
    Format the given number of bytes into byte units.

    Args:
        n: The number of bits to convert to bytes.
        digits: The number of digits to display, if decimals are used.

    Returns:
        The byte formatted number of bytes.

    Examples:
        >>> format_bytes(10)
        '10b'

        >>> format_bytes(1000)
        '1000b'

        >>> format_bytes(512+1024)
        '1.50Kb'

        >>> format_bytes(1024*1024*1.25)
        '1.25Mb'

        Digits defaults to 2, but can be specified.

        >>> format_bytes(512+1024, digits=1)
        '1.5Kb'

        The number of digits does not have an effect on integer values

        >>> format_bytes(1000, digits=1)
        '1000b'

    """
    negative = n < 0
    units = ["b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb"]
    i = 0
    value = abs(float(n))
    while value >= 1024 and i < len(units) - 1:
        value /= 1024.0
        i += 1
    prefix = "-" if negative else ""
    if value.is_integer():
        return f"{prefix}{int(value)}{units[i]}"
    return f"{prefix}{value:.{digits}f}{units[i]}"


def parse_bytes(bytes_str: str) -> int:
    """
    Parse the given string into the number of bytes.

    Args:
        bytes_str: The string to convert to the number of bits.

    Returns:
        The number of bits represented.

    Examples:
        >>> parse_bytes('10b')
        10

        >>> parse_bytes('1.50Kb')
        1536

        >>> parse_bytes('1.25Mb')
        1310720

        >>> parse_bytes('1250Kb')
        1280000

        >>> parse_bytes('0.025Kb')
        26

        Works with output of format bytes

        >>> parse_bytes(format_bytes(512+1024))
        1536

        >>> parse_bytes(format_bytes(512+1024, digits=1))
        1536

        Provides the cailing of any fractions:

        >>> parse_bytes('1.1b')
        2

    """
    values = dict(
        Kb=1024,
        Mb=1024**2,
        Gb=1024**3,
        Tb=1024**4,
        Pb=1024**5,
        Eb=1024**6,
        b=1,
    )

    bytes_str = bytes_str.strip()
    value = None
    for tail, multiplier in values.items():
        if bytes_str.endswith(tail):
            value = float(bytes_str.removesuffix(tail)) * multiplier
            break
    else:
        raise ValueError(f"Did not understand the suffix of {bytes_str}.")

    if value is None:
        raise RuntimeError("Suffix found, but value was invalid")

    return int(math.ceil(value))


def count_digits(num: int) -> int:
    """
    Counts the number of digits in an integer:

    Args:
        num: The number to count the difits of.

    Returns:
        The number of digits needed to represent num.

    Examples:
        >>> count_digits(1), count_digits(2)
        (1, 1)

        >>> count_digits(10), count_digits(12)
        (2, 2)

        >>> count_digits(100), count_digits(314)
        (3, 3)

        >>> count_digits(-100), count_digits(-10)
        (3, 2)

        >>> count_digits(0)
        1
    """
    digits = 1
    num = abs(num) // 10
    while num > 0:
        digits += 1
        num = num // 10
    return digits


def slice_len(slc: slice) -> int:
    """
    Returns the length of a slice.

    Args:
        slc: The slice to calculate the length of.
             This is taken as a literal number.
             Weird results may occur if negatives are used.

    Returns:
        The number of items in the slice, if it can expand endlessly.

    Examples:
        >>> slice_len(slice(5))
        5

        >>> slice_len(slice(1, 5))
        4

        >>> slice_len(slice(1, 5, 2))
        2

        >>> slice_len(slice(1, -5, 2))
        -3
    """
    inc = slc.step or 1

    if slc.start is None:
        return slc.stop // inc
    return (slc.stop - slc.start) // inc


def slice_range(slc: slice) -> range:
    """
    Returns the range of the slice.
    Note that this is different to slice.indices(len) in that it does not take a length.

    Args:
        slc: The slice to wrap in a range.

    Returns:
        A range object covering the slice.

    Examples:
        >>> slice_range(slice(5))
        range(0, 5)

        >>> slice_range(slice(1, 5))
        range(1, 5)

        >>> slice_range(slice(1, 5, 2))
        range(1, 5, 2)
    """
    if slc.start is None and slc.step is None:
        return range(slc.stop)
    if slc.step is None:
        return range(slc.start, slc.stop)
    if slc.start is None:
        return range(0, slc.stop, slc.step)
    return range(slc.start, slc.stop, slc.step)


def slice_from_values(start: Number, stop: Number, values: Number1D) -> slice:
    start_index = bisect_left(values, start)
    stop_index = bisect_right(values, stop)
    return slice(start_index, stop_index)


class NotTqdm:
    """
    A small utility to provide the same, high level api, but without doing any thing.
    Note that this may be made, somewhat, redundant with the argument: disable=True.
    """

    def __init__(self, iterator: Iterable | None = None, **kwargs):
        self.iterator = iterator

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self) -> Iterator[Any]:
        if self.iterator is None:
            raise TypeError("NotTqdm expected an iterable when used as an iterator.")
        for item in self.iterator:
            yield item

    def update(self) -> None:
        pass


def json_add(filename, *keys, value) -> None:
    """
    Adds the given value to the json file.
    Overriding one that exists.
    """
    old_data = {}
    if filename.exists():
        with open(filename, "r") as fd:
            old_data = json.load(fd)
    if len(keys) >= 1:
        new_data = old_data
        for key in keys[:-1]:
            if key not in new_data:
                new_data[key] = {}
            new_data = new_data[key]
        new_data[keys[-1]] = value
    else:
        assert isinstance(value, dict)
        old_data.update(value)
    with open(filename, "w") as fd:
        json.dump(old_data, fd, indent=2)


def indices(shape: Shape, axis=None) -> Iterator[tuple[slice | int, ...]]:
    if axis is None:
        yield (slice(None) for ii in range(len(shape)))
    else:
        if isinstance(axis, int):
            axis = [axis]
        ndim = len(shape)
        axis = np.sort([a if a >= 0 else a + ndim for a in axis])
        iterable_shape = [shape[ii] for ii in axis]
        ndims = len(shape)
        slices = np.array([0 if ii in axis else slice(None) for ii in range(ndims)])
        for values in np.ndindex(*iterable_shape):
            slices[axis] = values
            yield tuple(slices)


def iterate(array: np.ndarray, axis=None) -> Iterator[np.ndarray]:
    for slc in indices(array.shape, axis):
        yield array[*slc]


def reduce_shape(shape: Shape, axis=None) -> Shape:
    """
    Returns the data shape for the given axis.

    Args:
        shape: The initial shape to be reduced from.
        axis: The axis to remove.

    Returns:
        The new shape with all the axis removed.

    Examples:
        >>> reduce_shape((1,2,3))
        (1, 2, 3)

        >>> reduce_shape((1,2,3), axis=-1)
        (1, 2)

        >>> reduce_shape((1,2,3), axis=0)
        (2, 3)

        >>> reduce_shape((1,2,3), axis=(0, -1))
        (2,)
    """
    if axis is None:
        return shape
    if isinstance(axis, int):
        axis = [axis]
    ndim = len(shape)
    axis = np.sort([a if a >= 0 else a + ndim for a in axis])
    return Shape(v for ii, v in enumerate(shape) if ii not in axis)


class FileGuard(AbstractContextManager):
    """
    This context takes in a collection and checks their state after the block runs.
    It can delete files if a process fails (to avoid leaving polluting temporary files on process failures.)
    It can check that files exist after a process (asserting a know outcome)
    """

    def __init__(
        self,
        *paths: Path,
        delete_on_failure: bool = True,
        check_exist_on_success: bool = True,
    ):
        """
        Args:
            paths: The paths to be gaurded.
            delete_on_failure: Whether to deleted the guarded paths
                               if an error occurs in the context.
            check_exist_on_success: Whether to assert that the guarded paths
                                    exist after the context successfully completes.
        """
        self.paths = paths
        self.delete_on_failure = delete_on_failure
        self.check_exist_on_success = check_exist_on_success

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            if self.delete_on_failure:
                for path in self.paths:
                    path.unlink(missing_ok=True)
        elif self.check_exist_on_success:
            nonexisting_files = [str(path) for path in self.paths if not path.exists()]
            if len(nonexisting_files) != 0:
                raise FileNotFoundError(
                    f"Expected the following files to exists, but did not: {', '.join(nonexisting_files)}"
                )

        return False


def simplify_chunks(
    chunks: tuple[tuple[int, ...], ...] | tuple[int, ...],
) -> tuple[int, ...]:
    """
    Returns the simplified chunks representation (tuple[int,...]) from the given chunks.
    Asserts that all the values for each dimension are the same, except the last.

    Args:
        chunks: Chunks represented by individual lengths in each dimension,
                spanning the whole space. As returned by something like Dask.

    Returns:
        A simplified representation, where each dimension has a single number.

    Examples:
        >>> simplify_chunks((1,2,3))
        (1, 2, 3)

        >>> simplify_chunks(((1,), (2,), (3,)))
        (1, 2, 3)

        >>> simplify_chunks(((1,1), (2,2), (3,3)))
        (1, 2, 3)

        >>> simplify_chunks(((1,1), (2,1), (3,1)))
        (1, 2, 3)

    """
    result = []
    for c in chunks:
        if isinstance(c, int):
            result.append(c)
        elif len(c) == 1:
            result.append(c[0])
        elif len(c) == 0:
            raise ValueError("Chunk component had size of 0.")
        else:
            main_value = c[0]
            value_count = len(c)
            for ii, m in enumerate(c):
                if ii == value_count - 1:
                    if m > main_value:
                        raise ValueError(
                            "The last chunk component is larger than the main value."
                        )
                elif m != main_value:
                    raise ValueError(
                        "Chunk components are not all the same size (except the last)."
                    )
            result.append(main_value)

    return tuple(result)
