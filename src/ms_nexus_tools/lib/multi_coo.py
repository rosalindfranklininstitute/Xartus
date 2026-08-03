# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from .dtypes import Any1D, Intp1D
from .bounds import Shape


def find_uniques(
    coords: np.ndarray[tuple[int, ...], np.dtype[np.int32]],
    shape: Shape,
    count: bool = False,
) -> tuple[Intp1D, Intp1D | None]:
    """
    Finds the uniqe indices in the data. Optionally returns the number of occurances of each unique value.
    This assumes the coords array has been sorted.

    >>> find_uniques(np.array([[0,0],[0,1],[1,0]]).T, shape=(2,2), count=True)
    (array([0, 1, 2]), array([1, 1, 1]))

    >>> find_uniques(np.array([[0,0],[0,1],[1,0],[1,0]]).T, shape=(2,2), count=True)
    (array([0, 1, 2]), array([1, 1, 2]))

    >>> find_uniques(np.array([[0,0],[0,1],[1,0],[1,0]]).T, shape=(2,2), count=False)
    (array([0, 1, 2]), None)

    >>> coords = np.array([[0,0],[0,1],[1,0],[1,0],[1,1],[1,1]]).T
    >>> ind, _ = find_uniques(coords, shape=(2,2), count=False)
    >>> coords[:, ind].T
    array([[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]], shape=(4, 2))

    """
    # Inspired by sparse.COO
    # See https://github.com/pydata/sparse/blob/main/LICENSE
    # This is the BSD 3-clause license
    linear: Intp1D = np.ravel_multi_index(coords, shape)
    unique_mask = np.diff(linear) != 0

    counts = np.array([], dtype=np.intp)

    if unique_mask.sum() == len(unique_mask):
        return np.arange(len(linear), dtype=np.intp), np.ones(
            (len(linear),), dtype=np.intp
        ) if count else None

    unique_mask = np.append(True, unique_mask)

    # coords = coords[:, unique_mask]
    (unique_inds,) = np.nonzero(unique_mask)
    if count:
        counts = np.diff(unique_inds)
        counts = np.append(counts, len(linear) - unique_inds[-1])
    else:
        counts = None

    return unique_inds, counts


@dataclass
class MultiCOO:
    coords: np.ndarray[tuple[int, ...], np.dtype[np.int32]]
    values: dict[str, Any1D]

    def __getitem__(self, key: str) -> Any1D:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def sort(self, shape) -> None:
        # Inspired by sparse.COO
        # See https://github.com/pydata/sparse/blob/main/LICENSE
        # This is the BSD 3-clause license

        linear = np.ravel_multi_index(self.coords, shape)
        if np.all(np.diff(linear) >= 0):
            return
        order = np.argsort(linear)
        self.coords = self.coords[:, order]
        self.values = {k: v[order] for k, v in self.values.items()}

    def acc_duplicates(
        self,
        shape: Shape,
        count=False,
        accumulators: dict[str, np.ufunc] = {},
        default_accumulator=np.add,
    ) -> Intp1D | None:
        # Inspired by sparse.COO
        # See https://github.com/pydata/sparse/blob/main/LICENSE
        # This is the BSD 3-clause license

        acc: defaultdict[str, np.ufunc] = defaultdict(lambda: default_accumulator)
        for k, v in accumulators.items():
            acc[k] = v

        linear: Intp1D = np.ravel_multi_index(self.coords, shape)
        unique_mask = np.diff(linear) != 0

        counts = None

        if unique_mask.sum() == len(unique_mask):
            return np.ones((len(linear),), dtype=np.intp) if count else counts

        unique_mask = np.append(True, unique_mask)

        self.coords = self.coords[:, unique_mask]
        (unique_inds,) = np.nonzero(unique_mask)
        if count:
            counts = np.diff(unique_inds)
            counts = np.append(counts, len(linear) - unique_inds[-1])

        self.values = {
            k: acc[k].reduceat(
                v,
                unique_inds,
                dtype=v.dtype,
            )
            for k, v in self.values.items()
        }

        return counts
