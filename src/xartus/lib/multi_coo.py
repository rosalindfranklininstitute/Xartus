# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .dtypes import Any1D, Intp1D
from .bounds import Shape


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
        """
        Accumulates the duplicated indices in the data. Optionally returns the number of occurrences of each unique value.
        This assumes the coords array has been sorted.

        >>> data = MultiCOO(np.array([[0,0],[0,1],[1,0]]).T, values=dict(signal=np.array([1,1,1])))
        >>> data.acc_duplicates(shape=(2,2), count=True)
        array([1, 1, 1])

        >>> data.coords.T
        array([[0, 0],
               [0, 1],
               [1, 0]], shape=(3, 2))

        >>> data.values
        {'signal': array([1, 1, 1])}

        >>> data = MultiCOO(np.array([[0,0],[0,1],[1,0],[1,0]]).T, values=dict(signal=np.array([1,1,1,1])))
        >>> data.acc_duplicates(shape=(2,2), count=True)
        array([1, 1, 2])

        >>> data.coords.T
        array([[0, 0],
               [0, 1],
               [1, 0]], shape=(3, 2))

        >>> data.values
        {'signal': array([1, 1, 2])}

        >>> data = MultiCOO(np.array([[0,0],[0,1],[1,0],[1,0]]).T, values=dict(signal=np.array([1,1,1,1])))
        >>> data.acc_duplicates(shape=(2,2), count=False)

        >>> data.coords.T
        array([[0, 0],
               [0, 1],
               [1, 0]], shape=(3, 2))

        >>> data.values
        {'signal': array([1, 1, 2])}

        A different accuulator can be use. For example using sum (The default):

        >>> data = MultiCOO(np.array([[0,0],[0,1],[1,0],[1,0]]).T, values=dict(signal=np.array([1,1,2,3])))
        >>> data.acc_duplicates(shape=(2,2), count=False)
        >>> data.values
        {'signal': array([1, 1, 5])}

        Specify using max for a specific value:

        >>> data = MultiCOO(np.array([[0,0],[0,1],[1,0],[1,0]]).T, values=dict(signal=np.array([1,1,2,3])))
        >>> data.acc_duplicates(shape=(2,2), count=False, accumulators=dict(signal=np.maximum))
        >>> data.values
        {'signal': array([1, 1, 3])}

        Specify using min as the default_accumulator:

        >>> data = MultiCOO(np.array([[0,0],[0,1],[1,0],[1,0]]).T, values=dict(signal=np.array([1,1,2,3])))
        >>> data.acc_duplicates(shape=(2,2), count=False, default_accumulator=np.minimum)
        >>> data.values
        {'signal': array([1, 1, 2])}

        """
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

    def set_binned_indices(self, primary_axis, values, edges) -> None:
        labels = np.searchsorted(edges[:-2], values)
        self.coords[primary_axis, :] = labels
