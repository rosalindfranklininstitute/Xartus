# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
"""
Utilities for drawing images from 2D data.
Here are utilities to mark annotations.
"""

from typing import Any, Hashable
from enum import Enum
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xarray as xr

from .dtypes import Float1D32, Number
from .utils import slice_from_values
from .plotting import Plottable


class OriginLocation(Enum):
    UPPER_LEFT = "upper left"
    UPPER_RIGHT = "upper right"
    LOWER_LEFT = "lower left"
    LOWER_RIGHT = "lower right"


def adjust_origin(
    a: np.ndarray,
    new: OriginLocation,
    current: OriginLocation = OriginLocation.UPPER_LEFT,
) -> np.ndarray:

    left = (OriginLocation.UPPER_LEFT, OriginLocation.LOWER_LEFT)
    right = (OriginLocation.UPPER_RIGHT, OriginLocation.LOWER_RIGHT)

    upper = (OriginLocation.UPPER_LEFT, OriginLocation.UPPER_RIGHT)
    lower = (OriginLocation.LOWER_LEFT, OriginLocation.LOWER_RIGHT)

    axis = []
    if (current in left and new in right) or (current in right and new in left):
        axis.append(1)
    if (current in upper and new in lower) or (current in lower and new in upper):
        axis.append(0)

    if len(axis) > 0:
        return np.flip(a, axis=tuple(axis))
    return a


@dataclass
class XYRectangle:
    """
    Represents a 2D slice of data.
    """

    x_start: float
    x_stop: float
    y_start: float
    y_stop: float

    def x_slice(self, x_values: Float1D32) -> slice:
        return slice_from_values(self.x_start, self.x_stop, x_values)

    def y_slice(self, y_values: Float1D32) -> slice:
        return slice_from_values(self.y_start, self.y_stop, y_values)

    def get_plot_rect(self, **kwargs) -> Rectangle:
        x = self.x_start
        w = self.x_stop - x
        y = self.y_start
        h = self.y_stop - y
        return Rectangle((x, y), w, h, **kwargs)


def imshow_sparse(
    ax: plt.Axes,
    darray: xr.DataArray,
    x: Hashable | None = None,
    y: Hashable | None = None,
    xy_rectangles: list[Plottable[XYRectangle]] = [],
    diff_selector=np.median,
    **kwargs,
) -> tuple[Any, tuple[Number, Number]]:
    """
    This is like plt.imshow, except that it will correctly
    plot images where the x- and y-values are not uniformly distributed.

    If there are not coordinates for x or y
    (or ``dims[0]`` or ``dims[1]``, if either is ``None``),
    then ``np.arange(0, darray.shape[i])`` is used.
    If this is true for both dimensions, then the result should be the same as ``plt.imshow`` but with the origin being at the bottom left, onstead of the top left.

    Args:
        darray: Must be two-dimensional.
        x: Coordinate for x axis. If ``None``, use ``darray.dims[1]``.
        y: Coordinate for y axis. If ``None``, use ``darray.dims[0]``.

    Returns:
        The object from imshow, and the min and max values.

    """
    if x is None and darray.dims[1] not in darray.coords:
        x_values = np.arange(darray.shape[1])
    else:
        x = x if x is not None else darray.dims[1]
        ax.set_xlabel(str(x))
        x_values = darray.coords[x]
    if y is None and darray.dims[0] not in darray.coords:
        y_values = np.arange(darray.shape[0])
    else:
        x = x if x is not None else darray.dims[1]
        ax.set_xlabel(str(x))
        y_values = darray.coords[y] if y is not None else darray.coords[darray.dims[0]]

    im_min, im_max = np.percentile(darray, [0, 100])

    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    mnx = np.min(x_values)
    mxx = np.max(x_values)
    mny = np.min(y_values)
    mxy = np.max(y_values)
    dx = diff_selector(np.diff(x_values))
    dy = diff_selector(np.diff(y_values))
    img, xedges, yedges = np.histogram2d(
        xx.ravel(),
        yy.ravel(),
        weights=darray.data.ravel(),
        bins=[
            np.arange(mnx - dx / 100, mxx + dx / 100 + dx, dx),
            np.arange(mny - dy / 100, mxy + dy / 100 + dx, dy),
        ],
    )
    im = ax.imshow(np.flip(img.T, axis=0), extent=(mnx, mxx, mny, mxy), **kwargs)

    for xy_rect in xy_rectangles:
        rect = xy_rect.value.get_plot_rect(
            linewidth=2,
            edgecolor=xy_rect.color,
            facecolor=xy_rect.color,
            alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            *rect.get_bbox().max,
            xy_rect.title,
            color=xy_rect.color,
            fontsize=12,
        )
    return im, (im_min, im_max)
