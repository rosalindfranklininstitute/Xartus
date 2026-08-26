---
file_format: mystnb
kernelspec:
  name: python3
---
<!--
SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>

SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
-->

# Imshow Sparse
The default tools available in ``matplotlib`` and ``xarray`` are sufficient for any plotting needs.
However there is a class of image where the x- and y-axis are not fully populated. 
``imshow`` handles this by ignoring the gaps. ``pcolormesh`` handles it by stretching the pixels. Sometimes it is desirable to leave the gaps in the image. 
This is what {py:func}``xartus.lib.image.imshow_sparse`` is for. This notebook demonstrates its use. 

First we create some data with gaps in the axis.

```{code-cell} 
import numpy as np
import xarray as xr

x = np.array([1, 2, 3, 4, 5, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20])
y = np.array([11, 12, 13, 14, 18, 19, 21, 22, 25, 26, 27, 28, 29, 30])
n = x.size * y.size
values = np.arange(n).reshape(x.size, y.size)

xr_values= xr.DataArray(
    values,
    dims=['x','y'],
    coords={'x': x, 'y':y}
)
```

Plotting this as a scatter plot shows how the data should be laid out in an image. 
```{code-cell}
import matplotlib.pyplot as plt
import matplotlib.colors as colors

cmap = "viridis"

fig, ax = plt.subplots()

xx, yy = np.meshgrid(x, y, indexing="ij")
ax.scatter(
    xx.ravel(),
    yy.ravel(),
    c=values.ravel(),
    cmap=cmap,
    s=50,
)

ax.set_xlabel("x")
ax.set_ylabel("y")
```

But we cannot ge the markers to be pixels: rectangles that take up the whole
space spanned by their x and y values. Here we compare the three alternatives:
using ``imshow``, ``pcolormesh`` and ``imshow_sparse``.


```{code-cell}

vmin = xr_values.min()
vmax = xr_values.max()

norm = colors.Normalize(vmin=vmin, vmax=vmax)

fig, axs = plt.subplots(1,3)

im = xr_values.plot.imshow(x='x', y='y', ax=axs[0], add_colorbar=False, cmap=cmap, norm=norm)
axs[0].set_title('imshow')

pc = xr_values.plot.pcolormesh(x='x', y='y', ax=axs[1], add_colorbar=False, cmap=cmap, norm=norm)
axs[1].set_title('pcolormesh')

from xartus.lib import image

pl = image.imshow_sparse(axs[2], xr_values, x="x", y="y", cmap=cmap, norm=norm)
axs[2].set_title('imshow_sparse')

for ax in axs:
    ax.set_aspect("equal", adjustable="box")
```

This smooth data shows that they all layout the same, but with and without
gaps. For some data, where the gaps are small and the data is smooth, ``pcolormesh`` is a satisfying solution.
However when the gaps need to be preserved, as is the case with some biological imaging, ``imshow_sparse`` offers a better solution.

```{code-cell}
values_rnd = np.random.rand(x.size, y.size) * 0.9 * n + 0.1 * n
xr_values_rnd = xr.DataArray(values_rnd, dims=["x", "y"], coords={"x": x, "y": y})

vmin = xr_values_rnd.min()
vmax = xr_values_rnd.max()

norm = colors.Normalize(vmin=vmin, vmax=vmax)

fig, axs = plt.subplots(1,3)

im = xr_values_rnd.plot.imshow(x='x', y='y', ax=axs[0], add_colorbar=False, cmap=cmap, norm=norm)
axs[0].set_title('imshow')

pc = xr_values_rnd.plot.pcolormesh(x='x', y='y', ax=axs[1], add_colorbar=False, cmap=cmap, norm=norm)
axs[1].set_title('pcolormesh')

pl = image.imshow_sparse(axs[2], xr_values_rnd, x="x", y="y", cmap=cmap, norm=norm)
axs[2].set_title('imshow_sparse')

for ax in axs:
    ax.set_aspect("equal", adjustable="box")

```

## Limitations

``imshow_sparse`` is build on top of ``np.histogram2d``. This means that it
load the whole image into memory, and assumes a ``numpy`` array. In particular
``dask`` arrays do not support ``histogram2d`` and so will need to be computed
before ``imshow__sparse`` can be used.


Using ``np.histogram2d`` is reasonably fast, considering the alternative (like
plotting each rectangle individually). However, it is still not fast in
comparison to general plotting. Care should be taken if large images are
plotted.
