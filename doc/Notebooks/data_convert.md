---
file_format: mystnb
kernelspec:
  name: python3
---
<!--
SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>

SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
-->

# Data Converter

`Xartus` provides a plugin framework for converting data into NeXus format.
This consists of two parts: 
1. {py:class}``AbstractDataSource``: which, when inherited and implemented, reads data. And,
2. ``data_convert``: which takes ina `DataSource` and writes a NeXus file. Taking care of chunking and concurrency.

Here is some test data:
```{code-cell} 
from pathlib import Path 
import os

import matplotlib.pyplot as plt
import numpy as np

filename = Path(os.environ["DOCS_SOURCE_DIR"])/"Xantu's humminbird.png"

im_data = plt.imread(filename)
print(im_data.shape)
print(im_data.dtype)

plt.imshow(im_data)
```

Now we implement the {py:class}``AbstractDataSource`` for this data:

```{code-cell} python
from typing import Any, Callable
import datetime as dt

from xartus.lib import Shape, Chunk
from xartus.lib import AbstractDataSource, Axis, Signal, AxisType, DataShape
from xartus.lib import MultiCOO


class HummingbirdSource(AbstractDataSource):
    def __init__(self, filename):
        self.data = plt.imread(filename)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def instrument_metadata(self) -> dict[str, Any]:
        return {"name": "Notebook"}

    def experiment_metadata(self) -> dict[str, Any]:
        return {"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}

    def shape(self) -> DataShape:
        return DataShape(self.data.shape, is_sparse=False, worst_case_density=1.0)

    def signal_definition(self) -> Signal:
        return Signal("pixels", np.float32, units=None)

    def output_chunks(self) -> dict[str, Shape]:
        return {"pixels": (2, 2, 1)}

    def read_chunks(self) -> list[Shape] | None:
        return None

    def chunk_read_count(self, memory_chunk: Shape) -> int:
        return np.prod(memory_chunk)

    def axis_definitions(self) -> list[Axis]:
        return [
            Axis("y", 0, AxisType.EXACT, np.int8, units="pix"),
            Axis("x", 1, AxisType.EXACT, np.int8, units="pix"),
            Axis("colour", 2, AxisType.EXACT, np.int8, units="intens"),
        ]

    def exact_axis_values(self, axis: Axis) -> np.ndarray:
        return np.arange(self.data.shape[axis.primary_axis], dtype=np.int8)

    def binned_axis_edges(self, axis: Axis) -> np.ndarray:
        raise NotImplementedError()

    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        return {
            "total_image": ("col",),
            "total_colour": ("x", "y"),
        }

    def fill_chunk(
        self,
        memory_chunk: Chunk,
        update: Callable[[int], None],
    ) -> np.ndarray | MultiCOO:
        update(np.prod(memory_chunk.shape))
        return self.data[*memory_chunk]
```

Now we convert the data into a NeXus file using `data_convert`

```{code-cell}
from xartus.api import data_convert

out_path = Path('./out.nxs')

process_args = data_convert.ProcessArgs(
    in_path=filename,
    out_path=out_path,
    chunk_max_byte_count=1024 * 1024,
    memory_max_byte_count=1024 * 1024 * 1024,
    show_progress=False,
    data_source=HummingbirdSource(filename),
)
data_convert.process(process_args, {})

print(out_path.exists())
```

Finally we may load the data into an xarray:

```{code-cell}
from matplotlib.colors import LogNorm
import xartus.lib.xarray_backend
import xarray as xr

dataset = xr.open_dataset(
          out_path,
          engine="nexus", 
          entry_path="/entry/data",
          )
print('Dataset:')
print(dataset)

for ii, cmap in enumerate(['Reds','Greens','Blues']):
    ax = plt.subplot(2,3,ii+1)
    dataset.pixels.sel(colour=ii).plot(ax=ax, x='x', y='y', yincrease=False, cmap=cmap)
    ax.set_title(cmap)

ax = plt.subplot(2,3,5)
dataset.pixels.sel(colour=3).plot(ax=ax, x='x', y='y', yincrease=False, cmap='Greys')
_ = ax.set_title('alpha')

dataset.close()
out_path.unlink()

```
