---
file_format: mystnb
kernelspec:
  name: python3
---
<!--
SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>

SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
-->

# Xarray extensions

This is a small demonstration of loading a nexus file, inspecting it. 
Then the data is manipulated and plotted.
FInally the summary data is written to disk.

Setup the environment.

```{code-cell}
import os
from pathlib import Path

import numpy as np
import xarray as xr

import xartus.lib.xarray_backend

filename = Path(os.environ["DOCS_SOURCE_DIR"])/"test_data.nxs" 
```

This displays the layout of the entry we are interested in:
```{code-cell}
from xartus.lib.h5_printer import print_group
import h5py

with h5py.File(filename, 'r') as fle:
  print_group(fle['/entry/images'])

```

Now load it into an `xarray`:
```{code-cell}
dataset = xr.open_dataset(
          filename,
          engine="nexus", 
          entry_path="/entry/images",
          )
print(dataset)
da = dataset['data']
print(da)
```

Now plot the sum over all the 'mz' axis of the data.
```{code-cell}
da.sum(dim='mz').plot(x='x', y='y', yincrease=False)
```

The data actually contains 4 distinct images. This selects then plots the first.
```{code-cell}
first_man = da.isel(mz=slice(0,60)).sum(dim='mz')
print(first_man)
first_man.plot(x='x', y='y', yincrease=False)
```

Gather the sum of each man into a single `DataArray`, and plot it to show all of them.
```{code-cell}
men = xr.concat(
          [da.isel(mz=slice(60*ii, 60*(ii+1))).sum(dim='mz') for ii in range(4)], 
          dim=xr.DataArray(np.arange(4), dims=('man',), name='man'),
          )
print(men)
men.plot(x='x', y='y', col='man', col_wrap=2, yincrease=False)
```

Write the data:
```{code-cell}

from xartus.lib.utils import FileGuard, FileAction

out_filename=Path('./tmp.nxs')
with FileGuard(out_filename, on_complete=FileAction.DELETE):
    ds = xr.Dataset(data_vars={'men': men})
    print(ds)
    print()

    ds.nexus.write_to(out_filename, entry_path='/entry')

    with h5py.File(out_filename, 'r') as fle:
        print_group(fle['/'])
```

