<!--
SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>

SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
-->

# Xartus: <strong>Xarr</strong>ay <strong>t</strong>hrough NeX<strong>us</strong>

![xartus-version](./badges/xartus-version.svg)
![xartus-requires-python](./badges/xartus-requires-python.svg)
![xartus-license](./badges/xartus-license.svg)

![tests](./badges/tests.svg)
![skipped](./badges/skipped.svg)
![coverage](./badges/coverage.svg)
![last-run](./badges/last-run.svg)

This repo is hope to a collection of tools for converting data into the [NeXus](https://www.nexusformat.org/) format.
In addition, there are tools to read and write NeXus files to and from [Xarray](https://docs.xarray.dev/en/stable/index.html) DataArrays, DataSets and Datatrees.

The library has two core parts and a few helper utilities.

  1. The `data_converter` and `data_source` allows reading any data into a NeXus file.
  2. An xarray `BackendEntrypoint` (`NexusEntrypoint`) allows reading NeXus files into xarray.
  3. Xarray accessors (`array.nexus`, `dataset.nexus`, and `datatree.nexus`) allow saving the data to NeXus files.
  4. `utils` has some helper functions that are commmonly needed when reading/processing common data formats.
  5. `plotting` adds alternate plotting to pcolormesh, and plotting slices of a dataset.
