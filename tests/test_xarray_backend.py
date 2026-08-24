# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from xartus.lib.exceptions import InvalidEntryError
from pathlib import Path

import numpy as np
import xarray as xr

from xartus.api import data_convert
from xartus.lib.data_source import Axis, AxisType

from nexus_pixel_man_test_data import Man2DDataSource, ManData, data_files

import pytest


@pytest.fixture(scope="module")
def man_data_and_nexus():
    man_file = data_files()["man1"]
    man_data = ManData()
    man_data_source = Man2DDataSource(
        man_data,
        supplementary_axes=[
            Axis("time", 0, AxisType.EXACT, np.float32, "s"),
            Axis("error", 2, AxisType.EXACT, np.float32, ""),
        ],
        multipliers=dict(x=0.1, y=0.1, mz=0.1, time=1.0, error=1.0),
    )
    filename = Path(__file__).parent / "man.nxs"
    if filename.exists():
        filename.unlink()

    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=filename,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})
    yield man_data, filename
    filename.unlink()


def test_default_dataarray(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    da = xr.open_dataarray(nx_file, engine="nexus")

    assert da is not None
    assert isinstance(da, xr.DataArray)

    assert da.name == "data"
    assert "x" in da.coords
    assert "y" in da.coords
    assert "mz" in da.coords
    assert "time" in da.coords
    assert "error" in da.coords
    assert "x" in da.dims
    assert "y" in da.dims
    assert "mz" in da.dims

    da.close()


def test_specific_dataarray(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    da = xr.open_dataarray(nx_file, engine="nexus", entry_path="/entry/images/data")

    assert da is not None
    assert isinstance(da, xr.DataArray)

    assert da.name == "data"
    assert "x" in da.coords
    assert "y" in da.coords
    assert "mz" in da.coords
    assert "time" in da.coords
    assert "error" in da.coords
    assert "x" in da.dims
    assert "y" in da.dims
    assert "mz" in da.dims

    da.close()


def test_default_dataset(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    ds = xr.open_dataset(nx_file, engine="nexus")

    assert ds is not None
    assert isinstance(ds, xr.Dataset)

    assert "data" in ds
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "mz" in ds.coords
    assert "time" in ds.coords
    assert "error" in ds.coords
    assert "x" in ds.dims
    assert "y" in ds.dims
    assert "mz" in ds.dims

    ds.close()


def test_specific_dataset(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    ds = xr.open_dataset(nx_file, engine="nexus", entry_path="/entry/images/")

    assert ds is not None
    assert isinstance(ds, xr.Dataset)

    assert "data" in ds
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "mz" in ds.coords
    assert "time" in ds.coords
    assert "error" in ds.coords
    assert "x" in ds.dims
    assert "y" in ds.dims
    assert "mz" in ds.dims

    ds.close()


@pytest.mark.skip(
    reason="Specific dataarray backends are not, yet, supported by xarray."
)
def test_error_on_load_non_nxdata_dataset(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    with pytest.raises(InvalidEntryError, match="Expected .* to be NXdata"):
        xr.open_dataset(nx_file, engine="nexus", entry_path="/entry/")


def test_whole_tree(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    dt = xr.open_datatree(nx_file, engine="nexus")

    assert "entry" in dt

    assert "instrument" in dt["entry"]
    assert "name" in dt["entry/instrument"].attrs
    assert not dt["entry/instrument"].has_data

    assert "experiment" in dt["entry"]
    assert "date" in dt["entry/experiment"].attrs
    assert "force_sparse" in dt["entry/experiment"].attrs
    assert not dt["entry/experiment"].has_data

    assert "images" in dt["entry"]
    assert "spectra" in dt["entry"]
    assert "total_image" in dt["entry"]
    assert "total_spectra" in dt["entry"]

    assert "data" in dt["entry/images"]
    assert "data" in dt["entry/spectra"]

    assert "x" in dt["entry/images/data"].coords
    assert "time" in dt["entry/images/data"].coords
    assert "y" in dt["entry/images/data"].coords
    assert "mz" in dt["entry/images/data"].coords
    assert "error" in dt["entry/images/data"].coords

    dt.close()


def test_subtree(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    dt = xr.open_datatree(nx_file, engine="nexus", root="/entry/images")
    assert "data" in dt
    dt.close()


def test_error_on_load_non_nxentry_datatree(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    with pytest.raises(
        InvalidEntryError, match="Expected .* to be NXroot, NXentry or NXsubentry"
    ):
        xr.open_datatree(nx_file, engine="nexus", root="/entry/images/data")
