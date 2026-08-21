# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
import sys
from xarray.testing import assert_identical
from xartus.lib.h5_printer import print_group
from pathlib import Path
import h5py
from xartus.lib.utils import FileGuard, FileAction, DeferredAction
import xarray as xr
import numpy as np

from nexus_pixel_man_test_data import Man2DDataSource, ManData, data_files

import xartus.lib.xarray_backend  # noqa: F401 # Needed to register extensions
from xartus.lib.data_source import Axis, AxisType
from xartus.api import data_convert

import pytest


def assert_equal_recursive(actual, expected):
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal_recursive(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_equal_recursive(a, e)
    else:
        assert actual == expected


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


def test_dataarray_there_and_back():

    shape = [3, 10, 1000]

    ints = np.arange(0, np.prod(shape)).reshape(shape)

    dims = ["x", "y", "z"]

    initial_da = xr.DataArray(
        ints,
        name="ints",
        dims=dims,
        coords={key: np.arange(s) for key, s in zip(dims, shape, strict=True)},
        attrs={"min": 0, "max": np.prod(shape)},
    )
    path = Path("./test.nxs")
    with FileGuard(path, on_complete=FileAction.DELETE), DeferredAction() as defer:
        with h5py.File(path, "w") as fle:
            grp = fle.create_group("/entry/")
            grp.attrs["NX_class"] = "NXentry"

        initial_da.nexus.write_to(path, "/entry/cats/")
        assert path.exists()

        final_da = xr.open_dataarray(path, engine="nexus", entry_path="/entry/cats")
        defer.on_complete(final_da.close)

        assert_identical(initial_da, final_da)


def test_dataset_there_and_back():

    shape = [3, 10, 1000]

    ints = np.arange(0, np.prod(shape)).reshape(shape)
    fractions = np.arange(0, np.prod(shape)).reshape(shape) / np.prod(shape)

    dims = ["x", "y", "z"]

    initial_ds = xr.Dataset(
        {
            "ints": xr.DataArray(
                ints,
                dims=dims,
                coords={key: np.arange(s) for key, s in zip(dims, shape, strict=True)},
                attrs={"min": 0, "max": np.prod(shape)},
            ),
            "fractions": xr.DataArray(
                fractions,
                dims=dims,
                coords={key: np.arange(s) for key, s in zip(dims, shape, strict=True)},
                attrs={"min": 0, "max": 1},
            ),
        }
    )

    path = Path("./test.nxs")
    with FileGuard(path, on_complete=FileAction.DELETE), DeferredAction() as defer:
        with h5py.File(path, "w") as fle:
            fle["/"].attrs["NX_class"] = "NXroot"

        initial_ds.nexus.write_to(path, "/entry/")

        assert path.exists()

        final_ds = xr.open_dataset(path, engine="nexus", entry_path="/entry/")
        defer.on_complete(final_ds.close)

        assert_identical(initial_ds, final_ds)


def test_datatree_there_and_back():

    shape = [3, 10, 1000]

    ints = np.arange(0, np.prod(shape)).reshape(shape)
    fractions = np.arange(0, np.prod(shape)).reshape(shape) / np.prod(shape)

    dims = ["x", "y", "z"]

    initial_dt = xr.DataTree(
        xr.Dataset(
            {
                "ints": xr.DataArray(
                    ints,
                    dims=dims,
                    coords={
                        key: np.arange(s) for key, s in zip(dims, shape, strict=True)
                    },
                    attrs={"min": 0, "max": np.prod(shape)},
                ),
                "fractions": xr.DataArray(
                    fractions,
                    dims=dims,
                    coords={
                        key: np.arange(s) for key, s in zip(dims, shape, strict=True)
                    },
                    attrs={"min": 0, "max": 1},
                ),
            }
        ),
        children={
            "child": xr.DataTree(
                xr.Dataset(
                    {
                        "sub_int": xr.DataArray(
                            ints,
                            dims=dims,
                            coords={
                                key: np.arange(s)
                                for key, s in zip(dims, shape, strict=True)
                            },
                            attrs={"min": 0, "max": np.prod(shape)},
                        ),
                        "sub_fractions": xr.DataArray(
                            fractions,
                            dims=dims,
                            coords={
                                key: np.arange(s)
                                for key, s in zip(dims, shape, strict=True)
                            },
                            attrs={"min": 0, "max": 1},
                        ),
                    }
                )
            ),
            "params": xr.DataTree(
                xr.Dataset(attrs={"NX_class": "NXparameters", "python": sys.version})
            ),
        },
    )

    path = Path("./test.nxs")
    with FileGuard(path, on_complete=FileAction.DELETE), DeferredAction() as defer:
        initial_dt.nexus.write_to(path)
        assert path.exists()

        final_dt = xr.open_datatree(path, engine="nexus")
        defer.on_complete(final_dt.close)

        assert_identical(initial_dt, final_dt)


def test_nexus_there_and_back(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    with h5py.File(nx_file, "r") as fle:
        print_group(fle)

    path = Path("./test.nxs")
    with FileGuard(path, on_complete=FileAction.DELETE), DeferredAction() as defer:
        dt = xr.open_datatree(nx_file, engine="nexus")
        defer.on_complete(dt.close)
        dt.nexus.write_to(path)
        assert path.exists()

        def compare(nxe, fle):
            initial_names = list(nxe)
            final_names = list(fle)

            assert initial_names == final_names

            for name in initial_names:
                initial_entry = nxe[name]
                final_entry = fle[name]

                initial_attrs = dict(initial_entry.attrs)
                final_attrs = dict(final_entry.attrs)
                try:
                    assert_equal_recursive(initial_attrs, final_attrs)
                except AssertionError as e:
                    message = f"Expected {initial_entry.name} to have same attrs as {final_entry.name}. But found a difference."
                    print("initial_attrs: ", file=sys.stderr)
                    print(initial_attrs, file=sys.stderr)
                    print("final_attrs: ", file=sys.stderr)
                    print(final_attrs, file=sys.stderr)
                    raise AssertionError(message) from e

                if isinstance(initial_entry, h5py.Dataset):
                    assert isinstance(final_entry, h5py.Dataset)
                    np.testing.assert_allclose(initial_entry, final_entry)
                else:
                    assert isinstance(initial_entry, h5py.Group)
                    assert isinstance(final_entry, h5py.Group)
                    compare(initial_entry, final_entry)

        with h5py.File(nx_file, "r") as nxe, h5py.File(path, "r") as fle:
            compare(fle, nxe)
