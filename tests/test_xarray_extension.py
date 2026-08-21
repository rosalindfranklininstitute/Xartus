# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
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

        initial_da.nexus.write_to(path, "/entry/")
        assert path.exists()

        final_da = xr.open_dataarray(path, engine="nexus", entry_path="/entry/ints")
        defer.on_complete(final_da.close)

        assert initial_da == final_da


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
            grp = fle.create_group("/entry/")
            grp.attrs["NX_class"] = "NXentry"

        initial_ds.nexus.write_to(path, "/entry/")
        assert path.exists()

        final_ds = xr.open_dataset(path, engine="nexus", entry_path="/entry/")
        defer.on_complete(final_ds.close)

        assert initial_ds == final_ds


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
        )
    )

    path = Path("./test.nxs")
    with FileGuard(path, on_complete=FileAction.DELETE), DeferredAction() as defer:
        initial_dt.nexus.write(path)
        assert path.exists()

        final_dt = xr.open_datatree(path, engine="nexus")
        defer.on_complete(final_dt.close)

        assert initial_dt == final_dt


def test_nexus_there_and_back(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]

    path = Path("./test.nxs")
    with FileGuard(path, on_complete=FileAction.DELETE), DeferredAction() as defer:
        dt = xr.open_datatree(nx_file, engine="nexus")
        defer.on_complete(dt.close)

        def compare(nxe, fle):
            initial_names = list(nxe)
            final_names = list(fle)

            assert initial_names == final_names

            for name in initial_names:
                initial_entry = nxe[name]
                final_entry = fle[name]

                assert dict(initial_entry.attrs) == dict(final_entry.attrs)

                if isinstance(initial_entry, h5py.Dataset):
                    assert isinstance(final_entry, h5py.Dataset)
                    np.testing.assert_allclose(initial_entry, final_entry)
                else:
                    assert isinstance(initial_entry, h5py.Group)
                    assert isinstance(final_entry, h5py.Group)
                    compare(initial_entry, final_entry)

        dt.nexus.write(path)
        assert path.exists()

        with h5py.File(nx_file, "r") as nxe, h5py.File(path, "r") as fle:
            compare(fle, nxe)
