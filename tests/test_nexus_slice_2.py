# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import h5py

from ms_nexus_tools.api import nexus_slice, data_convert
from ms_nexus_tools.lib.data_source import Axis, AxisType
from ms_nexus_tools.lib.nexus_slicer import NexusSlicer

from nexus_pixel_man_test_data import Man2DDataSource, ManData, data_files

import pytest
from hypothesis import given, strategies as st


@pytest.fixture(scope="module")
def man_data_and_nexus():
    man_file = data_files()["man1"]
    man_data = ManData()
    man_data_source = Man2DDataSource(
        man_data,
        supplimentary_axes=[
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


def test_slicer_preserves_data_and_axis(man_data_and_nexus, out_file):
    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    partial_data = slicer[:, :, :]

    a = partial_data.as_numpy()
    partial_data.store("group", out_file)

    with h5py.File(out_file, "r") as out_nx, h5py.File(nx_file, "r") as in_nx:
        np.testing.assert_allclose(out_nx["entry/group/signal"][...], a)
        np.testing.assert_allclose(
            out_nx["entry/group/signal"][...], in_nx["entry/images/signal"][...]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/x"][...], in_nx["entry/images/x"][...]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/time"][...], in_nx["entry/images/time"][...]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/y"][...], in_nx["entry/images/y"][...]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/mz"][...], in_nx["entry/images/mz"][...]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/error"][...], in_nx["entry/images/error"][...]
        )


def test_reject_invalid_slices(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    # Reject missing axes on dimension
    with pytest.raises(KeyError):
        slicer[("y", 12), :, :]

    # Reject values out of bounds of the axis
    with pytest.raises(IndexError):
        slicer[("x", 30), :, :]
    with pytest.raises(IndexError):
        slicer[("x", -1), :, :]


def test_equivalent_slicing(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    # subsequent slicing adds
    assert (
        slicer[("x", 0, 10), :, ("mz", 5, 10)][("x", 0, 5), :, :]
        == slicer[("x", 0, 5), :, ("mz", 5, 10)]
    )
    assert (
        slicer[("x", 0, 5), :, ("mz", 5, 10)][("x", 0, 10), :, :]
        == slicer[("x", 0, 5), :, ("mz", 5, 10)]
    )

    # slicing by different axis
    assert slicer[("x", 0, 10), :, :] == slicer[("time", 0, 100), :, :]

    # slicing by raw index
    assert slicer[:, ("y", 0, 10), :] == slicer[:, 0:100, :]

    # subsequent slicing adds, even with different axis
    assert (
        slicer[("x", 0, 10), :, ("mz", 5, 10)][("time", 0, 50), :, :]
        == slicer[("x", 0, 5), :, ("mz", 5, 10)]
    )


@st.composite
def slice_spec(draw, length):
    start = draw(st.none() | st.integers(min_value=0, max_value=length))
    stop = draw(st.none() | st.integers(min_value=0, max_value=length))
    step = draw(st.none() | st.integers(min_value=-length, max_value=length))
    return slice(start, stop, step)


def slice_to_ax(name, slc: slice, multiplier: float) -> tuple | slice:
    if slc.start is None:
        if slc.step is None:
            if slc.stop is None:
                return slice(None)
            return (name, slc.stop * multiplier)
        return (name, None, slc.stop * multiplier, slc.step * multiplier)
    if slc.step is None:
        if slc.stop is None:
            return (name, slc.start * multiplier, None)
        return (name, slc.start * multiplier, slc.stop * multiplier)
    if slc.stop is None:
        return (name, slc.start * multiplier, None, slc.step * multiplier)
    return (name, slc.start * multiplier, slc.stop * multiplier, slc.step * multiplier)


@given(dim1=slice_spec(240), dim2=slice_spec(240), dim3=slice_spec(240))
def test_valid_slices(dim1, dim2, dim3, man_data_and_nexus, out_file):

    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    x_slice = slice_to_ax("x", dim1, 0.1)
    y_slice = slice_to_ax("y", dim2, 0.1)
    mz_slice = slice_to_ax("mz", dim3, 0.1)

    if any([s.step == 0 for s in [dim1, dim2, dim3]]):
        with pytest.raises(ValueError):
            partial_data = slicer[x_slice, y_slice, mz_slice]
        return

    partial_data = slicer[x_slice, y_slice, mz_slice]

    a = partial_data.as_numpy()
    partial_data.store("group", out_file)

    with h5py.File(out_file, "r") as out_nx, h5py.File(nx_file, "r") as in_nx:
        np.testing.assert_allclose(out_nx["entry/group/signal"], a)
        np.testing.assert_allclose(
            out_nx["entry/group/signal"][...],
            in_nx["entry/images/signal"][dim1, dim2, dim3],
        )

        np.testing.assert_allclose(
            out_nx["entry/group/x"][...], in_nx["entry/images/x"][dim1]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/time"][...], in_nx["entry/images/time"][dim1]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/y"][...], in_nx["entry/images/y"][dim2]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/mz"][...], in_nx["entry/images/mz"][dim3]
        )
        np.testing.assert_allclose(
            out_nx["entry/group/error"][...], in_nx["entry/images/error"][dim3]
        )


def test_accumulators_add_unless_different(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    # Different names are equivalent
    assert slicer.accumulate(np.add, "mz") == slicer.accumulate(np.add, "error")

    # Multiple calls are equivalent to one call.
    assert slicer.accumulate(np.add, "mz").accumulate(np.add, "x") == slicer.accumulate(
        np.add, "x", "mz"
    )

    # Multiple calls that don't change anything have no effect.
    assert slicer.accumulate(np.add, "mz").accumulate(
        np.add, "mz"
    ) == slicer.accumulate(np.add, "mz")

    assert slicer.accumulate(np.add, "mz").accumulate(
        np.add, "error"
    ) == slicer.accumulate(np.add, "mz")

    # Changing the type of accumulator is invalid
    with pytest.raises(ValueError):
        slicer.accumulate(np.add, "mz").accumulate(np.subtract, "mz")

    # TODO (dmd): We could allow this by alowing looping and accumulating being ordered operations. Thus the following would add through mz, THEN subract though x.
    # But that has the disagvantage that I am then fully implmenting a processing pipeline on data!
    # It is probable that doing this it would be better to just wrap the nexus dataset axes ina pandas object, sortof.
    # Having said that, if we get this sort of API to work on top of somethign like nexusformat, it will make nexus increadibly pythonic!
    with pytest.raises(TypeError):
        slicer.accumulate(np.add, "mz").accumulate(np.subtract, "x")


def test_looping_add(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    assert slicer.loop("mz").loop("x") == slicer.loop("x", "mz")


def test_cannot_loop_and_accumulate(man_data_and_nexus):
    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    with pytest.raises(ValueError):
        slicer.loop("mz").accumulate(np.add, "mz")

    with pytest.raises(ValueError):
        slicer.accumulate(np.add, "mz").loop("mz")

    with pytest.raises(ValueError):
        slicer.accumulate(np.add, "mz").loop("error")


def test_loop_and_slice(man_data_and_nexus, nx_dir):
    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    out_file = nx_dir / "rows.nxs"
    slicer[:, ("y", 0.2, 0.6), ("mz", 9, 21)].loop("x").store("rows", out_file)

    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        for yy in range(8):
            name = f"/entry/rows/x_{yy * 0.1:.3g}/data"
            assert f"{name}/signal" in fle
            assert f"{name}/x" not in fle
            assert f"{name}/y" in fle
            assert f"{name}/mz" in fle
        data = fle[f"{name}/signal"][:, :]
        assert data.shape == (4, 60)
        np.testing.assert_allclose(data, man_data_and_nexus[0].dense[yy, 2:6, 120:180])

    out_file = nx_dir / "pixels.nxs"
    slicer[("x", 0.2, 0.3), :, :].loop("x", "y").store("pixels", out_file)
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        for yy in range(8):
            name = f"/entry/pixels/x_0.2-y_{yy * 0.1:.3g}/data"
            assert f"{name}/signal" in fle
            assert f"{name}/x" not in fle
            assert f"{name}/y" not in fle
            assert f"{name}/mz" in fle
        data = fle[f"{name}/signal"][:]
        assert data.shape == (240,)
        np.testing.assert_allclose(data, man_data_and_nexus[0].dense[2, yy, :])


def test_sum_and_slice(man_data_and_nexus, nx_dir):
    nx_file = man_data_and_nexus[1]
    slicer = NexusSlicer(nx_file, "/entry/images")

    out_file = nx_dir / "all.nxs"
    slicer.accumulate(np.add, "mz").store("all", out_file)

    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/all/data/signal" in fle
        assert "/entry/all/data/x" in fle
        assert "/entry/all/data/time" in fle
        assert "/entry/all/data/y" in fle
        assert "/entry/all/data/mz" not in fle
        assert "/entry/all/data/error" not in fle
        data = fle["/entry/all/data/signal"][:, :]
        assert data.shape == (8, 8)
        np.testing.assert_allclose(data, np.sum(man_data_and_nexus[0].dense, axis=2))

    out_file = nx_dir / "slice.nxs"
    slicer[:, :, ("mz", 12, 18)].accumulate(np.add, "mz").store("slice", out_file)
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/slice/data/signal" in fle
        assert "/entry/slice/data/x" in fle
        assert "/entry/slice/data/time" in fle
        assert "/entry/slice/data/y" in fle
        assert "/entry/slice/data/mz" not in fle
        assert "/entry/slice/data/error" not in fle
        data = fle["/entry/slice/data/signal"][:, :]
        assert data.shape == (8, 8)
        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
        )

    out_file = nx_dir / "cross.nxs"
    slicer[:, :, ("mz", 12, 18)].accumulate(np.add, "x", "y").store("cross", out_file)
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/cross/data/signal" in fle
        assert "/entry/cross/data/x" not in fle
        assert "/entry/cross/data/time" not in fle
        assert "/entry/cross/data/y" not in fle
        assert "/entry/cross/data/mz" in fle
        assert "/entry/cross/data/error" in fle
        data = fle["/entry/data/signal"][:]
        assert data.shape == (60,)
        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=(0, 1))
        )
