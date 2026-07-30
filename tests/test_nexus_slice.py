# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from functools import reduce

from pathlib import Path

from shutil import rmtree

import numpy as np
import h5py

from ms_nexus_tools.lib.chunker import count_chunks_to_cover
from ms_nexus_tools.api import nexus_slice, data_convert
from ms_nexus_tools.lib.data_source import Axis, AxisDensity
from ms_nexus_tools.api.nexus_slice import (
    SliceType,
    ActionType,
    GroupType,
    MissingArgumentError,
)

from .data import man_source

import pytest

from icecream import ic


@pytest.fixture(scope="module")
def man_data_and_nexus():
    man_data = man_source.ManData()
    man_data_source = man_source.ManSource(
        man_data,
        supplimentary_axes=[
            Axis("time", 0, AxisDensity.CONTINUOUS, np.int16, "s"),
            Axis("error", 2, AxisDensity.CONTINUOUS, np.int16, ""),
        ],
        multipliers=dict(x=0.1, y=0.1, mz=0.1, time=1.0, error=1.0),
    )
    filename = Path(__file__).parent / "man.nxs"
    if filename.exists():
        filename.unlink()
    process_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "data" / "Man1.txt",
        out_path=filename,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})
    yield man_data, filename
    filename.unlink()


@pytest.fixture
def nx_dir():
    filename = Path(__file__).parent / "test_files"
    if filename.exists():
        rmtree(filename)
    filename.mkdir()
    yield filename
    rmtree(filename)


def test_fully_specified(man_data_and_nexus, nx_dir):

    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        group_type=[
            ["man3", "view"],
            ["spectra", "view"],
        ],
        paths=[
            ["man3", "/entry/images/data/", "/entry/spectra/data/"],
            ["spectra", "/entry/images/data/", "/entry/spectra/data/"],
        ],
        action=[
            ["man3", "x", "leave"],
            ["man3", "y", "leave"],
            ["man3", "mz", "sum"],
            ["spectra", "x", "sum"],
            ["spectra", "y", "sum"],
            ["spectra", "mz", "leave"],
        ],
        slice=[
            ["man3", "x", "all"],
            ["man3", "y", "all"],
            ["man3", "mz", "range", str(12), str(18)],
            ["spectra", "x", "all"],
            ["spectra", "y", "all"],
            ["spectra", "mz", "all"],
        ],
    )
    nexus_slice.process(process_args, {})

    out_file = nx_dir / "spectra.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" not in fle
        assert "/entry/data/time" not in fle
        assert "/entry/data/y" not in fle
        assert "/entry/data/mz" in fle
        assert "/entry/data/error" in fle
        data = fle["/entry/data/signal"][:]
        assert np.all(data == np.sum(man_data_and_nexus[0].dense, axis=(0, 1)))

    out_file = nx_dir / "man3.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" not in fle
        assert "/entry/data/error" not in fle
        data = fle["/entry/data/signal"][:, :]

        assert np.all(
            data == np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
        )


def test_using_defaults(man_data_and_nexus, nx_dir):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["man3", "mz", "sum"],
            ["spectra", "x", "sum"],
            ["spectra", "y", "sum"],
        ],
        default_slice=["all"],
        slice=[
            ["man3", "mz", "range", str(12), str(18)],
        ],
    )
    nexus_slice.process(process_args, {})

    out_file = nx_dir / "spectra.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" not in fle
        assert "/entry/data/time" not in fle
        assert "/entry/data/y" not in fle
        assert "/entry/data/mz" in fle
        assert "/entry/data/error" in fle
        data = fle["/entry/data/signal"][:]
        assert np.all(data == np.sum(man_data_and_nexus[0].dense, axis=(0, 1)))

    out_file = nx_dir / "man3.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" not in fle
        assert "/entry/data/error" not in fle
        data = fle["/entry/data/signal"][:, :]

        assert np.all(
            data == np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
        )


def test_error_on_missing(man_data_and_nexus, nx_dir):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["man3", "mz", "sum"],
            ["spectra", "x", "sum"],
            ["spectra", "y", "sum"],
        ],
        default_slice=["all"],
        slice=[
            ["man3", "mz", "range", str(12), str(18)],
        ],
    )

    process_args.default_group_type = GroupType.Error
    with pytest.raises(MissingArgumentError, match="Missing 'group type'.*'man3'."):
        nexus_slice.process(process_args, {})
    process_args.default_group_type = GroupType.View

    process_args.default_paths = []
    with pytest.raises(MissingArgumentError, match="Missing 'paths'.*'man3'."):
        nexus_slice.process(process_args, {})
    process_args.default_paths = ["/entry/images/data/", "/entry/spectra/data/"]

    process_args.default_action = ActionType.Error
    with pytest.raises(MissingArgumentError, match="Missing 'action'.*'man3'.*'x'."):
        nexus_slice.process(process_args, {})

    process_args.default_action = ActionType.Leave

    process_args.default_slice = []
    with pytest.raises(MissingArgumentError, match="Missing 'slice'.*'man3'.*'x'."):
        nexus_slice.process(process_args, {})
    process_args.default_slice = ["all"]


def test_leave_and_slice(man_data_and_nexus, nx_dir):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["man_all", "mz", "leave"],
        ],
        default_slice=["all"],
        slice=[
            ["man3_range", "mz", "range", str(12), str(18)],
            ["man3_centre", "mz", "centred", str(15), str(6)],
            ["man3_value", "mz", "value", str(15)],
        ],
    )
    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man_all.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" in fle
        assert "/entry/data/error" in fle
        data = fle["/entry/data/signal"][:, :, :]
        assert data.shape == (8, 8, 240)
        assert np.all(data == man_data_and_nexus[0].dense)

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man3_range.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" in fle
        assert "/entry/data/error" in fle
        data = fle["/entry/data/signal"][:, :, :]
        assert data.shape == (8, 8, 60)
        assert np.all(data == man_data_and_nexus[0].dense[:, :, 120:180])

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man3_centre.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" in fle
        assert "/entry/data/error" in fle
        data = fle["/entry/data/signal"][:, :, :]
        assert data.shape == (8, 8, 60)
        assert np.all(data == man_data_and_nexus[0].dense[:, :, 120:180])

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man3_value.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" in fle
        assert "/entry/data/error" in fle
        data = fle["/entry/data/signal"][:, :, :]
        assert data.shape == (8, 8, 1)
        value_axis = fle["/entry/data/mz"][:]
        assert value_axis.shape == (1,)
        assert value_axis[0] == 15

        assert np.all(data[:, :, 0] == man_data_and_nexus[0].dense[:, :, 150])


def test_loop_and_slice(man_data_and_nexus, nx_dir):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["rows", "x", "loop"],
            ["pixels", "x", "loop"],
            ["pixels", "y", "loop"],
        ],
        default_slice=["all"],
        slice=[
            ["rows", "y", "range", str(0.2), str(0.6)],
            ["rows", "mz", "centred", str(15), str(6)],
            ["pixels", "x", "value", str(0.2)],
        ],
    )

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "rows.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        for yy in range(8):
            name = f"/entry/x_{yy * 0.1:.3g}/data"
            assert f"{name}/signal" in fle
            assert f"{name}/x" not in fle
            assert f"{name}/y" in fle
            assert f"{name}/mz" in fle
        data = fle[f"{name}/signal"][:, :]
        assert data.shape == (4, 60)
        assert np.all(data == man_data_and_nexus[0].dense[yy, 2:6, 120:180])

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "pixels.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        for yy in range(8):
            name = f"/entry/x_0.2-y_{yy * 0.1:.3g}/data"
            assert f"{name}/signal" in fle
            assert f"{name}/x" not in fle
            assert f"{name}/y" not in fle
            assert f"{name}/mz" in fle
        data = fle[f"{name}/signal"][:]
        assert data.shape == (240,)
        assert np.all(data == man_data_and_nexus[0].dense[2, yy, :])


def test_sum_and_slice(man_data_and_nexus, nx_dir):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["man_all", "mz", "sum"],
            ["man3_range", "mz", "sum"],
            ["man3_centre", "mz", "sum"],
            ["man3_value", "mz", "sum"],
            ["man3_cross_value", "x", "sum"],
            ["man3_cross_value", "y", "sum"],
        ],
        default_slice=["all"],
        slice=[
            ["man3_range", "mz", "range", str(12), str(18)],
            ["man3_centre", "mz", "centred", str(15), str(6)],
            ["man3_value", "mz", "value", str(15)],
            ["man3_cross_value", "mz", "value", str(15)],
        ],
    )
    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man_all.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" not in fle
        assert "/entry/data/error" not in fle
        data = fle["/entry/data/signal"][:, :]
        assert data.shape == (8, 8)
        assert np.all(data == np.sum(man_data_and_nexus[0].dense, axis=2))

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man3_range.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" not in fle
        assert "/entry/data/error" not in fle
        data = fle["/entry/data/signal"][:, :]
        assert data.shape == (8, 8)
        assert np.all(
            data == np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
        )

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man3_centre.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" not in fle
        assert "/entry/data/error" not in fle
        data = fle["/entry/data/signal"][:, :]
        assert data.shape == (8, 8)
        assert np.all(
            data == np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
        )

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man3_value.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" not in fle
        assert "/entry/data/error" not in fle
        data = fle["/entry/data/signal"][:, :]
        assert data.shape == (8, 8)
        assert np.all(data[:, :] == man_data_and_nexus[0].dense[:, :, 150])

    nexus_slice.process(process_args, {})
    out_file = nx_dir / "man3_cross_value.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" not in fle
        assert "/entry/data/time" not in fle
        assert "/entry/data/y" not in fle
        assert "/entry/data/mz" in fle
        assert "/entry/data/error" in fle
        data = fle["/entry/data/signal"][:]
        assert data.shape == (1,)
        assert np.all(
            data == np.sum(man_data_and_nexus[0].dense[:, :, 150], axis=(0, 1))
        )


def test_multiaxis_off_default_slice(man_data_and_nexus, nx_dir):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["man3", "error", "sum"],
            ["spectra", "time", "sum"],
            ["spectra", "y", "sum"],
        ],
        default_slice=["all"],
        slice=[
            ["man3", "error", "range", str(120), str(180)],
        ],
    )
    nexus_slice.process(process_args, {})

    out_file = nx_dir / "spectra.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" not in fle
        assert "/entry/data/time" not in fle
        assert "/entry/data/y" not in fle
        assert "/entry/data/mz" in fle
        assert "/entry/data/error" in fle
        data = fle["/entry/data/signal"][:]
        assert np.all(data == np.sum(man_data_and_nexus[0].dense, axis=(0, 1)))

    out_file = nx_dir / "man3.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" in fle
        assert "/entry/data/time" in fle
        assert "/entry/data/y" in fle
        assert "/entry/data/mz" not in fle
        assert "/entry/data/error" not in fle
        data = fle["/entry/data/signal"][:, :]

        assert np.all(
            data == np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
        )


def test_complete_sum(man_data_and_nexus, nx_dir):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["all", "x", "sum"],
            ["all", "y", "sum"],
            ["all", "mz", "sum"],
        ],
        default_slice=["all"],
    )
    nexus_slice.process(process_args, {})
    out_file = nx_dir / "all.nxs"
    assert out_file.exists()
    with h5py.File(out_file, "r") as fle:
        assert "/entry/data/signal" in fle
        assert "/entry/data/x" not in fle
        assert "/entry/data/time" not in fle
        assert "/entry/data/y" not in fle
        assert "/entry/data/mz" not in fle
        assert "/entry/data/error" not in fle
        data = fle["/entry/data/signal"][...]
        assert data.shape == (1,)
        assert np.all(data == np.sum(man_data_and_nexus[0].dense))


def test_binned_axis(man_data_and_nexus, nx_dir):
    filename = Path(__file__).parent / "man_binned.nxs"
    try:
        man_data = man_source.ManData()
        man_data_source = man_source.ManSource(
            man_data,
            supplimentary_axes=[
                Axis("mz", 2, AxisDensity.BINNED, np.int16, "mz"),
                Axis("error", 2, AxisDensity.BINNED, np.int16, ""),
            ],
            multipliers=dict(x=0.1, y=0.1, mz=0.1, error=1.0),
        )
        if filename.exists():
            filename.unlink()
        process_args = data_convert.ProcessArgs(
            in_path=Path(__file__).parent / "data" / "Man1.txt",
            out_path=filename,
            chunk_max_byte_count=1024 * 1024,
            memory_max_byte_count=1024 * 1024 * 1024,
            data_source=man_data_source,
        )
        data_convert.process(process_args, {})

        nexus_args = nexus_slice.ProcessArgs(
            in_path=man_data_and_nexus[1],
            out_dir=nx_dir,
            default_group_type=GroupType.View,
            default_paths=["/entry/images/data/", "/entry/spectra/data/"],
            default_action=ActionType.Leave,
            action=[
                ["all", "y", "sum"],
                ["all", "mz", "sum"],
            ],
            default_slice=["all"],
            slice=[
                ["all", "x", "range", str(0.1), str(0.7)],
                ["sliced", "x", "range", str(0.1), str(0.7)],
                ["sliced", "y", "range", str(0.1), str(0.7)],
                ["sliced", "mz", "range", str(12), str(18)],
            ],
        )
        nexus_slice.process(nexus_args, {})
        out_file = nx_dir / "all.nxs"
        assert out_file.exists()
        with h5py.File(out_file, "r") as fle:
            assert "/entry/data/signal" in fle
            assert "/entry/data/x" in fle
            assert "/entry/data/y" not in fle
            assert "/entry/data/mz" not in fle
            assert "/entry/data/error" not in fle
            data = fle["/entry/data/signal"][...]
            assert data.shape == (6,)
            assert np.all(data == np.sum(man_data_and_nexus[0].dense[1:7], axis=(1, 2)))

        out_file = nx_dir / "sliced.nxs"
        assert out_file.exists()
        with h5py.File(out_file, "r") as fle:
            assert "/entry/data/signal" in fle
            assert "/entry/data/x" in fle
            assert "/entry/data/y" in fle
            assert "/entry/data/mz" in fle
            assert "/entry/data/error" in fle
            data = fle["/entry/data/signal"][...]
            assert data.shape == (6, 6, 60)
            assert fle["/entry/data/x"].shape == (6,)
            assert fle["/entry/data/y"].shape == (6,)
            assert fle["/entry/data/mz"].shape == (60,)
            assert np.all(data == man_data_and_nexus[0].dense[1:7, 1:7, 120:180])

    finally:
        filename.unlink()


@pytest.mark.skip(reason="I do not, yet, know how to test this.")
def test_view_type(man_data_and_nexus, nx_dir):
    pass


@pytest.mark.skip(reason="I do not, yet, know how to test this.")
def test_summary_type(man_data_and_nexus, nx_dir):
    pass
