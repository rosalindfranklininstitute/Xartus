# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from shutil import rmtree

import numpy as np
import h5py

from PIL import Image as PILImage

from ms_nexus_tools.api import nexus_slice, data_convert
from ms_nexus_tools.lib.data_source import Axis, AxisType
from ms_nexus_tools.api.nexus_slice import (
    ActionType,
    GroupType,
    MissingArgumentError,
)

from nexus_pixel_man_test_data import Man2DDataSource, ManData, data_files


import pytest


@pytest.fixture(scope="module")
def man_file():
    return data_files()["man1"]


@pytest.fixture(scope="module")
def man_data_and_nexus(man_file):
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


@pytest.fixture(scope="module")
def man_images():
    filenames = [
        Path(__file__).parent / "data" / f"reference_man{ii}.2d.png"
        for ii in range(1, 5)
    ]
    return [PILImage.open(filename) for filename in filenames]


@pytest.fixture(scope="module")
def hand_spectra():
    filenames = [
        Path(__file__).parent / "data" / f"reference_hand{ii}.1d.png"
        for ii in range(1, 4)
    ]
    return [PILImage.open(filename) for filename in filenames]


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
        plot_image=False,
        plot_spectrum=False,
    )
    nexus_slice.process(process_args, {})

    man3_file = nx_dir / "man3.1d.png"
    assert not man3_file.exists()

    spectra_file = nx_dir / "spectra.1d.png"
    assert not spectra_file.exists()

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
        assert data.shape == (240,)
        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense, axis=(0, 1))
        )

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
        assert data.shape == (8, 8)

        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
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
        plot_image=False,
        plot_spectrum=False,
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
        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense, axis=(0, 1))
        )

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

        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
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
        plot_image=False,
        plot_spectrum=False,
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
        plot_image=False,
        plot_spectrum=False,
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
        np.testing.assert_allclose(data, man_data_and_nexus[0].dense)

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
        np.testing.assert_allclose(data, man_data_and_nexus[0].dense[:, :, 120:180])

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
        np.testing.assert_allclose(data, man_data_and_nexus[0].dense[:, :, 120:180])

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

        np.testing.assert_allclose(
            data[:, :, 0], man_data_and_nexus[0].dense[:, :, 150]
        )


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
        plot_image=False,
        plot_spectrum=False,
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
        np.testing.assert_allclose(data, man_data_and_nexus[0].dense[yy, 2:6, 120:180])

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
        np.testing.assert_allclose(data, man_data_and_nexus[0].dense[2, yy, :])


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
        plot_image=False,
        plot_spectrum=False,
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
        np.testing.assert_allclose(data, np.sum(man_data_and_nexus[0].dense, axis=2))

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
        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
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
        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
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
        np.testing.assert_allclose(data[:, :], man_data_and_nexus[0].dense[:, :, 150])

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
        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense[:, :, 150], axis=(0, 1))
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
        plot_image=False,
        plot_spectrum=False,
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
        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense, axis=(0, 1))
        )

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

        np.testing.assert_allclose(
            data, np.sum(man_data_and_nexus[0].dense[:, :, 120:180], axis=2)
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
        plot_image=False,
        plot_spectrum=False,
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
        np.testing.assert_allclose(data, np.sum(man_data_and_nexus[0].dense))


def test_binned_axis(man_data_and_nexus, nx_dir, man_file):
    filename = Path(__file__).parent / "man_binned.nxs"
    try:
        man_data = ManData()
        man_data_source = Man2DDataSource(
            man_data,
            supplimentary_axes=[
                Axis("mz", 2, AxisType.BINNED, np.int16, "mz"),
                Axis("error", 2, AxisType.BINNED, np.int16, ""),
            ],
            multipliers=dict(x=0.1, y=0.1, mz=0.1, error=1.0),
        )
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
            plot_image=False,
            plot_spectrum=False,
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
            np.testing.assert_allclose(
                data, np.sum(man_data_and_nexus[0].dense[1:7], axis=(1, 2))
            )

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
            np.testing.assert_allclose(
                data, man_data_and_nexus[0].dense[1:7, 1:7, 120:180]
            )

    finally:
        filename.unlink()


def test_view_2d_plot(man_data_and_nexus, nx_dir, man_images):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["man1", "mz", "sum"],
            ["man2", "mz", "sum"],
            ["man2", "x", "leave"],
            ["man3", "mz", "sum"],
            ["man4", "mz", "sum"],
        ],
        default_slice=["all"],
        slice=[
            ["man1", "mz", "range", str(0), str(6)],
            ["man2", "mz", "range", str(6), str(12)],
            ["man2", "x", "all"],
            ["man3", "mz", "range", str(12), str(18)],
            ["man4", "mz", "range", str(18), str(24)],
        ],
        plot_image=True,
        plot_spectrum=False,
    )
    nexus_slice.process(process_args, {})

    for ii in range(4):
        man_image_file = nx_dir / f"man{ii + 1}.2d.png"
        assert man_image_file.exists()

        man_image = np.array(PILImage.open(man_image_file))
        np.testing.assert_allclose(man_image, man_images[ii])


def test_view_1d_plot(man_data_and_nexus, nx_dir, hand_spectra):
    process_args = nexus_slice.ProcessArgs(
        in_path=man_data_and_nexus[1],
        out_dir=nx_dir,
        default_group_type=GroupType.View,
        default_paths=["/entry/images/data/", "/entry/spectra/data/"],
        default_action=ActionType.Leave,
        action=[
            ["hand2", "y", "leave"],
            ["hand2", "mz", "leave"],
            ["hand2", "x", "leave"],
        ],
        default_slice=["all"],
        slice=[
            ["hand1", "x", "value", str(0.1)],
            ["hand1", "y", "value", str(0.5)],
            ["hand2", "y", "value", str(0.4)],
            ["hand2", "mz", "all"],
            ["hand2", "x", "value", str(0.0)],
            ["hand3", "x", "value", str(0.1)],
            ["hand3", "y", "value", str(0.3)],
        ],
        plot_image=True,
        plot_spectrum=True,
    )
    nexus_slice.process(process_args, {})

    for ii in range(3):
        hand_image_file = nx_dir / f"hand{ii + 1}.1d.png"
        assert hand_image_file.exists()

        hand_image = np.array(PILImage.open(hand_image_file))
        np.testing.assert_allclose(hand_image, hand_spectra[ii])


@pytest.mark.skip(reason="Not yet implemented")
def test_view_loop_2d_plot(man_data_and_nexus, nx_dir, hand_spectra):
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_view_loop_1d_plot(man_data_and_nexus, nx_dir, hand_spectra):
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_summary_type(man_data_and_nexus, nx_dir):
    pass
