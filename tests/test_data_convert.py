# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
import copy
import itertools
from functools import reduce

from pathlib import Path

import numpy as np
import h5py

from ms_nexus_tools.lib.chunker import count_chunks_to_cover
from ms_nexus_tools.api import data_convert
from ms_nexus_tools.lib.data_source import Axis, AxisType

from nexus_pixel_man_test_data import Man2DDataSource, ManData, data_files

import pytest


class Parameters:
    def __init__(
        self,
        chunks: bool = False,
        memory: bool = False,
        bins: bool = False,
        dtypes: bool = False,
        mult: bool = False,
    ):
        self.chunks = chunks
        self.memory = memory
        self.bins = bins
        self.dtypes = dtypes
        self.mult = mult

    def args(self) -> str:
        name = []
        if self.chunks:
            name.append("chunk_max_byte_count")
        if self.memory:
            name.append("memory_max_byte_count")
        if self.bins:
            name.append("mz_binning")
        if self.dtypes:
            name.append("mz_dtype")
        if self.mult:
            name.append("mult")
        return ",".join(name)

    def _param(self, values) -> object | None:
        parts = []
        value_copy = list(copy.copy(values))
        if self.chunks:
            chunk = value_copy.pop(0)
            parts.append("m-chunk" if chunk == 240 * 2 else "s-chunk")
            if self.memory:
                value = value_copy.pop(0)
                if value < chunk:
                    return None
                if value < 8 * 8 * 240 * 2:
                    parts.append("m-mem")
                else:
                    parts.append("s-mem")
        elif self.memory:
            value = value_copy.pop(0)
            if value < 8 * 8 * 240 * 2:
                return None
            parts.append("s-mem")
        if self.bins:
            parts.append(f"bin{value_copy.pop(0)}")
        if self.dtypes:
            parts.append("int" if value_copy.pop(0) == np.int16 else "float")
        if self.mult:
            parts.append(f"mult{value_copy.pop(0)}")
        return pytest.param(*values, id="_".join(parts))

    def params(self) -> list:
        values = []
        if self.chunks:
            values.append([240 * 2, 1024 * 1024])
        if self.memory:
            values.append([240 * 4, 1024 * 1024 * 1024])
        if self.bins:
            values.append([1, 2, 3])
        if self.dtypes:
            values.append([np.int16, np.float32])
        if self.mult:
            values.append([0.1, 1.0, 1.1])
        return [
            p
            for p in [self._param(v) for v in itertools.product(*values)]
            if p is not None
        ]


@pytest.fixture(scope="module")
def man_data():
    return ManData()


@pytest.fixture
def nx_file():
    filename = Path(__file__).parent / "test.nxs"
    if filename.exists():
        filename.unlink()
    yield filename
    filename.unlink()


@pytest.fixture
def man_file():
    return data_files()["man1"]


def get_dataset_total_and_used_chunks(fle, name):
    shape = fle[name].shape
    chunks = fle[name].chunks
    dsid = fle[name].id
    n = dsid.get_num_chunks()
    count = count_chunks_to_cover(shape, chunks)
    total_chunks = reduce(lambda x, y: x * y, count)
    return total_chunks, n


def check_basic_axis_correct(
    fle, man_data_source: Man2DDataSource, max_chunk_item_count
):
    for data_name in ["images", "spectra"]:
        assert f"/entry/{data_name}/data/signal" in fle
        for axis in man_data_source.axes.values():
            assert f"/entry/{data_name}/data/{axis.name}" in fle
            assert fle[f"/entry/{data_name}/data/{axis.name}"].dtype == axis.dtype
            assert f"{axis.name}_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert (
                fle[f"/entry/{data_name}/data"].attrs[f"{axis.name}_indices"]
                == axis.primary_axis
            )

        assert np.all(fle[f"/entry/{data_name}/data"].attrs["axes"] == ["x", "y", "mz"])

        actual_item_count = np.prod(fle[f"/entry/{data_name}/data/signal"].chunks)
        assert actual_item_count <= max_chunk_item_count


def check_dense_corret(fle, man_data_source: Man2DDataSource):
    for data_name in ["images", "spectra"]:
        assert f"/entry/{data_name}/data/signal" in fle
        for axis in man_data_source.axes.values():
            assert f"/entry/{data_name}/data/{axis.name}" in fle
            assert fle[f"/entry/{data_name}/data/{axis.name}"].dtype == axis.dtype
            assert f"{axis.name}_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert (
                fle[f"/entry/{data_name}/data"].attrs[f"{axis.name}_indices"]
                == axis.primary_axis
            )

        assert np.all(fle[f"/entry/{data_name}/data"].attrs["axes"] == ["x", "y", "mz"])

        assert (
            fle[f"/entry/{data_name}/data/signal"].shape
            == man_data_source.man_data.shape
        )

        np.testing.assert_allclose(
            fle[f"/entry/{data_name}/data/signal"][:, :, :],
            man_data_source.man_data.dense,
        )

    total_image = np.sum(man_data_source.man_data.dense, axis=2)
    assert fle["/entry/total_image/data/signal"].shape == (2, *total_image.shape)
    np.testing.assert_allclose(
        fle["/entry/total_image/data/signal"][1, :, :], total_image
    )

    total_spectra = np.sum(man_data_source.man_data.dense, axis=(0, 1))
    assert fle["/entry/total_spectra/data/signal"].shape == (2, *total_spectra.shape)
    np.testing.assert_allclose(
        fle["/entry/total_spectra/data/signal"][1, :], total_spectra
    )


def check_binned_correct(
    fle,
    man_data_source: Man2DDataSource,
    max_count_per_bin,
    should_be_sparse,
):
    any_binned = False
    has_some_sparsity = False
    for data_name in ["images", "spectra"]:
        name = f"/entry/{data_name}/data/signal"
        assert name in fle

        total_chunks, n = get_dataset_total_and_used_chunks(fle, name)
        has_some_sparsity |= n < total_chunks

        data_part = fle[name]
        for ii in range(4):
            np.testing.assert_allclose(
                np.sum(
                    data_part[
                        :,
                        :,
                        (60 // max_count_per_bin) * ii : (60 // max_count_per_bin)
                        * (ii + 1),
                    ],
                    axis=2,
                ),
                np.sum(
                    man_data_source.man_data.dense[:, :, 60 * ii : 60 * (ii + 1)],
                    axis=2,
                ),
            )

        for axis in man_data_source.axes.values():
            if axis.axis_type == AxisType.BINNED:
                any_binned = True
                assert (
                    fle[f"/entry/{data_name}/data/{axis.name}_exact"].dtype
                    == axis.dtype
                )
                assert all(
                    fle[f"/entry/{data_name}/data"].attrs[f"{axis.name}_exact_indices"]
                    == list(range(4))
                )
                exact = fle[f"/entry/{data_name}/data/{axis.name}_exact"][:, :, :]
                desired = np.tile(
                    fle[f"/entry/{data_name}/data/{axis.name}"],
                    (8, 8, 1),
                )
                if np.issubdtype(axis.dtype, np.integer):
                    mask = exact == 0
                    desired[mask] = 0
                else:
                    mask = np.isnan(exact)
                    desired[mask] = np.nan

                np.testing.assert_allclose(exact, desired)

    total_spectra = fle["/entry/total_spectra/data/signal"]
    for ii in range(4):
        fle_slice = total_spectra[
            :,
            (60 // max_count_per_bin) * ii : (60 // max_count_per_bin) * (ii + 1),
        ]
        data_slice = man_data_source.man_data.dense[:, :, 60 * ii : 60 * (ii + 1)]
        np.testing.assert_allclose(
            np.sum(fle_slice[1, :]),
            np.sum(data_slice),
        )
        np.testing.assert_allclose(
            np.max(fle_slice[0, :]),
            np.max(data_slice),
        )

    if should_be_sparse:
        assert has_some_sparsity

    if any_binned:
        assert "/entry/item_counts/data/signal" in fle
        assert np.all(fle["/entry/item_counts/data/signal"][...] <= max_count_per_bin)
        assert np.max(fle["/entry/item_counts/data/signal"][...]) == max_count_per_bin
        for axis in man_data_source.axes.values():
            assert f"/entry/item_counts/data/{axis.name}" in fle
            assert f"/entry/item_counts/data/{axis.name}_exact" not in fle

    assert "/entry/item_counts_total_spectra" in fle
    assert "/entry/item_counts_total_image" not in fle


dense_single_axis_params = Parameters(chunks=True)


@pytest.mark.parametrize(
    dense_single_axis_params.args(), dense_single_axis_params.params()
)
def test_dense_single_axis(nx_file, man_file, man_data, chunk_max_byte_count):
    man_data_source = Man2DDataSource(man_data)

    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=nx_file,
        chunk_max_byte_count=chunk_max_byte_count,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_basic_axis_correct(
            fle, man_data_source, process_args.chunk_max_byte_count / 2
        )
        check_dense_corret(fle, man_data_source)


def test_dense_multi_axis_single_chunk(nx_file, man_file, man_data):
    man_data_source = Man2DDataSource(
        man_data,
        supplimentary_axes=[
            Axis("time", 0, AxisType.EXACT, np.int16, "s"),
            Axis("error", 2, AxisType.EXACT, np.int16, ""),
        ],
    )

    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_basic_axis_correct(
            fle, man_data_source, process_args.chunk_max_byte_count / 2
        )
        check_dense_corret(fle, man_data_source)


def test_all_dimensions_binned(nx_file, man_file, man_data):
    man_data_source = Man2DDataSource(
        man_data,
        supplimentary_axes=[
            Axis("x", 0, AxisType.BINNED, np.float32, "m"),
            Axis("y", 1, AxisType.BINNED, np.float32, "m"),
            Axis("mz", 2, AxisType.BINNED, np.float32, "m"),
        ],
        binning={"x": 2, "y": 2, "mz": 2},
        multipliers={"x": 0.1, "y": 0.2, "mz": 0.3},
    )

    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_basic_axis_correct(
            fle, man_data_source, process_args.chunk_max_byte_count / 2
        )
        for data_name in ["images", "spectra"]:
            name = f"/entry/{data_name}/data/signal"
            assert name in fle
            for axis in man_data_source.axes.values():
                assert (
                    fle[f"/entry/{data_name}/data/{axis.name}_exact"].dtype
                    == axis.dtype
                )
                assert all(
                    fle[f"/entry/{data_name}/data"].attrs[f"{axis.name}_exact_indices"]
                    == list(range(4))
                )
                data_part = fle[f"/entry/{data_name}/data/signal"]
                assert np.sum(data_part) == np.sum(man_data_source.man_data.dense)

        assert np.all(fle["/entry/item_counts/data/signal"][...] <= 8)
        assert np.max(fle["/entry/item_counts/data/signal"][...]) == 8


def test_sparse_all_exact(nx_file, man_file, man_data):
    man_data_source = Man2DDataSource(
        man_data,
        force_sparse=True,
    )

    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=nx_file,
        chunk_max_byte_count=240 * 2,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_basic_axis_correct(
            fle, man_data_source, process_args.chunk_max_byte_count / 2
        )
        check_dense_corret(
            fle,
            man_data_source,
        )
        for name in fle["/entry/images/data/"]:
            assert "_exact" not in name


binned_params = Parameters(chunks=True, memory=True, bins=True, dtypes=True, mult=True)


@pytest.mark.parametrize(binned_params.args(), binned_params.params())
def test_binned(
    nx_file,
    man_file,
    man_data,
    chunk_max_byte_count,
    memory_max_byte_count,
    mz_binning,
    mz_dtype,
    mult,
):
    man_data_source = Man2DDataSource(
        man_data,
        supplimentary_axes=[Axis("mz", 2, AxisType.BINNED, mz_dtype, "mz")],
        binning={"mz": mz_binning},
        multipliers={"mz": mult},
    )

    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=nx_file,
        chunk_max_byte_count=chunk_max_byte_count,
        memory_max_byte_count=memory_max_byte_count,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_basic_axis_correct(
            fle,
            man_data_source,
            process_args.chunk_max_byte_count / 2,
        )
        check_binned_correct(
            fle,
            man_data_source,
            max_count_per_bin=mz_binning,
            should_be_sparse=chunk_max_byte_count == 240 * 2,
        )


def test_binned_multi_continuous_axis_single_chunk(nx_file, man_file, man_data):
    man_data_source = Man2DDataSource(
        man_data,
        supplimentary_axes=[
            Axis("mz", 2, AxisType.BINNED, np.int16, "s"),
            Axis("time", 0, AxisType.EXACT, np.int16, "s"),
        ],
    )

    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_basic_axis_correct(
            fle,
            man_data_source,
            process_args.chunk_max_byte_count / 2,
        )
        check_binned_correct(
            fle,
            man_data_source,
            max_count_per_bin=1,
            should_be_sparse=False,
        )


binned_multi_axis_params = Parameters(chunks=True, bins=True, dtypes=True, mult=True)


@pytest.mark.parametrize(
    binned_multi_axis_params.args(), binned_multi_axis_params.params()
)
def test_binned_multi_axis(
    nx_file, man_file, man_data, chunk_max_byte_count, mz_binning, mz_dtype, mult
):
    man_data_source = Man2DDataSource(
        man_data,
        supplimentary_axes=[
            Axis("mz", 2, AxisType.BINNED, mz_dtype, "mz"),
            Axis("error", 2, AxisType.BINNED, np.float32, "%"),
        ],
        binning={"mz": mz_binning, "error": mz_binning},
        multipliers={"mz": mult, "error": mult / 2},
    )

    process_args = data_convert.ProcessArgs(
        in_path=man_file,
        out_path=nx_file,
        chunk_max_byte_count=chunk_max_byte_count,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_basic_axis_correct(
            fle,
            man_data_source,
            process_args.chunk_max_byte_count / 2,
        )
        check_binned_correct(
            fle,
            man_data_source,
            max_count_per_bin=mz_binning,
            should_be_sparse=chunk_max_byte_count == 240 * 2,
        )
