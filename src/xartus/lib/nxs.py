# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
import h5py

from typing import Any, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt

from .bounds import Shape


def create_nxfield(
    parent: h5py.Group,
    name: str,
    dtype: npt.DTypeLike | None = None,
    shape: Shape | None = None,
    compression: str | None = None,
    compression_opts: Any = None,
    chunks: Shape | bool | None = None,
    value: Any = None,
    **kwargs,
) -> h5py.Dataset:
    assert value is not None or (dtype is not None and shape is not None)
    ds = parent.create_dataset(
        name=name,
        shape=shape,
        dtype=dtype,
        data=value,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        **kwargs,
    )
    ds.attrs["NX_class"] = "NXfield"
    return ds


def create_nxgroup(
    parent: h5py.Group,
    name: str,
    nx_class: str = "NXobject",
    **kwargs,
) -> h5py.Group:
    grp = parent.create_group(name, **kwargs)
    grp.attrs["NX_class"] = nx_class
    return grp


def _check_valid_value(
    value: np.integer | np.floating | None, dtype: npt.DTypeLike
) -> bool:
    if value is None:
        return True

    dtype = np.dtype(dtype)

    if np.issubdtype(dtype, np.integer):
        if not np.isfinite(value):
            return False

        info = np.iinfo(dtype)
        return info.min <= value <= info.max

    if np.issubdtype(dtype, np.floating):
        if not np.isfinite(value):
            return True

        info = np.finfo(dtype)
        return -info.max <= value <= info.max

    return False


@dataclass
class NxAxis:
    name: str
    indices: list[int]
    dtype: npt.DTypeLike | None = None
    shape: Shape | None = None
    fillvalue: np.integer | np.floating | None = None
    units: str | None = None
    chunk_shape: Shape | None = None
    compression: str | None = None
    compression_opts: Any = None
    values: np.ndarray | None = None

    def __post_init__(self):
        if self.values is not None:
            if self.dtype is not None:
                if not np.issubdtype(self.values.dtype, self.dtype):
                    raise ValueError("Provided values.dtype and dtype did not match.")
            else:
                self.dtype = self.values.dtype
            if self.shape is not None:
                if self.values.shape != self.shape:
                    raise ValueError("Provided values.shape and shape did not match.")
            else:
                self.shape = self.values.shape
        else:
            if self.dtype is None:
                raise ValueError(
                    "Must provide either values or dtype (or both, matching)"
                )
            if self.shape is None:
                raise ValueError(
                    "Must provide either values or shape (or both, matching)"
                )

        if not _check_valid_value(self.fillvalue, self.dtype):
            raise ValueError("Provided fill value does not have the correct dtype.")

    def add_to_group(self, group: h5py.Group) -> None:
        group.attrs[f"{self.name}_indices"] = self.indices
        kwargs = {}
        if self.fillvalue is not None:
            kwargs["fillvalue"] = self.fillvalue
        fld = create_nxfield(
            parent=group,
            name=self.name,
            dtype=self.dtype,
            shape=self.shape,
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=self.chunk_shape,
            value=self.values,
            **kwargs,
        )
        fld.attrs["units"] = self.units

    def copy_with_incremented_indices(self, inc: int) -> "NxAxis":
        return NxAxis(
            name=self.name,
            indices=[i + inc for i in self.indices],
            dtype=self.dtype,
            shape=self.shape,
            fillvalue=self.fillvalue,
            units=self.units,
            chunk_shape=self.chunk_shape,
            compression=self.compression,
            compression_opts=self.compression_opts,
            values=self.values,
        )


class NxAxes(list[list[NxAxis]]):
    def default_list(self) -> list[str]:
        return [v[0].name for v in self]

    def list_all(self) -> list[NxAxis]:
        results = []
        for v in self:
            results.extend(v)
        return results

    def add_to_group(self, group: h5py.Group) -> None:
        group.attrs["axes"] = self.default_list()
        for ax in self.list_all():
            ax.add_to_group(group)


class FieldOptions(NamedTuple):
    compression: Any
    compression_opts: int | None
    max_bytes_per_chunk: int
    shuffle: bool


class NexusFile:
    def __init__(self, filename: Path, mode: str = "r", locking=None):
        self.filename = filename

        self._mode = mode
        self._file = h5py.File(filename, mode, locking=locking)

        if mode == "w" or mode == "w-" or mode == "x":
            self._file.attrs["NX_class"] = "NXroot"
            self.entry = self._file.create_group("entry")
            self.entry.attrs["NX_class"] = "NXentry"
            self._file.attrs["default"] = "entry"

            create_nxgroup(self.entry, "instrument", nx_class="NXinstrument")
            create_nxgroup(self.entry, "experiment", nx_class="NXparameters")

        else:
            self.entry = self._file["entry"]

    def close(self) -> None:
        self._file.close()

    def as_context(self) -> h5py.File:
        return self._file

    def _get_instrument(self) -> h5py.Group:
        return self.entry["instrument"]

    def _set_instrument(self, value: h5py.Group) -> None:
        assert isinstance(value, h5py.Group)
        self.entry["instrument"] = value

    instrument = property(
        _get_instrument,
        _set_instrument,
        None,
        "The instrument group",
    )

    def _get_experiment(self) -> h5py.Group:
        return self.entry["experiment"]

    def _set_experiment(self, value: h5py.Group) -> None:
        assert isinstance(value, h5py.Group)
        self.entry["experiment"] = value

    experiment = property(
        _get_experiment,
        _set_experiment,
        None,
        "The experiment group",
    )
