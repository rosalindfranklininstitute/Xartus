# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

from pathlib import Path
from ms_nexus_tools.lib.exceptions import InvalidEntryError
from typing import Any, Iterable, cast

import dask.array as da
import h5py
import xarray as xr
from xarray.backends import BackendEntrypoint

from icecream import ic


def insert_into_axes(axes: list[str], axis: list[str]) -> list[str]:
    """
    This inserts the values from axis into axes, trying to respect the order.
    New elements are inserted as late as possible before any existing elements.
    >>> axes = [
    ...     ["x", "y"],
    ...     ["x", "y", "mz"],
    ...     ["y", "iim", "mz"],
    ...     ["x", "c", "mz"],
    ... ]
    >>> all = []
    >>> for axis in axes:
    ...     print(all)
    ...     all = insert_into_axes(all, axis)
    []
    ['x', 'y']
    ['x', 'y', 'mz']
    ['x', 'y', 'iim', 'mz']
    >>> print(all)
    ['x', 'y', 'iim', 'c', 'mz']
    """
    out = axes.copy()
    inx = [axes.index(a) if a in out else -1 for a in axis]
    offset = 0
    n = len(out)
    for ii, (a, vii) in enumerate(zip(axis, inx, strict=True)):
        if vii < 0:
            prev_i = inx[max(0, ii - 1)] + 1
            next_i = inx[min(n - 1, ii + 1)]
            out.insert(offset + max(prev_i, next_i), a)
            offset += 1
    return out


class NexusEntrypoint(BackendEntrypoint):
    def __init__(self):
        self.nx_file: None | h5py.File = None
        self.filename: None | Path = None

    def __del__(self):
        self._close()

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        entry_path: None | str = None,
    ) -> xr.Dataset:
        try:
            should_close = self._open(filename_or_obj)
            assert self.nx_file is not None

            if entry_path is None:
                path = "/"
                while "signal" not in self.nx_file[path].attrs:
                    if "default" not in self.nx_file[path].attrs:
                        raise InvalidEntryError("Could not find default signal.")
                    path = f"{path.removesuffix('/')}/{self.nx_file[path].attrs['default']}"
                if "signal" not in self.nx_file[path].attrs:
                    raise InvalidEntryError("Could not find default signal.")
                entry_path = path

            entries: dict[str, xr.DataArray] = {}

            entry_array = self._read_nxdata(entry_path)
            entries[cast(str, entry_array.name)] = entry_array

            ds = xr.Dataset(data_vars=entries)

            if should_close:
                ds.set_close(self._close)

        except:
            self._close()
            raise
        else:
            return ds

    def open_datatree(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        root: str = "/",
    ) -> xr.DataTree:
        try:
            should_close = self._open(filename_or_obj)
            assert self.nx_file is not None

            root_data = self._read_all_data_on_nxentry(root)
            tree_root = xr.DataTree(dataset=root_data)
            base_path = root.removesuffix("/")
            for path in self.nx_file[root]:
                entry_path = f"{base_path}/{path}"
                nx_class = self.nx_file[entry_path].attrs["NX_class"]
                match nx_class:
                    case "NXdata":
                        pass  # Already read
                    case "NXentry" | "NXsubentry":
                        tree_root[path] = self.open_datatree(
                            self.nx_file, root=entry_path
                        )
                    case _:
                        tree_root[path] = xr.DataTree(
                            xr.Dataset(attrs=dict(self.nx_file[entry_path].attrs))
                        )

            if should_close:
                tree_root.set_close(self._close)
        except:
            self._close()
            raise
        else:
            return tree_root

    def _read_nxdata(self, data_path) -> xr.DataArray:
        assert self.nx_file is not None

        if self.nx_file[data_path].attrs["NX_class"] != "NXdata":
            message = f"Expected {self.filename}:{data_path} to be NXdata."
            raise InvalidEntryError(message)

        path_attrs = dict(self.nx_file[data_path].attrs)
        del path_attrs["NX_class"]
        del path_attrs["axes"]
        del path_attrs["signal"]

        signal = self.nx_file[data_path].attrs["signal"]
        axes: list[str] = self.nx_file[data_path].attrs["axes"]

        inner_coords: dict[str, tuple[tuple[str, ...], Any, dict[str, Any]]] = {}
        for name in self.nx_file[data_path]:
            index_path = f"{name}_indices"
            if index_path in self.nx_file[data_path].attrs:
                del path_attrs[index_path]

                inx = self.nx_file[data_path].attrs[index_path]
                if isinstance(inx, int):
                    inx = [inx]
                if not isinstance(inx, Iterable):
                    raise TypeError(
                        f"Expected {data_path}/{index_path} to be int or list[int] but found {type(inx)}"
                    )

                coord_dims = tuple([axes[i] for i in inx])

                axis_path = f"{data_path}/{name}"
                axis_attrs = dict(self.nx_file[axis_path].attrs)

                chunks = self.nx_file[axis_path].chunks
                if chunks is None:
                    if self.nx_file[axis_path].dtype == "O":
                        chunks = self.nx_file[axis_path].shape
                    else:
                        chunks = "auto"
                values = da.from_array(self.nx_file[axis_path], chunks=chunks)

                if "unit" in axis_attrs:
                    axis_attrs["units"] = axis_attrs["unit"]
                    del axis_attrs["unit"]

                inner_coords[name] = (
                    coord_dims,
                    values,
                    axis_attrs,
                )
        signal_path = f"{data_path}/{signal}"
        chunks = self.nx_file[signal_path].chunks or "auto"
        return xr.DataArray(
            da.from_array(self.nx_file[signal_path], chunks=chunks),
            name=signal,
            dims=axes,
            coords=inner_coords,
            attrs=path_attrs,
        )

    def _read_all_data_on_nxentry(self, entry_path: str) -> xr.Dataset:
        assert self.nx_file is not None

        root_attrs = dict(self.nx_file[entry_path].attrs)
        if "NX_class" not in self.nx_file[entry_path].attrs:
            if "HDF5_Version" not in self.nx_file[entry_path].attrs:
                message = f"Expected {self.filename}:{entry_path} to be file root or have NX_class."
                raise InvalidEntryError(message)
        elif self.nx_file[entry_path].attrs["NX_class"] not in (
            "NXentry",
            "NXsubentry",
        ):
            message = (
                f"Expected {self.filename}:{entry_path} to be NXentry or NXsubentry."
            )
            raise InvalidEntryError(message)
        else:
            del root_attrs["NX_class"]

        entries: dict[str, xr.DataArray] = {}

        base_path = entry_path.removesuffix("/")

        for path in self.nx_file[entry_path]:
            sub_path = f"{base_path}/{path}"
            ic(base_path, path, sub_path)
            nx_class = self.nx_file[sub_path].attrs["NX_class"]
            if nx_class == "NXdata":
                entry_array = self._read_nxdata(sub_path)
                entry_name = path
                entries[entry_name] = entry_array
        return xr.Dataset(data_vars=entries, attrs=root_attrs)

    def _open(self, filename_or_obj) -> bool:
        """
        Opens the file, if appropriate.
        Returns True if the file was opened. False means that the file obj passed is being managed elsewhere.

        self.nx_file is populated if the file is opened.
        self.filename is always populated, even if the file was not reopened.
        """
        should_close = True
        if isinstance(filename_or_obj, h5py.File):
            should_close = False
            if self.nx_file is None:
                self.nx_file = filename_or_obj
            elif self.nx_file != filename_or_obj:
                raise ValueError("The passed in file is different?!")
            self.filename = filename_or_obj.filename
        else:
            self.nx_file = h5py.File(filename_or_obj, "r")
            ic("open file", filename_or_obj, self.nx_file)
            self.filename = filename_or_obj
        return should_close

    def _close(self) -> None:
        if self.nx_file is not None:
            self.nx_file.close()

    open_dataset_parameters = ["filename_or_obj", "drop_variables", "entry_path"]
    open_datatree_parameters = ["filename_or_obj", "drop_variables", "root"]

    def guess_can_open(self, filename_or_obj) -> bool:
        try:
            ext = Path(filename_or_obj).suffix
        except TypeError:
            return False
        return ext in {".nxs", ".nexus", ".nx"}

    description = "Use NeXus files in Xarray"

    url = "https://link_to/your_backend/documentation"
