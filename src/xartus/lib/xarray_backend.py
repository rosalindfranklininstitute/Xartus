# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

import logging
from xartus.lib.utils import DeferredAction
from pathlib import Path
from .exceptions import InvalidEntryError, EntryExistsError
from typing import Any, Iterable

import dask.array as da
import h5py
import xarray as xr
from xarray.backends import BackendEntrypoint
import numpy as np

logger = logging.getLogger(__name__)


class ArraySignalName:
    """
    A class used for providing a name for the field on the NXdata.

    A NXdata group has a signal field. This is pointed to in the @signal attribute.
    It can have any name.
    However, `xarray.DataArray` has the same name as the name of the NXdata:
    >>> dims = ["x", "y", "z"]
    >>> shape = [3, 10, 1000]
    >>> ints = np.arange(0, np.prod(shape)).reshape(shape)
    >>> ds = xr.Dataset(
    ...    {
    ...        "ints": xr.DataArray(
    ...            ints,
    ...            dims=dims,
    ...            coords={key: np.arange(s) for key, s in zip(dims, shape, strict=True)},
    ...            attrs={"min": 0, "max": np.prod(shape)},
    ...         ),
    ...     }
    ... )
    >>> assert ds.ints.name == "ints"

    Thus if `xarray.DataArray.name` is used for the signal you end up with:
    ```
    > /entry/sub/ints: <HDF5 group "/entry/sub/ints" (6 members)>
    | - @NX_class: NXdata
    | - @signal: ints
    | > /entry/sub/ints/ints: <HDF5 dataset "ints": shape (3, 10, 1000), type "<i2">
    ```

    This class has a __getitem__ method that allows rename this. It is passed the group path and the name of the NXdata entry and should return the name of the signal field.
    The default implementation always gives:
    >>> signal_name = ArraySignalName()
    >>> signal_name['/entry/spectra', 'ints']
    'signal'

    which would give:
    ```
    > /entry/sub/ints: <HDF5 group "/entry/sub/ints" (6 members)>
    | - @NX_class: NXdata
    | - @signal: ints
    | > /entry/sub/ints/ints: <HDF5 dataset "ints": shape (3, 10, 1000), type "<i2">
    ```

    """

    def __getitem__(self, name: tuple[str, str]) -> str:
        """
        Takes in group_path and NXdata name, and should return the name of the signal field.


        Args:
            name: A tuple of (group_path, NXdata_name)
        Returns:
            The name of the signal field ont he NXdata
        """
        return "signal"


def _write_dataarray(
    dataarray: xr.DataArray, group: h5py.Group, name: str, signal_name: ArraySignalName
) -> None:
    if name in group:
        message = f"Cannot create {name} in {group.name}: it already exists."
        raise EntryExistsError(message)

    nx_data: h5py.Group = group.create_group(name)
    nx_data.attrs["NX_class"] = "NXdata"

    for k, v in dataarray.attrs.items():
        nx_data.attrs[k] = v

    if "signal" in dataarray.attrs:
        sname = dataarray.attrs["signal"]
    else:
        sname = signal_name[group.name, name]
    signal = nx_data.create_dataset(
        sname,
        shape=dataarray.shape,
        dtype=dataarray.dtype,
        chunks=dataarray.chunks,
    )
    nx_data.attrs["signal"] = sname

    chunks = dataarray.shape if dataarray.chunks is None else dataarray.chunks
    if isinstance(dataarray.data, da.Array):
        a = dataarray.data
    else:
        a = da.from_array(dataarray.data, chunks=chunks)
    a.store(signal, lock=True, compute=True)

    nx_data.attrs["axes"] = list(dataarray.dims)

    for coord in dataarray.coords.values():
        indices = [int(ii) for ii, d in enumerate(dataarray.dims) if d in coord.dims]
        nx_data.attrs[f"{coord.name}_indices"] = indices
        axis = nx_data.create_dataset(
            coord.name,
            shape=coord.shape,
            dtype=coord.dtype,
            chunks=coord.chunks,
        )
        chunks = coord.shape if coord.chunks is None else coord.chunks
        if isinstance(coord.data, da.Array):
            a = coord.data
        else:
            a = da.from_array(coord.data, chunks=chunks)
        a.store(
            axis,
            lock=True,
            compute=True,
        )

        for k, v in coord.attrs.items():
            axis.attrs[k] = v


def _write_dataset(
    dataset: xr.Dataset | xr.DataTree,
    group: h5py.Group,
    nx_class,
    signal_name: ArraySignalName,
) -> None:
    attrs = dict(dataset.attrs)
    attrs["NX_class"] = nx_class
    for k, v in attrs.items():
        group.attrs[k] = v
    for name, var in dataset.data_vars.items():
        _write_dataarray(var, group, str(name), signal_name)


def _write_datatree(
    datatree: xr.DataTree, group: h5py.Group, nx_class, signal_name: ArraySignalName
) -> None:
    if "NX_class" in datatree.attrs:
        nx_class = datatree.attrs["NX_class"]
    if nx_class not in ("NXentry", "NXsubentry") and datatree.has_data:
        msg = f"Expected tree node {group.name} with class {nx_class} to not have any data."
        raise InvalidEntryError(msg)
    _write_dataset(datatree, group, nx_class, signal_name)
    nx_class = "NXentry" if nx_class == "NXroot" else "NXsubentry"
    for name, child in datatree.children.items():
        sub_group = group.create_group(name)
        _write_datatree(child, sub_group, nx_class, signal_name)


@xr.register_dataarray_accessor("nexus")
class NexusDataArray:
    """
    Provides the DataArray.nexus object.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self.dataarray = xarray_obj

    def write_to(
        self,
        filename_or_obj: Path | h5py.File,
        data_path: str,
        signal_name: None | ArraySignalName = None,
    ) -> None:
        """
        Write the given DataArray to the NeXus file creating the given data_path.

        Creates an NXdata and writes each the signal and coords.
        This expects the parent path to be an NXentry or NXsubentry

        Args:
            filename_or_obj: The NeXus file to write to.
            data_path: The path of the entry to create.
        """
        with DeferredAction() as defer:
            if isinstance(filename_or_obj, h5py.File):
                nx_file = filename_or_obj
            else:
                nx_file = h5py.File(filename_or_obj, "a")
                defer.on_complete(nx_file.close)
            filename = nx_file.filename

            parts = data_path.removesuffix("/").split("/")
            name = parts[-1]
            base_path = "/".join(parts[:-1]) + "/"

            if "NX_class" not in nx_file[base_path].attrs:
                message = f"Expected {filename}:{base_path} to have NX_class."
                raise InvalidEntryError(message)
            if nx_file[base_path].attrs["NX_class"] not in ("NXentry", "NXsubentry"):
                message = (
                    f"Expected {filename}:{base_path} to be file NXentry or NXsubentry."
                )
                raise InvalidEntryError(message)

            if name != self.dataarray.name:
                logger.warning(
                    f"Writing array with name {self.dataarray.name} to NXdata with name {name}. The array name will be lost."
                )
            signal_name = signal_name if signal_name is not None else ArraySignalName()
            _write_dataarray(self.dataarray, nx_file[base_path], name, signal_name)


@xr.register_dataset_accessor("nexus")
class NexusDataset:
    """
    Provides the Dataset.nexus object.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self.dataset = xarray_obj

    def write_to(
        self,
        filename_or_obj: Path | h5py.File,
        entry_path: str = "/",
        signal_name: None | ArraySignalName = None,
    ) -> None:
        """
        Write the given Dataset to the NeXus file creating the entry_path.

        Creates an NXentry or NXsubentry and writes each array as a NXdata.
        If the parent path is the root a NXentry is created.
        If the parent path is a NXentry or NXsubentry a NXsubentry is created.
        If the parent path is anything else an exception is raised

        Args:
            filename_or_obj: The NeXus file to write to.
            entry_path: The path of the NXentry/NXsubentry to create.
        """
        with DeferredAction() as defer:
            if isinstance(filename_or_obj, h5py.File):
                nx_file = filename_or_obj
            else:
                nx_file = h5py.File(filename_or_obj, "a")
                defer.on_complete(nx_file.close)
            filename = nx_file.filename

            if entry_path in nx_file:
                message = f"Cannot create {filename}:{entry_path}: it already exists."
                raise EntryExistsError(message)

            base_path = "/".join(entry_path.removesuffix("/").split("/")[:-1]) + "/"

            if "NX_class" not in nx_file[base_path].attrs:
                if nx_file[base_path].name != "/":
                    message = f"Expected {filename}:{base_path} to be file root or have NX_class."
                    raise InvalidEntryError(message)
                nx_class = "NXroot"
            else:
                nx_class = nx_file[base_path].attrs["NX_class"]

            if nx_class not in ("NXroot", "NXentry", "NXsubentry"):
                message = f"Expected {filename}:{base_path} to be file NXroot, NXentry or NXsubentry."
                raise InvalidEntryError(message)

            nx_class = "NXentry" if nx_class == "NXroot" else "NXsubentry"
            group: h5py.Group = nx_file.create_group(entry_path)
            signal_name = signal_name if signal_name is not None else ArraySignalName()
            _write_dataset(self.dataset, group, nx_class, signal_name)

    def write_into(
        self,
        filename_or_obj: Path | h5py.File,
        entry_path: str = "/",
        signal_name: None | ArraySignalName = None,
    ) -> None:
        """
        Write the given Dataset to the NeXus file extending the existing entry at entry_path.

        Expects the path to point at a NXentry or NXsubentry.

        Args:
            filename_or_obj: The NeXus file to write to.
            entry_path: The path of the NXentry/NXsubentry to write to.
        """
        with DeferredAction() as defer:
            if isinstance(filename_or_obj, h5py.File):
                nx_file = filename_or_obj
            else:
                nx_file = h5py.File(filename_or_obj, "a")
                defer.on_complete(nx_file.close)
            filename = nx_file.filename

            if "NX_class" not in nx_file[entry_path].attrs:
                message = f"Expected {filename}:{entry_path} to have NX_class."
                raise InvalidEntryError(message)
            if nx_file[entry_path].attrs["NX_class"] not in ("NXentry", "NXsubentry"):
                message = f"Expected {filename}:{entry_path} to be file NXentry or NXsubentry."
                raise InvalidEntryError(message)

            nx_class = nx_file[entry_path].attrs["NX_class"]
            signal_name = signal_name if signal_name is not None else ArraySignalName()
            _write_dataset(self.dataset, nx_file[entry_path], nx_class, signal_name)


@xr.register_datatree_accessor("nexus")
class NexusDataTree:
    """
    Provides the DataTree.nexus object.
    """

    def __init__(self, xarray_obj: xr.DataTree):
        self.datatree = xarray_obj

    def write_to(
        self,
        filename_or_obj: Path | h5py.File,
        root_path: str = "/",
        mode="a",
        signal_name: None | ArraySignalName = None,
    ) -> None:
        """
        Write the given DataTree to the NeXus file creating the root at the given root_path.

        Creates an NXentry or NXsubentry and writes each array as a NXdata.
        If the parent path is the root a NXentry is created.
        If the parent path is a NXentry or NXsubentry a NXsubentry is created.
        If the parent path is anything else an exception is raised

        Args:
            filename_or_obj: The NeXus file to write to.
            root_path: The path to the root of the tree.
            mode: The mode to open the file with if filename_or_obj is a path.
        """
        if mode not in ("r+", "w", "w-", "x", "a"):
            raise ValueError("Expected mode to be one of r+, w, w- or x, or a")
        with DeferredAction() as defer:
            if isinstance(filename_or_obj, h5py.File):
                nx_file = filename_or_obj
            else:
                nx_file = h5py.File(filename_or_obj, mode)
                defer.on_complete(nx_file.close)
            filename = nx_file.filename

            if root_path in nx_file:
                message = f"Cannot create {filename}:{root_path}: it already exists."
                raise EntryExistsError(message)

            base_path = "/".join(root_path.removesuffix("/").split("/")[:-1]) + "/"

            if "NX_class" not in nx_file[base_path].attrs:
                if nx_file[base_path].name != "/":
                    message = f"Expected {filename}:{base_path} to be file root or have NX_class."
                    raise InvalidEntryError(message)
                nx_class = "NXroot"
            else:
                nx_class = nx_file[base_path].attrs["NX_class"]

            if nx_class not in ("NXroot", "NXentry", "NXsubentry"):
                message = f"Expected {filename}:{base_path} to be file NXroot, NXentry or NXsubentry."
                raise InvalidEntryError(message)

            nx_class = "NXentry" if nx_class == "NXroot" else "NXsubentry"
            group: h5py.Group = nx_file.create_group(root_path)
            signal_name = signal_name if signal_name is not None else ArraySignalName()

            _write_datatree(self.datatree, group, nx_class, signal_name)

    def write_into(
        self,
        filename_or_obj: Path | h5py.File,
        root_path: str = "/",
        mode="a",
        signal_name: None | ArraySignalName = None,
    ) -> None:
        """
        Write the given DataTree to the NeXus file extending the existing entry at root_path.

        Expects the path to point at a NXentry or NXsubentry if it the node contains data.
        If the node does not have data, NXroot is permitted.

        Args:
            filename_or_obj: The NeXus file to write to.
            root_path: The path to the root of the tree.
            mode: The mode to open the file with if filename_or_obj is a path.
        """
        if mode not in ("r+", "w", "w-", "x", "a"):
            raise ValueError("Expected mode to be one of r+, w, w- or x, or a")
        with DeferredAction() as defer:
            if isinstance(filename_or_obj, h5py.File):
                nx_file = filename_or_obj
            else:
                nx_file = h5py.File(filename_or_obj, mode)
                defer.on_complete(nx_file.close)
            filename = nx_file.filename

            if "NX_class" not in nx_file[root_path].attrs:
                if nx_file[root_path].name != "/":
                    message = f"Expected {filename}:{root_path} to be file root or have NX_class."
                    raise InvalidEntryError(message)
                nx_class = "NXroot"
            else:
                nx_class = nx_file[root_path].attrs["NX_class"]

            group = nx_file[root_path]
            signal_name = signal_name if signal_name is not None else ArraySignalName()
            _write_datatree(self.datatree, group, nx_class, signal_name)


def insert_into_axes(axes: list[str], axis: list[str]) -> list[str]:
    """
    This inserts the values from axis into axes, trying to respect the order.
    New elements are inserted as late as possible before any existing elements.

    Args:
        axes: The original list of axes to update. (Note that this list is not changes)
        axis: The axis to insert into the list.

    Returns:
        A copy of axes with the axis inserted.

    Examples:
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

    def open_dataarray(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        data_path: None | str = None,
    ) -> xr.DataArray:
        """
        Opens the specified NXdata and returns it as the sole data array on a dataset.

        Args:
            filename_or_obj: The path to the file to read, or the file itself.
            drop_variables: Unused
            entry_path: The path within the file to read.
                        If None, the default entry is used.

        Returns:
            Returns a xarray.Dataset containing a data array.
        """
        # TODO (dmd): implement this
        # https://github.com/pydata/xarray/issues/10562
        raise NotImplementedError()
        try:
            should_close = self._open(filename_or_obj)
            assert self.nx_file is not None

            if data_path is None:
                path = "/"
                while "signal" not in self.nx_file[path].attrs:
                    if "default" not in self.nx_file[path].attrs:
                        raise InvalidEntryError("Could not find default signal.")
                    path = f"{path.removesuffix('/')}/{self.nx_file[path].attrs['default']}"
                if "signal" not in self.nx_file[path].attrs:
                    raise InvalidEntryError("Could not find default signal.")
                data_path = path

            darray = self._read_nxdata(data_path)

            if should_close:
                darray.set_close(self._close)

        except:
            self._close()
            raise
        else:
            return darray

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        entry_path: None | str = None,
    ) -> xr.Dataset:
        """
        Opens the specified NXdata and returns it as the sole data array on a dataset.

        Args:
            filename_or_obj: The path to the file to read, or the file itself.
            drop_variables: Unused
            entry_path: The path within the file to read.
                        If None, the default entry is used.

        Returns:
            Returns a xarray.Dataset containing a data array.
        """
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

            if (
                "NX_class" in self.nx_file[entry_path].attrs
                and self.nx_file[entry_path].attrs["NX_class"] == "NXdata"
            ):
                darray = self._read_nxdata(entry_path)
                ds = xr.Dataset({darray.name: darray})
            else:
                ds = self._read_all_data_on_nxentry(entry_path)

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
        """
        Opens the specified NXentry, NXsubentry or root and returns a DataTree of the object.

        This recursively loads as follows:
        - all NXdata groups are loaded into xarray.DataArrays on each node.
        - all NXentry of NXsubentry are loaded as nodes in the tree.
        - all other NX classes are have their arrtibutes loaded into a dictionary in the node at attrs[group_name].

        Args:
            filename_or_obj: The path to the file to read, or the file itself.
            drop_variables: Unused
            root: The path of the root of the tree within the file to read.

        Returns:
            Returns a xarray.DataTree representing the NeXus data rooted at root.
        """
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
        path_attrs["signal"]

        signal = self.nx_file[data_path].attrs["signal"]
        axes: list[str] = self.nx_file[data_path].attrs["axes"]

        inner_coords: dict[str, tuple[tuple[str, ...], Any, dict[str, Any]]] = {}
        for name in self.nx_file[data_path]:
            index_path = f"{name}_indices"
            if index_path in self.nx_file[data_path].attrs:
                del path_attrs[index_path]

                inx = self.nx_file[data_path].attrs[index_path]
                if isinstance(inx, np.integer):
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

                inner_coords[name] = (
                    coord_dims,
                    values,
                    axis_attrs,
                )
        signal_path = f"{data_path}/{signal}"
        name = data_path.removesuffix("/").split("/")[-1]
        chunks = self.nx_file[signal_path].chunks or "auto"
        return xr.DataArray(
            da.from_array(self.nx_file[signal_path], chunks=chunks),
            name=name,
            dims=axes,
            coords=inner_coords,
            attrs=path_attrs,
        )

    def _read_all_data_on_nxentry(self, entry_path: str) -> xr.Dataset:
        assert self.nx_file is not None

        root_attrs = dict(self.nx_file[entry_path].attrs)
        if "NX_class" not in self.nx_file[entry_path].attrs:
            if self.nx_file[entry_path].name != "/":
                message = f"Expected {self.filename}:{entry_path} to be file root or have NX_class."
                raise InvalidEntryError(message)
        elif self.nx_file[entry_path].attrs["NX_class"] not in (
            "NXroot",
            "NXentry",
            "NXsubentry",
        ):
            message = f"Expected {self.filename}:{entry_path} to be NXroot, NXentry or NXsubentry."
            raise InvalidEntryError(message)
        else:
            del root_attrs["NX_class"]

        entries: dict[str, xr.DataArray] = {}

        base_path = entry_path.removesuffix("/")

        for path in self.nx_file[entry_path]:
            sub_path = f"{base_path}/{path}"
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
            self.filename = filename_or_obj
        return should_close

    def _close(self) -> None:
        if self.nx_file is not None:
            self.nx_file.close()

    open_dataarray_parameters = ["filename_or_obj", "drop_variables", "data_path"]
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
