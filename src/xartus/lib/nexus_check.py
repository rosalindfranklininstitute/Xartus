# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
import h5py
import numpy as np

from .exceptions import InvalidEntryError


def _check_root(group: h5py.Group) -> None:
    if group.name != "/":
        raise InvalidEntryError("NXroot should be '/'.")
    if "default" not in group.attrs:
        raise InvalidEntryError("NXroot should have a 'default' attribute.")
    default = group.attrs["default"]
    if default not in group:
        raise InvalidEntryError("The default object should exist.")


def _check_entry(group: h5py.Group) -> None:
    if group.parent.attrs["NX_class"] != "NXroot":
        raise InvalidEntryError("NXentry should be a child of NXroot.")
    if "default" not in group.attrs:
        raise InvalidEntryError("NXentry should have a 'default' attribute.")
    default = group.attrs["default"]
    if default not in group:
        raise InvalidEntryError("The default object should exist.")


def _check_subentry(group: h5py.Group) -> None:
    if group.parent.attrs["NX_class"] not in ("NXentry", "NXsubentry"):
        raise InvalidEntryError(
            "NXsubenry should be a child of either NXentry or NXsubentry."
        )
    if "default" not in group.attrs:
        raise InvalidEntryError("NXsubentry should have a 'default' attribute.")
    default = group.attrs["default"]
    if default not in group:
        raise InvalidEntryError("The default object should exist.")


def _check_data(group: h5py.Group) -> None:
    if group.parent.attrs["NX_class"] not in ("NXentry", "NXsubentry"):
        raise InvalidEntryError(
            "NXdata should be a child of either NXentry or NXsubentry."
        )
    if "signal" not in group.attrs:
        raise InvalidEntryError("NXdata should have a 'signal' attribute.")
    signal = group.attrs["signal"]
    if signal not in group:
        raise InvalidEntryError(
            "NXdata does not have a NXfield for the specified signal."
        )

    if "auxiliary_signals" in group.attrs:
        if not isinstance(group.attrs["auxiliary_signals"], np.ndarray):
            raise InvalidEntryError("NXdata auxiliary_signals should be a list.")

        for aux_signal in group.attrs["auxiliary_signals"]:
            if aux_signal not in group:
                msg = f"NXdata does not have a NXfield for the auxiliary signal '{aux_signal}'."
                raise InvalidEntryError(msg)

    if not isinstance(group[signal], h5py.Dataset):
        raise InvalidEntryError("NXdata signal should be a dataset.")

    if "axes" not in group.attrs:
        raise InvalidEntryError("NXdata should have an 'axes' attribute.")

    axes = group.attrs["axes"]
    if not isinstance(axes, np.ndarray):
        raise InvalidEntryError("NXdata axes should be a list.")

    signal_shape = group[signal].shape
    if len(signal_shape) != len(axes):
        raise InvalidEntryError(
            "NXdata axes should have a value for every dimension of the signal."
        )

    for name, value in group.attrs.items():
        if name.endswith("_indices"):
            indices = value
            if not isinstance(indices, (np.integer, np.ndarray)):
                msg = f"NXdata {name} should be an integer or a list."
                raise InvalidEntryError(msg)
            axis_name = name.removesuffix("_indices")
            if axis_name not in group:
                msg = f"NXdata has indices for axis {axis_name}, but an axis field is not present."
                raise InvalidEntryError(msg)
            if isinstance(indices, np.integer):
                indices = [indices]
            if any(s != i for s, i in zip(sorted(indices), indices, strict=True)):
                msg = f"NXdata {name} should be in order."
                raise InvalidEntryError(msg)

            axis_shape = group[axis_name].shape
            if len(indices) != len(axis_shape):
                msg = f"NXdata axis {axis_name} and its indices should have the same number of dimensions."
                raise InvalidEntryError(msg)

            if any(i >= len(axes) or i < 0 for i in indices):
                msg = f"NXdata {name} has an index outside the dimensions."
                raise InvalidEntryError(msg)

            expected_axis_shape = tuple(signal_shape[i] for i in indices)
            if expected_axis_shape != axis_shape:
                msg = f"NXdata axis {axis_name} has a different shape to the relevant dimensions of the signal."
                raise InvalidEntryError(msg)


def _check_field(group: h5py.Group) -> None:
    if not isinstance(group, h5py.Dataset):
        raise InvalidEntryError("NXfield should be a Dataset.")
    if group.parent.attrs["NX_class"] != "NXdata":
        raise InvalidEntryError("NXfield should be a child of NXdata.")
    signal = group.parent.attrs["signal"]
    aux_signal = group.parent.attrs.get("auxiliary_signals", [])
    name = group.name.split("/")[-1]

    if (
        name != signal
        and name not in aux_signal
        and not name.endswith(("_errors", "_scaling_factor", "_offset"))
        and f"{name}_indices" not in group.parent.attrs
    ):
        raise InvalidEntryError(
            "NXfield should be a signal, auxiliary signal, error, scaling factor, offset or axis."
        )


def _check_other(group: h5py.Group) -> None:
    # TODO (dmd): make invalid to have random objects on NXroot and NXdata
    # https://github.com/rosalindfranklininstitute/Xartus/issues/34
    for value in group.values():
        if not isinstance(value, h5py.Group):
            raise InvalidEntryError("Group should only contain groups.")


def _check_group(group: h5py.Group) -> None:
    try:
        if "NX_class" not in group.attrs:
            raise InvalidEntryError("Group should have a NX_class")  # noqa: TRY301

        if isinstance(group, h5py.Dataset) and group.attrs["NX_class"] != "NXfield":
            raise InvalidEntryError("Dataset should be NXfield")  # noqa: TRY301

        match group.attrs["NX_class"]:
            case "NXroot":
                _check_root(group)
            case "NXentry":
                _check_entry(group)
            case "NXsubentry":
                _check_subentry(group)
            case "NXdata":
                _check_data(group)
            case "NXfield":
                _check_field(group)
            case _:
                _check_other(group)
    except InvalidEntryError as e:
        msg = f"{group.name}: {str(e)}"
        raise InvalidEntryError(msg) from e

    if isinstance(group, h5py.Group):
        for name in list(group):
            subgroup = group[name]
            _check_group(subgroup)


def check_nexus(fle: h5py.File) -> None:
    """
    Takes in a NeXus file asserts that it is valid.
    This does not exactly follow the specification:
    It is strict in requiring 'default', 'signal' and 'axes' attributes.
    It is lax in allowing generic NXobjects nearly everywhere.

    This method stops at the first detected violation and raises InvalidEntryError with a description.
    """
    # TODO (dmd): Make this return a list of errors, rather than stopping at the first.
    # https://github.com/rosalindfranklininstitute/Xartus/issues/35

    if "NX_class" not in fle["/"].attrs or fle["/"].attrs["NX_class"] != "NXroot":
        raise InvalidEntryError("/ should be NXroot")
    _check_group(fle["/"])
