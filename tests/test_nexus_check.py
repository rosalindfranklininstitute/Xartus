# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from pathlib import Path

import h5py
import numpy as np

from xartus.lib import check_nexus, InvalidEntryError

import pytest


@pytest.fixture
def nx_file():
    filename = Path(__file__).parent / "test.nxs"
    if filename.exists():
        filename.unlink()
    with h5py.File(filename, "w") as fle:
        yield fle
    filename.unlink()


def test_root(nx_file):

    with pytest.raises(InvalidEntryError, match="/ should be NXroot"):
        check_nexus(nx_file)

    nx_file["/"].attrs["NX_class"] = "NXobject"

    with pytest.raises(InvalidEntryError, match="/ should be NXroot"):
        check_nexus(nx_file)

    nx_file["/"].attrs["NX_class"] = "NXroot"

    with pytest.raises(InvalidEntryError, match="should have a 'default' attribute"):
        check_nexus(nx_file)
    nx_file["/"].attrs["default"] = "entry"

    with pytest.raises(InvalidEntryError, match="default object should exist."):
        check_nexus(nx_file)

    entry = nx_file["/"].create_group("entry")
    entry.attrs["NX_class"] = "NXroot"

    with pytest.raises(InvalidEntryError, match="NXroot should be '/'"):
        check_nexus(nx_file)

    entry.attrs["NX_class"] = "NXobject"

    check_nexus(nx_file)


def test_entry(nx_file):

    nx_file["/"].attrs["NX_class"] = "NXroot"
    nx_file["/"].attrs["default"] = "group"

    group = nx_file["/"].create_group("group")

    with pytest.raises(InvalidEntryError, match="should have a NX_class"):
        check_nexus(nx_file)

    group.attrs["NX_class"] = "NXobject"

    entry = group.create_group("entry")
    entry.attrs["NX_class"] = "NXentry"

    with pytest.raises(InvalidEntryError, match="should be a child of NXroot"):
        check_nexus(nx_file)

    group.attrs["NX_class"] = "NXentry"
    entry.attrs["NX_class"] = "NXobject"

    with pytest.raises(InvalidEntryError, match="should have a 'default' attribute"):
        check_nexus(nx_file)
    group.attrs["default"] = "DNE"

    with pytest.raises(InvalidEntryError, match="default object should exist."):
        check_nexus(nx_file)
    group.attrs["default"] = "entry"

    check_nexus(nx_file)


def test_subentry(nx_file):
    nx_file["/"].attrs["NX_class"] = "NXroot"
    nx_file["/"].attrs["default"] = "first"
    first = nx_file["/"].create_group("first")
    second = first.create_group("second")

    first.attrs["NX_class"] = "NXobject"
    second.attrs["NX_class"] = "NXobject"

    check_nexus(nx_file)

    first.attrs["NX_class"] = "NXsubentry"

    with pytest.raises(InvalidEntryError, match="should be a child of either"):
        check_nexus(nx_file)
    first.attrs["NX_class"] = "NXentry"
    first.attrs["default"] = "second"

    second.attrs["NX_class"] = "NXsubentry"

    with pytest.raises(InvalidEntryError, match="should have a 'default' attribute"):
        check_nexus(nx_file)

    second.attrs["default"] = "third"

    with pytest.raises(InvalidEntryError, match="default object should exist."):
        check_nexus(nx_file)

    third = second.create_group("third")
    third.attrs["NX_class"] = "NXobject"

    check_nexus(nx_file)


def test_data(nx_file):
    nx_file["/"].attrs["NX_class"] = "NXroot"
    nx_file["/"].attrs["default"] = "first"
    first = nx_file["/"].create_group("first")
    second = first.create_group("second")

    first.attrs["NX_class"] = "NXobject"
    second.attrs["NX_class"] = "NXobject"

    check_nexus(nx_file)

    second.attrs["NX_class"] = "NXdata"

    with pytest.raises(InvalidEntryError, match="should be a child of either"):
        check_nexus(nx_file)

    first.attrs["NX_class"] = "NXentry"
    first.attrs["default"] = "second"

    with pytest.raises(InvalidEntryError, match="should have a 'signal' attribute"):
        check_nexus(nx_file)
    second.attrs["signal"] = "DNE"

    with pytest.raises(
        InvalidEntryError, match="does not have a NXfield for the specified signal."
    ):
        check_nexus(nx_file)
    second.attrs["signal"] = "third"
    second.create_group("third")
    second["third"].attrs["NX_class"] = "NXobject"

    with pytest.raises(InvalidEntryError, match="NXdata signal should be a dataset."):
        check_nexus(nx_file)
    del second["third"]
    third = second.create_dataset("third", dtype=np.int32, shape=(2, 3))
    third.attrs["NX_class"] = "NXfield"

    with pytest.raises(InvalidEntryError, match=" have an 'axes' attribute"):
        check_nexus(nx_file)
    second.attrs["axes"] = "string"

    with pytest.raises(InvalidEntryError, match="axes should be a list"):
        check_nexus(nx_file)
    second.attrs["axes"] = []

    with pytest.raises(
        InvalidEntryError, match="axes should have a value for every dimension"
    ):
        check_nexus(nx_file)
    second.attrs["axes"] = ["a", "b"]

    second.attrs["auxiliary_signals"] = "aux"
    with pytest.raises(InvalidEntryError, match="auxiliary_signals should be a list"):
        check_nexus(nx_file)

    second.attrs["auxiliary_signals"] = ["aux"]
    with pytest.raises(
        InvalidEntryError,
        match="does not have a NXfield for the auxiliary signal 'aux'.",
    ):
        check_nexus(nx_file)

    aux = second.create_dataset("aux", dtype=np.int32, shape=(2, 3))
    aux.attrs["NX_class"] = "NXfield"

    check_nexus(nx_file)


def test_data_axes(nx_file):

    nx_file["/"].attrs["NX_class"] = "NXroot"
    nx_file["/"].attrs["default"] = "first"
    first = nx_file["/"].create_group("first")
    first.attrs["NX_class"] = "NXentry"
    first.attrs["default"] = "second"

    second = first.create_group("second")
    second.attrs["NX_class"] = "NXdata"
    second.attrs["signal"] = "second_sig"
    second.attrs["axes"] = ["a", "b"]

    second_sig = second.create_dataset("second_sig", dtype=np.int32, shape=(2, 3))
    second_sig.attrs["NX_class"] = "NXfield"

    check_nexus(nx_file)

    second.attrs["a_indices"] = "string"

    with pytest.raises(
        InvalidEntryError, match="_indices should be an integer or a list"
    ):
        check_nexus(nx_file)
    second.attrs["a_indices"] = 12

    with pytest.raises(InvalidEntryError, match="but an axis field is not present"):
        check_nexus(nx_file)
    axis_a = second.create_dataset("a", dtype=np.int32, shape=(2))
    axis_a.attrs["NX_class"] = "NXfield"

    with pytest.raises(InvalidEntryError, match="has an index outside the dimensions"):
        check_nexus(nx_file)
    second.attrs["a_indices"] = 1

    with pytest.raises(
        InvalidEntryError,
        match="has a different shape to the relevant dimensions of the signal.",
    ):
        check_nexus(nx_file)
    second.attrs["a_indices"] = 0

    check_nexus(nx_file)


def test_data_multi_dimension_axes(nx_file):

    nx_file["/"].attrs["NX_class"] = "NXroot"
    nx_file["/"].attrs["default"] = "first"
    first = nx_file["/"].create_group("first")
    first.attrs["NX_class"] = "NXentry"
    first.attrs["default"] = "second"

    second = first.create_group("second")
    second.attrs["NX_class"] = "NXdata"
    second.attrs["signal"] = "second_sig"
    second.attrs["axes"] = ["a", "b", "c"]

    second_sig = second.create_dataset("second_sig", dtype=np.int32, shape=(2, 3, 4))
    second_sig.attrs["NX_class"] = "NXfield"
    check_nexus(nx_file)

    axis_a = second.create_dataset("a", dtype=np.int32, shape=(2, 3))
    axis_a.attrs["NX_class"] = "NXfield"

    second.attrs["a_indices"] = [1, 2, 3]

    with pytest.raises(
        InvalidEntryError,
        match="and its indices should have the same number of dimensions",
    ):
        check_nexus(nx_file)
    second.attrs["a_indices"] = [2, 3]

    with pytest.raises(InvalidEntryError, match="has an index outside the dimensions"):
        check_nexus(nx_file)
    second.attrs["a_indices"] = [2, 0]

    with pytest.raises(
        InvalidEntryError,
        match="should be in order",
    ):
        check_nexus(nx_file)
    second.attrs["a_indices"] = [0, 2]

    with pytest.raises(
        InvalidEntryError,
        match="has a different shape to the relevant dimensions of the signal.",
    ):
        check_nexus(nx_file)
    second.attrs["a_indices"] = [0, 1]

    check_nexus(nx_file)


def test_field(nx_file):

    nx_file["/"].attrs["NX_class"] = "NXroot"
    nx_file["/"].attrs["default"] = "first"
    first = nx_file["/"].create_group("first")
    first.attrs["NX_class"] = "NXentry"
    first.attrs["default"] = "second"

    second = first.create_group("second")
    second.attrs["NX_class"] = "NXsubentry"
    second.attrs["default"] = "field"

    second.attrs["signal"] = "field"
    second.attrs["axes"] = ["a", "b", "c"]

    second.create_group("field")
    second["field"].attrs["NX_class"] = "NXfield"

    with pytest.raises(InvalidEntryError, match="NXfield should be a Dataset"):
        check_nexus(nx_file)

    del second["field"]
    second.create_dataset("field", dtype=np.int32, shape=(2, 3, 4))
    second["field"].attrs["NX_class"] = "NXfield"

    with pytest.raises(InvalidEntryError, match="NXfield should be a child of NXdata"):
        check_nexus(nx_file)
    second.attrs["NX_class"] = "NXdata"
    del second.attrs["default"]
    second.attrs["signal"] = "field"
    second.attrs["axes"] = ["a", "b", "c"]

    second.create_dataset("other", dtype=np.int32, shape=(2, 3, 4))
    second["other"].attrs["NX_class"] = "NXfield"

    with pytest.raises(InvalidEntryError, match="NXfield should be a"):
        check_nexus(nx_file)

    second.attrs["axes"] = ["other", "b", "c"]
    second.attrs["other_indices"] = [0, 1, 2]
    check_nexus(nx_file)


def test_others(nx_file):

    nx_file["/"].attrs["NX_class"] = "NXroot"
    nx_file["/"].attrs["default"] = "first"
    entry = nx_file["/"].create_group("first")
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["default"] = "second"

    data = entry.create_group("second")
    data.attrs["NX_class"] = "NXdata"
    data.attrs["signal"] = "second_sig"
    data.attrs["axes"] = ["a", "b", "c"]

    second_sig = data.create_dataset("second_sig", dtype=np.int32, shape=(2, 3, 4))
    second_sig.attrs["NX_class"] = "NXfield"
    check_nexus(nx_file)

    entry.create_dataset("other", dtype=np.int32, shape=(2, 3, 4))
    entry["other"].attrs["NX_class"] = "NXobject"
    with pytest.raises(InvalidEntryError, match="Dataset should be NXfield"):
        check_nexus(nx_file)

    del entry["other"]

    other = entry.create_group("other")
    other.attrs["NX_class"] = "NXobject"

    other.create_dataset("data", dtype=np.int32, shape=(2, 3, 4))
    other["data"].attrs["NX_class"] = "NXfield"
    with pytest.raises(InvalidEntryError, match="Group should only contain groups"):
        check_nexus(nx_file)

    del other["data"]

    other.create_group("data")
    other["data"].attrs["NX_class"] = "NXobject"

    check_nexus(nx_file)
