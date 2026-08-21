# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from pathlib import Path
import re

import pytest

from xartus.lib.utils import FileAction, FileGuard, count_digits


def test_count_digits():
    assert count_digits(0) == 1

    assert count_digits(1) == 1
    assert count_digits(11) == 2
    assert count_digits(111) == 3
    assert count_digits(1111) == 4
    assert count_digits(11111) == 5

    assert count_digits(-1) == 1
    assert count_digits(-11) == 2
    assert count_digits(-111) == 3
    assert count_digits(-1111) == 4
    assert count_digits(-11111) == 5


def test_file_guard_delete():

    filename = Path(__file__).parent / "test.txt"
    filename.unlink(missing_ok=True)

    with FileGuard(filename, on_failure=FileAction.DELETE):
        filename.touch()

    assert filename.exists()
    filename.unlink()

    with (
        pytest.raises(RuntimeError, match="Error!"),
        FileGuard(filename, on_failure=FileAction.DELETE),
    ):
        filename.touch()
        assert filename.exists()
        raise RuntimeError("Error!")

    assert not filename.exists()

    filename2 = Path(__file__).parent / "test2.txt"
    filename2.unlink(missing_ok=True)

    with FileGuard(
        filename,
        filename2,
        on_failure=FileAction.DELETE,
    ):
        filename.touch()
        filename2.touch()

    assert filename.exists()
    assert filename2.exists()
    filename.unlink()

    with (
        pytest.raises(RuntimeError, match="Error!"),
        FileGuard(filename, filename2, on_failure=FileAction.DELETE),
    ):
        filename.touch()
        filename2.touch()
        assert filename.exists()
        assert filename2.exists()
        raise RuntimeError("Error!")

    assert not filename.exists()
    assert not filename2.exists()


def filename_regex(*paths):
    return f".*{re.escape(', '.join([str(p) for p in paths]))}.*"


def test_file_guard_check():

    filename = Path(__file__).parent / "test.txt"
    filename.unlink(missing_ok=True)

    with FileGuard(filename, on_success=FileAction.CHECK_EXISTS):
        filename.touch()

    assert filename.exists()
    filename.unlink()

    with (
        pytest.raises(FileNotFoundError, match=filename_regex(filename)),
        FileGuard(filename, on_success=FileAction.CHECK_EXISTS),
    ):
        pass

    assert not filename.exists()

    filename2 = Path(__file__).parent / "test2.txt"
    filename2.unlink(missing_ok=True)

    with FileGuard(filename, filename2, on_success=FileAction.CHECK_EXISTS):
        filename.touch()
        filename2.touch()

    assert filename.exists()
    assert filename2.exists()
    filename.unlink()
    filename2.unlink()

    with (
        pytest.raises(FileNotFoundError, match=filename_regex(filename, filename2)),
        FileGuard(filename, filename2, on_success=FileAction.CHECK_EXISTS),
    ):
        pass

    assert not filename.exists()
    assert not filename2.exists()

    with (
        pytest.raises(FileNotFoundError, match=filename_regex(filename2)),
        FileGuard(filename, filename2, on_success=FileAction.CHECK_EXISTS),
    ):
        filename.touch()

    assert filename.exists()
    assert not filename2.exists()
    filename.unlink()
