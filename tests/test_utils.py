# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import re

import pytest

from ms_nexus_tools import lib as mnxlib


def test_count_digits():
    assert mnxlib.utils.count_digits(0) == 1

    assert mnxlib.utils.count_digits(1) == 1
    assert mnxlib.utils.count_digits(11) == 2
    assert mnxlib.utils.count_digits(111) == 3
    assert mnxlib.utils.count_digits(1111) == 4
    assert mnxlib.utils.count_digits(11111) == 5

    assert mnxlib.utils.count_digits(-1) == 1
    assert mnxlib.utils.count_digits(-11) == 2
    assert mnxlib.utils.count_digits(-111) == 3
    assert mnxlib.utils.count_digits(-1111) == 4
    assert mnxlib.utils.count_digits(-11111) == 5


def test_file_guard_delete():

    filename = Path(__file__).parent / "test.txt"
    filename.unlink(missing_ok=True)

    with mnxlib.utils.FileGuard(
        filename, delete_on_failure=True, check_exist_on_success=False
    ):
        filename.touch()

    assert filename.exists()
    filename.unlink()

    with (
        pytest.raises(RuntimeError, match="Error!"),
        mnxlib.utils.FileGuard(
            filename, delete_on_failure=True, check_exist_on_success=False
        ),
    ):
        filename.touch()
        assert filename.exists()
        raise RuntimeError("Error!")

    assert not filename.exists()

    filename2 = Path(__file__).parent / "test2.txt"
    filename2.unlink(missing_ok=True)

    with mnxlib.utils.FileGuard(
        filename, filename2, delete_on_failure=True, check_exist_on_success=False
    ):
        filename.touch()
        filename2.touch()

    assert filename.exists()
    assert filename2.exists()
    filename.unlink()

    with (
        pytest.raises(RuntimeError, match="Error!"),
        mnxlib.utils.FileGuard(
            filename, filename2, delete_on_failure=True, check_exist_on_success=False
        ),
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

    with mnxlib.utils.FileGuard(
        filename, delete_on_failure=False, check_exist_on_success=True
    ):
        filename.touch()

    assert filename.exists()
    filename.unlink()

    with (
        pytest.raises(FileNotFoundError, match=filename_regex(filename)),
        mnxlib.utils.FileGuard(
            filename, delete_on_failure=False, check_exist_on_success=True
        ),
    ):
        pass

    assert not filename.exists()

    filename2 = Path(__file__).parent / "test2.txt"
    filename2.unlink(missing_ok=True)

    with mnxlib.utils.FileGuard(
        filename, filename2, delete_on_failure=False, check_exist_on_success=True
    ):
        filename.touch()
        filename2.touch()

    assert filename.exists()
    assert filename2.exists()
    filename.unlink()
    filename2.unlink()

    with (
        pytest.raises(FileNotFoundError, match=filename_regex(filename, filename2)),
        mnxlib.utils.FileGuard(
            filename, filename2, delete_on_failure=False, check_exist_on_success=True
        ),
    ):
        pass

    assert not filename.exists()
    assert not filename2.exists()

    with (
        pytest.raises(FileNotFoundError, match=filename_regex(filename2)),
        mnxlib.utils.FileGuard(
            filename, filename2, delete_on_failure=False, check_exist_on_success=True
        ),
    ):
        filename.touch()

    assert filename.exists()
    assert not filename2.exists()
    filename.unlink()
