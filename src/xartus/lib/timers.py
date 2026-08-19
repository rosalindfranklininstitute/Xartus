# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

from typing import Any
from contextlib import AbstractContextManager
import time
import json
from collections.abc import Iterable
from pathlib import Path


class JSONTimerSkip(Exception):  # noqa: N818
    pass


class JSONTimer(AbstractContextManager):
    def __init__(self, filename: Path, keys: Iterable[str]):
        self.filename = filename
        self.keys = keys

        self._start: float = -1
        self._data: dict[str, Any] = {}

    def __enter__(self):

        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is JSONTimerSkip:
            return True

        end = time.monotonic()
        duration = end - self._start
        old_data = {}
        if self.filename.exists():
            with open(self.filename, "r") as fd:
                old_data = json.load(fd)
        new_data = old_data
        for key in self.keys:
            if key not in new_data:
                new_data[key] = {}
            new_data = new_data[key]
        new_data["duration"] = duration
        new_data.update(self._data)
        with open(self.filename, "w") as fd:
            json.dump(old_data, fd, indent=2)

        return False

    def skip_if_present(self) -> None:
        if self.filename.exists():
            with open(self.filename, "r") as fd:
                old_data = json.load(fd)
        skip = True
        new_data = old_data
        for key in self.keys:
            if key not in new_data:
                skip = False
                break
            new_data = new_data[key]
        skip = skip and "duration" in new_data
        if skip:
            raise JSONTimerSkip()

    def add_user_data(self, /, **kwargs) -> None:
        self._data.update(kwargs)
