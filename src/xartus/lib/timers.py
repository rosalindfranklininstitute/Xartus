# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause

from typing import Any, override
from enum import Enum
from contextlib import AbstractContextManager
import time
import json
from collections.abc import Iterable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExceptionAction(Enum):
    RERAISE = "reraise"
    SUPPRESS_AND_IGNORE = "suppress_and_ignore"
    SUPPRESS_AND_STORE = "suppress_and_store"


class JSONTimerSkip(Exception):  # noqa: N818
    pass


class JSONTimer(AbstractContextManager):
    def __init__(
        self,
        filename: Path,
        keys: Iterable[str],
        print_borders=False,
        suppress_exceptions=ExceptionAction.RERAISE,
    ):
        self.filename = filename
        self.keys = keys

        self._start: float = -1
        self._data: dict[str, Any] = {}
        self.print_borders = print_borders
        self.suppress_exceptions = suppress_exceptions

    @override
    def __enter__(self):
        if self.print_borders:
            logger.info(f"Beginning: {'.'.join(self.keys)}")

        self._start = time.monotonic()
        return self

    @override
    def __exit__(self, exc_type, exc_value, traceback) -> bool:

        if exc_type is JSONTimerSkip:
            if self.print_borders:
                logger.info(f"Skipped: {'.'.join(self.keys)}")
            return True
        if exc_type is not None:
            if self.print_borders:
                logger.info(f"Error: {'.'.join(self.keys)}")
                logger.error(f"{exc_type}: {exc_value}")
            match self.suppress_exceptions:
                case ExceptionAction.RERAISE:
                    return False
                case ExceptionAction.SUPPRESS_AND_IGNORE:
                    return True
                case _:
                    pass

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

        if exc_type is not None:
            assert self.suppress_exceptions is ExceptionAction.SUPPRESS_AND_STORE
            new_data["duration"] = f"{exc_type}: {exc_value}"
        else:
            new_data["duration"] = duration

        new_data.update(self._data)
        with open(self.filename, "w") as fd:
            json.dump(old_data, fd, indent=2)

        if exc_type is not None:
            assert self.suppress_exceptions is ExceptionAction.SUPPRESS_AND_STORE
            return True

        logger.info(f"Completed: {'.'.join(self.keys)}: {duration}s")
        return False

    def skip_if_present(self) -> None:
        if self.filename.exists():
            with open(self.filename, "r") as fd:
                old_data = json.load(fd)
        else:
            return
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

    def skip() -> None:
        raise JSONTimerSkip()

    def add_user_data(self, /, **kwargs) -> None:
        self._data.update(kwargs)
