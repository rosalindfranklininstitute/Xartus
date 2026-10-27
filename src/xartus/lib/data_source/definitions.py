# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
import numpy.typing as npt
from typing import NamedTuple
from dataclasses import dataclass

from ..bounds import Shape


class DataShape(NamedTuple):
    shape: Shape
    worst_case_density: float = 1.0


@dataclass
class Signal:
    name: str
    dtype: npt.DTypeLike
    units: str | None = None


@dataclass
class Axis:
    name: str
    primary_axis: int
    dtype: npt.DTypeLike
    units: str | None = None


class UnknownAxisError(Exception):
    def __init__(self, name: str):
        super().__init__(f"Unknown axis: {name}")
