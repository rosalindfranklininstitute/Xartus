# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: LicenseRef-RFI-Apache-2.0-Commons-clause
from pathlib import Path

import numpy as np

from xartus.api import data_convert
from xartus.lib.data_source import Axis, AxisType
from xartus.lib.utils import DeferredAction

from nexus_pixel_man_test_data import Man2DDataSource, ManData

import pytest


@pytest.fixture(scope="module")
def man_data_and_nexus():
    man_data = ManData()
    man_data_source = Man2DDataSource(
        man_data,
        supplementary_axes=[
            Axis("time", 0, AxisType.EXACT, np.float32, "s"),
            Axis("error", 2, AxisType.EXACT, np.float32, ""),
        ],
        multipliers=dict(x=0.1, y=0.1, mz=0.1, time=1.0, error=1.0),
    )
    filename = Path(__file__).parent / "man.nxs"
    if filename.exists():
        filename.unlink()

    process_args = data_convert.ProcessArgs(
        out_path=filename,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})
    yield man_data, filename
    filename.unlink()


@pytest.fixture
def defer():
    with DeferredAction() as d:
        yield d
