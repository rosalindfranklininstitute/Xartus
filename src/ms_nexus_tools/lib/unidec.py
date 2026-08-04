# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from typing import NamedTuple
from pathlib import Path

import h5py
import numpy as np


class Spectra(NamedTuple):
    name: str
    mass_values: np.ndarray[tuple[int]]
    intensity_values: np.ndarray[tuple[int]]


def write_unidec(
    basename: Path, total_spectra: Spectra, individual_spectra: list[Spectra]
) -> None:
    total_spectra_data = np.array(
        [total_spectra.mass_values, total_spectra.intensity_values]
    ).T
    np.savetxt(basename.with_suffix(".total_spectrum.txt"), total_spectra_data)

    line_count = len(individual_spectra)
    if line_count == 0:
        return
    with h5py.File(basename.with_suffix(".unidec.hdf5"), "w") as fle:
        dataset = fle.create_group("ms_dataset")
        dataset.attrs["num"] = line_count
        dataset.attrs["v1name"] = "Variable 1"
        dataset.attrs["v2name"] = "Variable 2"
        for ll, line in enumerate(individual_spectra):
            line_dataset = fle.create_group(f"ms_dataset/{ll}")
            line_dataset.attrs["name"] = line.name
            raw_line = line.intensity_values
            line_stats = np.percentile(raw_line, [0, 100])
            normal_line = (raw_line - line_stats[0]) / (line_stats[1] - line_stats[0])
            line_dataset.create_dataset(
                name="raw_data",
                data=np.array([line.mass_values, raw_line[:]]).T,
                chunks=(len(line.mass_values), 2),
            )
            line_dataset.create_dataset(
                name="processed_data",
                data=np.array([line.mass_values, normal_line[:]]).T,
                chunks=(len(line.mass_values), 2),
            )
