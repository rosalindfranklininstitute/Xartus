from pathlib import Path

import numpy as np
import h5py


class NexusSlicer:
    def __init__(self, in_file: Path | h5py.File, entry_path: str):
        pass

    def __getitem__(
        self,
        inx: slice
        | tuple[
            slice
            | tuple[str, float]
            | tuple[str, float, float]
            | tuple[str, float, float, float],
            ...,
        ],
    ) -> "NexusSlicer":
        return self

    def accumulate(self, accumulator: np.ufunc, *axis: str) -> "NexusSlicer":
        return self

    def loop(self, *axis: str) -> "NexusSlicer":
        return self

    @property
    def ndim(self) -> int:
        return 0

    @property
    def signal_name(self) -> str:
        return ""

    def axes(self, dimension: int) -> list[str]:
        return []

    def as_dictionary(
        self, dataset_name
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        return {dataset_name: np.array([])}

    def store(self, dataset_name, nx_file: Path | h5py.File) -> None:
        pass
