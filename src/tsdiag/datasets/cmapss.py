from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

CMAPSS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
CMAPSS_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
CMAPSS_SENSOR_NAMES = tuple(f"sensor_{i}" for i in range(1, 22))
CMAPSS_SETTING_NAMES = ("setting_1", "setting_2", "setting_3")
CMAPSS_COLUMN_COUNT = 26


@dataclass(frozen=True)
class CmapssTrajectory:
    subset: str
    unit_id: int
    cycle_index: np.ndarray
    operating_settings: np.ndarray
    sensors: np.ndarray


def _load_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != CMAPSS_COLUMN_COUNT:
        raise ValueError(
            f"C-MAPSS file must have {CMAPSS_COLUMN_COUNT} columns; got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("C-MAPSS file contains non-finite values")
    return matrix


def load_cmapss_trajectories(path: str | Path, subset: str) -> list[CmapssTrajectory]:
    subset = str(subset).upper()
    if subset not in CMAPSS_SUBSETS:
        raise ValueError(f"Unsupported C-MAPSS subset: {subset}")
    matrix = _load_matrix(path)
    unit_ids = matrix[:, 0].astype(int)
    output: list[CmapssTrajectory] = []
    for unit_id in np.unique(unit_ids):
        rows = matrix[unit_ids == unit_id]
        order = np.argsort(rows[:, 1])
        rows = rows[order]
        cycles = rows[:, 1].astype(float)
        if np.any(np.diff(cycles) <= 0):
            raise ValueError(f"Unit {unit_id} cycles are not strictly increasing")
        output.append(
            CmapssTrajectory(
                subset=subset,
                unit_id=int(unit_id),
                cycle_index=cycles,
                operating_settings=rows[:, 2:5].astype(float, copy=False),
                sensors=rows[:, 5:26].astype(float, copy=False),
            )
        )
    return output


def load_cmapss_rul(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float).reshape(-1)
    if not values.size or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("C-MAPSS RUL target file must contain finite non-negative values")
    return values


def validate_cmapss_subset(root: str | Path, subset: str) -> dict:
    root = Path(root)
    subset = str(subset).upper()
    train = load_cmapss_trajectories(root / f"train_{subset}.txt", subset)
    test = load_cmapss_trajectories(root / f"test_{subset}.txt", subset)
    rul = load_cmapss_rul(root / f"RUL_{subset}.txt")
    if len(test) != len(rul):
        raise ValueError(
            f"{subset}: test unit count {len(test)} does not match RUL target count {len(rul)}"
        )
    return {
        "subset": subset,
        "train_units": len(train),
        "test_units": len(test),
        "rul_targets": len(rul),
        "sensor_count": len(CMAPSS_SENSOR_NAMES),
        "setting_count": len(CMAPSS_SETTING_NAMES),
    }


def download_cmapss(
    destination: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    overwrite: bool = False,
) -> list[Path]:
    """Download the fixed NASA C-MAPSS train/test/RUL text files.

    Files are extracted by basename from NASA's official archive so the loader is
    insensitive to the archive's enclosing directory name.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    normalized = tuple(str(s).upper() for s in subsets)
    required = [
        root / name
        for subset in normalized
        for name in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt")
    ]
    if not overwrite and all(path.exists() and path.stat().st_size > 0 for path in required):
        return required

    with urlopen(CMAPSS_URL, timeout=120) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        members = archive.namelist()
        for output_path in required:
            matches = [name for name in members if Path(name).name == output_path.name]
            if len(matches) != 1:
                raise ValueError(f"Expected one archive member for {output_path.name}, found {matches}")
            output_path.write_bytes(archive.read(matches[0]))

    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Missing/empty C-MAPSS file after download: {path}")
    return required
