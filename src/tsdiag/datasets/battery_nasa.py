from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat


NASA_BATTERY_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_BATTERY_IDS = ("B0005", "B0006", "B0007", "B0018")
NASA_NOMINAL_CAPACITY_AH = 2.0
NASA_EOL_CAPACITY_AH = 1.4


@dataclass(frozen=True)
class NASABatteryDischargeCycle:
    battery_id: str
    cycle_index: int
    capacity_ah: float
    ambient_temperature_c: float | None
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray
    time_s: np.ndarray

    @property
    def soh(self) -> float:
        return float(self.capacity_ah / NASA_NOMINAL_CAPACITY_AH)


def _as_scalar(value, default: float | None = None) -> float | None:
    try:
        arr = np.asarray(value, dtype=float).squeeze()
        if arr.size == 1 and np.isfinite(float(arr)):
            return float(arr)
    except (TypeError, ValueError):
        pass
    return default


def _as_vector(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.ravel()


def _field(obj, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise KeyError(name)


def _cycle_type(cycle) -> str:
    value = _field(cycle, "type")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip().lower()
    return str(np.asarray(value).squeeze()).strip().lower()


def load_nasa_battery_discharge_cycles(
    path: str | Path,
    battery_id: str | None = None,
) -> list[NASABatteryDischargeCycle]:
    path = Path(path)
    battery_id = battery_id or path.stem
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if battery_id not in payload:
        raise ValueError(f"{path.name} does not contain MATLAB variable {battery_id}")

    root = payload[battery_id]
    cycles = np.atleast_1d(_field(root, "cycle")).ravel()
    output: list[NASABatteryDischargeCycle] = []

    for cycle in cycles:
        if _cycle_type(cycle) != "discharge":
            continue
        data = _field(cycle, "data")
        capacity = _as_scalar(_field(data, "Capacity"))
        if capacity is None or capacity <= 0.0:
            continue

        voltage = _as_vector(_field(data, "Voltage_measured"))
        current = _as_vector(_field(data, "Current_measured"))
        temperature = _as_vector(_field(data, "Temperature_measured"))
        time_s = _as_vector(_field(data, "Time"))
        n = min(voltage.size, current.size, temperature.size, time_s.size)
        if n < 2:
            continue
        voltage = voltage[:n]
        current = current[:n]
        temperature = temperature[:n]
        time_s = time_s[:n]
        finite = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temperature) & np.isfinite(time_s)
        if int(np.sum(finite)) < 2:
            continue

        ambient = _as_scalar(_field(cycle, "ambient_temperature"), default=None)
        output.append(
            NASABatteryDischargeCycle(
                battery_id=battery_id,
                cycle_index=len(output) + 1,
                capacity_ah=float(capacity),
                ambient_temperature_c=ambient,
                voltage_v=voltage[finite].astype(float, copy=False),
                current_a=current[finite].astype(float, copy=False),
                temperature_c=temperature[finite].astype(float, copy=False),
                time_s=time_s[finite].astype(float, copy=False),
            )
        )

    if not output:
        raise ValueError(f"No usable discharge cycles found in {path}")
    return output


def discharge_curve_features(cycle: NASABatteryDischargeCycle) -> dict[str, float | None]:
    time_s = np.asarray(cycle.time_s, dtype=float)
    voltage = np.asarray(cycle.voltage_v, dtype=float)
    current = np.asarray(cycle.current_a, dtype=float)
    temperature = np.asarray(cycle.temperature_c, dtype=float)
    duration = float(max(time_s[-1] - time_s[0], 0.0))

    mean_voltage = float(np.mean(voltage))
    voltage_area = None
    if duration > 0.0:
        voltage_area = float(np.trapezoid(voltage, x=time_s) / duration)

    temp_rise = float(temperature[-1] - temperature[0])
    mean_abs_current = float(np.mean(np.abs(current)))
    time_above_35 = 0.0
    if duration > 0.0 and voltage.size >= 2:
        mask = voltage >= 3.5
        if np.any(mask):
            time_above_35 = float(np.trapezoid(mask.astype(float), x=time_s) / duration)

    return {
        "capacity_ah": float(cycle.capacity_ah),
        "soh": float(cycle.soh),
        "duration_s": duration,
        "mean_voltage_v": mean_voltage,
        "normalized_voltage_area_v": voltage_area,
        "mean_abs_current_a": mean_abs_current,
        "max_temperature_c": float(np.max(temperature)),
        "temperature_rise_c": temp_rise,
        "fraction_time_above_3p5v": time_above_35,
        "ambient_temperature_c": cycle.ambient_temperature_c,
    }


def first_eol_cycle(
    cycles: Iterable[NASABatteryDischargeCycle],
    *,
    eol_capacity_ah: float = NASA_EOL_CAPACITY_AH,
) -> int | None:
    for cycle in cycles:
        if float(cycle.capacity_ah) <= float(eol_capacity_ah):
            return int(cycle.cycle_index)
    return None


def download_nasa_battery_subset(
    destination: str | Path,
    *,
    battery_ids: Iterable[str] = NASA_BATTERY_IDS,
    overwrite: bool = False,
) -> list[Path]:
    """Download only the four fixed FY08Q4 NASA cells from the official archive.

    The NASA download is an outer ZIP containing inner experiment ZIPs. We read
    the FY08Q4 inner archive in memory and persist only the requested MAT files.
    Raw archives are not committed to the repository.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    ids = tuple(str(value) for value in battery_ids)
    expected = [root / f"{battery_id}.mat" for battery_id in ids]
    if not overwrite and all(path.exists() and path.stat().st_size > 0 for path in expected):
        return expected

    outer_path = root / "_nasa_battery_dataset.zip"
    urlretrieve(NASA_BATTERY_URL, outer_path)
    try:
        with ZipFile(outer_path) as outer:
            inner_names = [
                name
                for name in outer.namelist()
                if name.endswith("1. BatteryAgingARC-FY08Q4.zip")
            ]
            if len(inner_names) != 1:
                raise ValueError(f"Expected one FY08Q4 inner archive, found {inner_names}")
            inner_bytes = outer.read(inner_names[0])

        with ZipFile(BytesIO(inner_bytes)) as inner:
            names = inner.namelist()
            for battery_id, output_path in zip(ids, expected):
                matches = [name for name in names if name.endswith(f"{battery_id}.mat")]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {battery_id}.mat member, found {matches}")
                output_path.write_bytes(inner.read(matches[0]))
    finally:
        if outer_path.exists():
            outer_path.unlink()

    for path in expected:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"NASA battery download produced missing/empty file: {path}")
    return expected
