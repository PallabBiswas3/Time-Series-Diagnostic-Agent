from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..datasets.battery_nasa import (
    NASA_BATTERY_IDS,
    NASA_EOL_CAPACITY_AH,
    discharge_curve_features,
    first_eol_cycle,
    load_nasa_battery_discharge_cycles,
)
from ..domains.battery_runner import BatteryPrognosticPipeline


DEFAULT_CUTPOINTS = (50, 70, 90, 110)


@dataclass
class BatteryPrognosisCase:
    battery_id: str
    observation_cycle: int
    observed_capacity_ah: float
    observed_soh: float
    true_eol_cycle: int
    true_rul_cycles: int
    predicted_rul_cycles: float | None
    predicted_eol_cycle: float | None
    absolute_error_cycles: float | None
    relative_error: float | None
    uncertainty_cycles: float | None
    confidence: float
    abstained: bool
    abstain_reason: str | None
    runtime_seconds: float


def _summary(cases: list[BatteryPrognosisCase]) -> dict:
    if not cases:
        return {"case_count": 0}
    covered = [row for row in cases if not row.abstained and row.predicted_rul_cycles is not None]
    errors = np.asarray([row.absolute_error_cycles for row in covered], dtype=float)
    signed = np.asarray([row.predicted_rul_cycles - row.true_rul_cycles for row in covered], dtype=float)
    rel = np.asarray([row.relative_error for row in covered], dtype=float)
    by_cutpoint: dict[str, dict] = {}
    for cutpoint in sorted({row.observation_cycle for row in cases}):
        subset = [row for row in cases if row.observation_cycle == cutpoint]
        subset_cov = [row for row in subset if not row.abstained and row.predicted_rul_cycles is not None]
        by_cutpoint[str(cutpoint)] = {
            "case_count": len(subset),
            "coverage": float(len(subset_cov) / len(subset)),
            "mae_cycles": (
                float(np.mean([row.absolute_error_cycles for row in subset_cov])) if subset_cov else None
            ),
        }

    by_battery: dict[str, dict] = {}
    for battery_id in sorted({row.battery_id for row in cases}):
        subset = [row for row in cases if row.battery_id == battery_id]
        subset_cov = [row for row in subset if not row.abstained and row.predicted_rul_cycles is not None]
        by_battery[battery_id] = {
            "case_count": len(subset),
            "coverage": float(len(subset_cov) / len(subset)),
            "mae_cycles": (
                float(np.mean([row.absolute_error_cycles for row in subset_cov])) if subset_cov else None
            ),
        }

    return {
        "case_count": len(cases),
        "covered_case_count": len(covered),
        "coverage": float(len(covered) / len(cases)),
        "abstention_rate": float(1.0 - len(covered) / len(cases)),
        "mae_cycles": float(np.mean(errors)) if errors.size else None,
        "rmse_cycles": float(np.sqrt(np.mean(signed**2))) if signed.size else None,
        "mean_relative_error": float(np.mean(rel)) if rel.size else None,
        "median_relative_error": float(np.median(rel)) if rel.size else None,
        "mean_signed_error_cycles": float(np.mean(signed)) if signed.size else None,
        "by_cutpoint": by_cutpoint,
        "by_battery": by_battery,
    }


def run_nasa_battery_benchmark(
    data_dir: str | Path,
    *,
    battery_ids: tuple[str, ...] = NASA_BATTERY_IDS,
    cutpoints: tuple[int, ...] = DEFAULT_CUTPOINTS,
    output_dir: str | Path | None = None,
) -> dict:
    root = Path(data_dir)
    cases: list[BatteryPrognosisCase] = []
    failures: list[dict] = []
    cycle_features: dict[str, list[dict]] = {}

    for battery_id in battery_ids:
        try:
            cycles = load_nasa_battery_discharge_cycles(root / f"{battery_id}.mat", battery_id)
            eol_cycle = first_eol_cycle(cycles, eol_capacity_ah=NASA_EOL_CAPACITY_AH)
            cycle_features[battery_id] = [
                {"cycle_index": cycle.cycle_index, **discharge_curve_features(cycle)} for cycle in cycles
            ]
            if eol_cycle is None:
                failures.append({
                    "battery_id": battery_id,
                    "error": "No observed discharge cycle reaches the fixed 1.4 Ah EOL threshold.",
                })
                continue

            capacities = np.asarray([cycle.capacity_ah for cycle in cycles], dtype=float)
            indices = np.asarray([cycle.cycle_index for cycle in cycles], dtype=int)
            for cutpoint in cutpoints:
                if cutpoint >= eol_cycle or cutpoint > len(cycles):
                    continue
                mask = indices <= int(cutpoint)
                started = perf_counter()
                result = BatteryPrognosticPipeline().run(
                    indices[mask],
                    capacities[mask],
                    battery_id=battery_id,
                )
                runtime = perf_counter() - started
                true_rul = int(eol_cycle - cutpoint)
                predicted_rul = (
                    None
                    if result.prognosis is None or result.prognosis.remaining_useful_life is None
                    else float(result.prognosis.remaining_useful_life)
                )
                predicted_eol = (
                    None if predicted_rul is None else float(cutpoint + predicted_rul)
                )
                absolute_error = (
                    None if predicted_rul is None else float(abs(predicted_rul - true_rul))
                )
                relative_error = (
                    None if absolute_error is None else float(absolute_error / max(true_rul, 1))
                )
                uncertainty_cycles = None
                if result.prognosis is not None:
                    raw_uncertainty = result.prognosis.details.get("uncertainty_cycles")
                    if raw_uncertainty is not None:
                        uncertainty_cycles = float(raw_uncertainty)

                cases.append(BatteryPrognosisCase(
                    battery_id=battery_id,
                    observation_cycle=int(cutpoint),
                    observed_capacity_ah=float(capacities[cutpoint - 1]),
                    observed_soh=float(capacities[cutpoint - 1] / 2.0),
                    true_eol_cycle=int(eol_cycle),
                    true_rul_cycles=true_rul,
                    predicted_rul_cycles=predicted_rul,
                    predicted_eol_cycle=predicted_eol,
                    absolute_error_cycles=absolute_error,
                    relative_error=relative_error,
                    uncertainty_cycles=uncertainty_cycles,
                    confidence=float(result.confidence),
                    abstained=bool(result.abstained),
                    abstain_reason=result.abstain_reason,
                    runtime_seconds=float(runtime),
                ))
        except Exception as exc:
            failures.append({"battery_id": battery_id, "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "source": "NASA Ames Prognostics Center of Excellence Battery Aging Dataset",
        "protocol": {
            "battery_ids": list(battery_ids),
            "nominal_capacity_ah": 2.0,
            "eol_capacity_ah": NASA_EOL_CAPACITY_AH,
            "cutpoints": list(cutpoints),
            "information_rule": "Each prediction uses only discharge-capacity history up to the observation cutpoint.",
            "threshold_tuning": "No benchmark labels or future capacity values are used to tune predictor thresholds.",
            "primary_task": "remaining discharge cycles to first observed capacity <= 1.4 Ah",
        },
        "summary": _summary(cases),
        "cases": [asdict(row) for row in cases],
        "cycle_features": cycle_features,
        "failures": failures,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "battery_nasa_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if cases:
            with (out / "battery_nasa_cases.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(cases[0]).keys()))
                writer.writeheader()
                writer.writerows(asdict(row) for row in cases)
    return payload
