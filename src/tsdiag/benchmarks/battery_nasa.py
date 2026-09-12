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
    eol_observed: bool
    last_observed_cycle: int
    true_eol_cycle: int | None
    true_rul_cycles: int | None
    censoring_lower_bound_rul_cycles: int | None
    predicted_rul_cycles: float | None
    predicted_eol_cycle: float | None
    absolute_error_cycles: float | None
    relative_error: float | None
    censoring_consistent: bool | None
    censoring_margin_cycles: float | None
    uncertainty_cycles: float | None
    confidence: float
    abstained: bool
    abstain_reason: str | None
    runtime_seconds: float


def _predicted(row: BatteryPrognosisCase) -> bool:
    return not row.abstained and row.predicted_rul_cycles is not None


def _point_metrics(rows: list[BatteryPrognosisCase]) -> dict:
    evaluable = [row for row in rows if row.eol_observed and row.true_rul_cycles is not None]
    covered = [row for row in evaluable if _predicted(row)]
    errors = np.asarray(
        [row.absolute_error_cycles for row in covered if row.absolute_error_cycles is not None],
        dtype=float,
    )
    signed = np.asarray(
        [
            row.predicted_rul_cycles - row.true_rul_cycles
            for row in covered
            if row.predicted_rul_cycles is not None and row.true_rul_cycles is not None
        ],
        dtype=float,
    )
    rel = np.asarray(
        [row.relative_error for row in covered if row.relative_error is not None],
        dtype=float,
    )
    return {
        "case_count": len(evaluable),
        "covered_case_count": len(covered),
        "coverage": float(len(covered) / len(evaluable)) if evaluable else None,
        "abstention_rate": float(1.0 - len(covered) / len(evaluable)) if evaluable else None,
        "mae_cycles": float(np.mean(errors)) if errors.size else None,
        "rmse_cycles": float(np.sqrt(np.mean(signed**2))) if signed.size else None,
        "mean_relative_error": float(np.mean(rel)) if rel.size else None,
        "median_relative_error": float(np.median(rel)) if rel.size else None,
        "mean_signed_error_cycles": float(np.mean(signed)) if signed.size else None,
    }


def _censoring_metrics(rows: list[BatteryPrognosisCase]) -> dict:
    censored = [row for row in rows if not row.eol_observed]
    covered = [row for row in censored if _predicted(row)]
    consistent = [row for row in covered if row.censoring_consistent is True]
    margins = np.asarray(
        [row.censoring_margin_cycles for row in covered if row.censoring_margin_cycles is not None],
        dtype=float,
    )
    return {
        "case_count": len(censored),
        "covered_case_count": len(covered),
        "coverage": float(len(covered) / len(censored)) if censored else None,
        "consistent_prediction_count": len(consistent),
        "consistency_rate_on_predictions": (
            float(len(consistent) / len(covered)) if covered else None
        ),
        "mean_margin_beyond_last_observed_cycle": (
            float(np.mean(margins)) if margins.size else None
        ),
    }


def _subset_summary(rows: list[BatteryPrognosisCase]) -> dict:
    predicted = [row for row in rows if _predicted(row)]
    point = _point_metrics(rows)
    censored = _censoring_metrics(rows)
    return {
        "case_count": len(rows),
        "covered_case_count": len(predicted),
        "coverage": float(len(predicted) / len(rows)) if rows else None,
        "abstention_rate": float(1.0 - len(predicted) / len(rows)) if rows else None,
        "evaluable_case_count": point["case_count"],
        "censored_case_count": censored["case_count"],
        "mae_cycles": point["mae_cycles"],
        "censoring_consistency_rate_on_predictions": censored[
            "consistency_rate_on_predictions"
        ],
    }


def _summary(cases: list[BatteryPrognosisCase]) -> dict:
    if not cases:
        return {"case_count": 0}

    predicted = [row for row in cases if _predicted(row)]
    point = _point_metrics(cases)
    censored = _censoring_metrics(cases)

    by_cutpoint: dict[str, dict] = {}
    for cutpoint in sorted({row.observation_cycle for row in cases}):
        subset = [row for row in cases if row.observation_cycle == cutpoint]
        by_cutpoint[str(cutpoint)] = _subset_summary(subset)

    by_battery: dict[str, dict] = {}
    for battery_id in sorted({row.battery_id for row in cases}):
        subset = [row for row in cases if row.battery_id == battery_id]
        by_battery[battery_id] = _subset_summary(subset)

    return {
        "case_count": len(cases),
        "covered_case_count": len(predicted),
        "coverage_all_cases": float(len(predicted) / len(cases)),
        "abstention_rate_all_cases": float(1.0 - len(predicted) / len(cases)),
        "point_error_metrics": point,
        "censoring_metrics": censored,
        # Backward-compatible point-error fields. Censored cases never contribute.
        "coverage": point["coverage"],
        "abstention_rate": point["abstention_rate"],
        "mae_cycles": point["mae_cycles"],
        "rmse_cycles": point["rmse_cycles"],
        "mean_relative_error": point["mean_relative_error"],
        "median_relative_error": point["median_relative_error"],
        "mean_signed_error_cycles": point["mean_signed_error_cycles"],
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
    censored_cells: list[dict] = []
    cycle_features: dict[str, list[dict]] = {}

    for battery_id in battery_ids:
        try:
            cycles = load_nasa_battery_discharge_cycles(root / f"{battery_id}.mat", battery_id)
            eol_cycle = first_eol_cycle(cycles, eol_capacity_ah=NASA_EOL_CAPACITY_AH)
            cycle_features[battery_id] = [
                {"cycle_index": cycle.cycle_index, **discharge_curve_features(cycle)}
                for cycle in cycles
            ]

            capacities = np.asarray([cycle.capacity_ah for cycle in cycles], dtype=float)
            indices = np.asarray([cycle.cycle_index for cycle in cycles], dtype=int)
            last_observed_cycle = int(indices[-1])

            if eol_cycle is None:
                censored_cells.append({
                    "battery_id": battery_id,
                    "last_observed_cycle": last_observed_cycle,
                    "last_observed_capacity_ah": float(capacities[-1]),
                    "censoring_reason": (
                        "Dataset ends before the fixed 1.4 Ah EOL threshold is observed."
                    ),
                })

            for cutpoint in cutpoints:
                if cutpoint > last_observed_cycle:
                    continue
                if eol_cycle is not None and cutpoint >= eol_cycle:
                    continue

                mask = indices <= int(cutpoint)
                started = perf_counter()
                result = BatteryPrognosticPipeline().run(
                    indices[mask],
                    capacities[mask],
                    battery_id=battery_id,
                )
                runtime = perf_counter() - started

                true_rul = None if eol_cycle is None else int(eol_cycle - cutpoint)
                lower_bound_rul = (
                    int(last_observed_cycle - cutpoint) if eol_cycle is None else None
                )
                predicted_rul = (
                    None
                    if result.prognosis is None or result.prognosis.remaining_useful_life is None
                    else float(result.prognosis.remaining_useful_life)
                )
                predicted_eol = None if predicted_rul is None else float(cutpoint + predicted_rul)
                absolute_error = (
                    None
                    if predicted_rul is None or true_rul is None
                    else float(abs(predicted_rul - true_rul))
                )
                relative_error = (
                    None
                    if absolute_error is None or true_rul is None
                    else float(absolute_error / max(true_rul, 1))
                )
                censoring_consistent = (
                    None
                    if eol_cycle is not None or predicted_eol is None
                    else bool(predicted_eol > last_observed_cycle)
                )
                censoring_margin = (
                    None
                    if eol_cycle is not None or predicted_eol is None
                    else float(predicted_eol - last_observed_cycle)
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
                    eol_observed=eol_cycle is not None,
                    last_observed_cycle=last_observed_cycle,
                    true_eol_cycle=(None if eol_cycle is None else int(eol_cycle)),
                    true_rul_cycles=true_rul,
                    censoring_lower_bound_rul_cycles=lower_bound_rul,
                    predicted_rul_cycles=predicted_rul,
                    predicted_eol_cycle=predicted_eol,
                    absolute_error_cycles=absolute_error,
                    relative_error=relative_error,
                    censoring_consistent=censoring_consistent,
                    censoring_margin_cycles=censoring_margin,
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
            "information_rule": (
                "Each prediction uses only discharge-capacity history up to the observation cutpoint."
            ),
            "threshold_tuning": (
                "No benchmark labels or future capacity values are used to tune predictor thresholds."
            ),
            "primary_task": "remaining discharge cycles to first observed capacity <= 1.4 Ah",
            "censoring_rule": (
                "Cells whose recorded data end above 1.4 Ah are retained as right-censored. "
                "They do not receive point RUL error; a non-abstained prediction is consistent only "
                "when predicted EOL lies after the last observed cycle."
            ),
        },
        "summary": _summary(cases),
        "cases": [asdict(row) for row in cases],
        "censored_cells": censored_cells,
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
