from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..datasets.battery_nasa import NASA_BATTERY_IDS, NASA_EOL_CAPACITY_AH, first_eol_cycle, load_nasa_battery_discharge_cycles
from ..domains.battery_prognosis_runner import BatteryPrognosticPipeline

DEFAULT_CUTPOINTS = (50, 70, 90, 110)


@dataclass
class BatteryPrognosisCase:
    battery_id: str
    observation_cycle: int
    eol_observed: bool
    last_observed_cycle: int
    true_eol_cycle: int | None
    true_rul_cycles: int | None
    predicted_rul_cycles: float | None
    predicted_eol_cycle: float | None
    predicted_eol_lower: float | None
    predicted_eol_upper: float | None
    absolute_error_cycles: float | None
    censoring_compatible: bool | None
    censoring_violation_cycles: float | None
    confidence: float
    abstained: bool
    runtime_seconds: float


def _predicted(row):
    return not row.abstained and row.predicted_rul_cycles is not None


def summarize_battery_cases(cases: list[BatteryPrognosisCase]) -> dict:
    exact = [r for r in cases if r.eol_observed and r.true_rul_cycles is not None]
    exact_pred = [r for r in exact if _predicted(r)]
    errors = np.asarray([r.absolute_error_cycles for r in exact_pred if r.absolute_error_cycles is not None], dtype=float)

    censored = [r for r in cases if not r.eol_observed]
    censored_pred = [r for r in censored if _predicted(r)]
    compatible = [r for r in censored_pred if r.censoring_compatible is True]
    violations = np.asarray([r.censoring_violation_cycles for r in censored_pred if r.censoring_violation_cycles is not None], dtype=float)

    return {
        "case_count": len(cases),
        "point_error_metrics": {
            "case_count": len(exact),
            "covered_case_count": len(exact_pred),
            "coverage": None if not exact else len(exact_pred) / len(exact),
            "mae_cycles": None if not errors.size else float(np.mean(errors)),
            "rmse_cycles": None if not errors.size else float(np.sqrt(np.mean(errors**2))),
        },
        "censoring_metrics": {
            "case_count": len(censored),
            "covered_case_count": len(censored_pred),
            "coverage": None if not censored else len(censored_pred) / len(censored),
            "interval_compatibility_rate": None if not censored_pred else len(compatible) / len(censored_pred),
            "mean_one_sided_violation_cycles": None if not violations.size else float(np.mean(violations)),
            "max_one_sided_violation_cycles": None if not violations.size else float(np.max(violations)),
        },
    }


def run_nasa_battery_benchmark(data_dir: str | Path, *, battery_ids=NASA_BATTERY_IDS, cutpoints=DEFAULT_CUTPOINTS) -> dict:
    root = Path(data_dir)
    cases: list[BatteryPrognosisCase] = []
    failures = []

    for battery_id in battery_ids:
        try:
            cycles = load_nasa_battery_discharge_cycles(root / f"{battery_id}.mat", battery_id)
            capacities = np.asarray([c.capacity_ah for c in cycles], dtype=float)
            indices = np.asarray([c.cycle_index for c in cycles], dtype=float)
            last_observed = int(indices[-1])
            eol = first_eol_cycle(cycles)

            for cutpoint in cutpoints:
                if cutpoint > last_observed or (eol is not None and cutpoint >= eol):
                    continue
                mask = indices <= cutpoint
                started = perf_counter()
                result = BatteryPrognosticPipeline(eol_capacity_ah=NASA_EOL_CAPACITY_AH).run(
                    indices[mask], capacities[mask], battery_id=battery_id
                )
                runtime = perf_counter() - started

                prognosis = result.prognosis
                pred_rul = None if prognosis is None else prognosis.remaining_useful_life
                pred_eol = None if pred_rul is None else float(cutpoint + pred_rul)
                interval = None if prognosis is None else prognosis.details.get("eol_interval_cycles")
                lower = None if not interval else float(interval[0])
                upper = None if not interval else float(interval[1])
                true_rul = None if eol is None else int(eol - cutpoint)
                abs_error = None if pred_rul is None or true_rul is None else float(abs(pred_rul - true_rul))

                # Right-censored cells only tell us that EOL happened after the
                # final recorded cycle. A prediction interval is compatible if
                # its optimistic upper bound reaches beyond that censoring time.
                censor_ok = None
                violation = None
                if eol is None and upper is not None:
                    censor_ok = bool(upper > last_observed)
                    violation = float(max(last_observed - upper, 0.0))

                cases.append(BatteryPrognosisCase(
                    battery_id=battery_id,
                    observation_cycle=int(cutpoint),
                    eol_observed=eol is not None,
                    last_observed_cycle=last_observed,
                    true_eol_cycle=None if eol is None else int(eol),
                    true_rul_cycles=true_rul,
                    predicted_rul_cycles=None if pred_rul is None else float(pred_rul),
                    predicted_eol_cycle=pred_eol,
                    predicted_eol_lower=lower,
                    predicted_eol_upper=upper,
                    absolute_error_cycles=abs_error,
                    censoring_compatible=censor_ok,
                    censoring_violation_cycles=violation,
                    confidence=float(result.confidence),
                    abstained=bool(result.abstained),
                    runtime_seconds=float(runtime),
                ))
        except Exception as exc:
            failures.append({"battery_id": battery_id, "error": f"{type(exc).__name__}: {exc}"})

    return {
        "source": "NASA Ames Prognostics Center of Excellence Battery Aging Dataset",
        "protocol": {
            "battery_ids": list(battery_ids),
            "cutpoints": list(cutpoints),
            "eol_capacity_ah": NASA_EOL_CAPACITY_AH,
            "information_rule": "Predictions use only capacity history available at each cutpoint.",
            "right_censoring_rule": "Censored cases are never assigned point RUL error. Interval compatibility is evaluated one-sided against the final observed healthy cycle.",
        },
        "summary": summarize_battery_cases(cases),
        "cases": [asdict(row) for row in cases],
        "failures": failures,
    }
