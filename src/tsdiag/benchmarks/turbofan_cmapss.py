from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..datasets.cmapss import (
    CMAPSS_SENSOR_NAMES,
    CMAPSS_SUBSETS,
    load_cmapss_rul,
    load_cmapss_trajectories,
)
from ..domains.runners import TurbofanDiagnosticPipeline


@dataclass
class CmapssRulCase:
    subset: str
    unit_id: int
    observed_cycles: int
    true_rul_cycles: float
    predicted_rul_cycles: float | None
    signed_error_cycles: float | None
    absolute_error_cycles: float | None
    confidence: float
    abstained: bool
    verification_status: str | None
    runtime_seconds: float


def _nasa_score(error: float) -> float:
    # Standard PHM/C-MAPSS asymmetric score: late/optimistic RUL estimates are
    # penalized more strongly than conservative early warnings.
    if error < 0:
        return float(np.exp(-error / 13.0) - 1.0)
    return float(np.exp(error / 10.0) - 1.0)


def _summarize(rows: list[CmapssRulCase]) -> dict:
    covered = [r for r in rows if not r.abstained and r.predicted_rul_cycles is not None]
    signed = np.asarray([r.signed_error_cycles for r in covered], dtype=float)
    absolute = np.asarray([r.absolute_error_cycles for r in covered], dtype=float)
    runtimes = np.asarray([r.runtime_seconds for r in rows], dtype=float)
    return {
        "case_count": len(rows),
        "covered_case_count": len(covered),
        "coverage": None if not rows else float(len(covered) / len(rows)),
        "abstention_rate": None if not rows else float(1.0 - len(covered) / len(rows)),
        "mae_cycles": None if not absolute.size else float(np.mean(absolute)),
        "rmse_cycles": None if not signed.size else float(np.sqrt(np.mean(signed**2))),
        "mean_signed_error_cycles": None if not signed.size else float(np.mean(signed)),
        "median_absolute_error_cycles": None if not absolute.size else float(np.median(absolute)),
        "nasa_asymmetric_score": None if not signed.size else float(np.sum([_nasa_score(v) for v in signed])),
        "mean_runtime_seconds": None if not runtimes.size else float(np.mean(runtimes)),
    }


def run_cmapss_benchmark(
    data_dir: str | Path,
    *,
    subsets: tuple[str, ...] = CMAPSS_SUBSETS,
    output_dir: str | Path | None = None,
) -> dict:
    """Evaluate the deterministic public turbofan pipeline on NASA C-MAPSS.

    The pipeline sees only each test engine's observed trajectory. NASA RUL
    labels are consumed exclusively here in the benchmark/evaluation layer.
    """
    root = Path(data_dir)
    cases: list[CmapssRulCase] = []
    failures: list[dict] = []
    by_subset: dict[str, dict] = {}

    for subset in tuple(str(s).upper() for s in subsets):
        subset_rows: list[CmapssRulCase] = []
        try:
            trajectories = load_cmapss_trajectories(root / f"test_{subset}.txt", subset)
            truth = load_cmapss_rul(root / f"RUL_{subset}.txt")
            if len(trajectories) != len(truth):
                raise ValueError(
                    f"{subset}: {len(trajectories)} test trajectories != {len(truth)} RUL targets"
                )

            for trajectory, true_rul in zip(trajectories, truth):
                started = perf_counter()
                result = TurbofanDiagnosticPipeline().run(
                    trajectory.sensors,
                    list(CMAPSS_SENSOR_NAMES),
                    trajectory.cycle_index,
                    operating_conditions=trajectory.operating_settings,
                )
                runtime = perf_counter() - started
                pred = (
                    None
                    if result.prognosis is None or result.prognosis.remaining_useful_life is None
                    else float(result.prognosis.remaining_useful_life)
                )
                signed = None if pred is None else float(pred - float(true_rul))
                absolute = None if signed is None else float(abs(signed))
                verification_status = None
                if result.verification:
                    verification_status = str(result.verification[0].status)
                row = CmapssRulCase(
                    subset=subset,
                    unit_id=int(trajectory.unit_id),
                    observed_cycles=int(len(trajectory.cycle_index)),
                    true_rul_cycles=float(true_rul),
                    predicted_rul_cycles=pred,
                    signed_error_cycles=signed,
                    absolute_error_cycles=absolute,
                    confidence=float(result.confidence),
                    abstained=bool(result.abstained or pred is None),
                    verification_status=verification_status,
                    runtime_seconds=float(runtime),
                )
                cases.append(row)
                subset_rows.append(row)
            by_subset[subset] = _summarize(subset_rows)
        except Exception as exc:
            failures.append({"subset": subset, "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "source": "NASA Ames Prognostics Center of Excellence C-MAPSS Turbofan Engine Degradation Simulation Data Set",
        "protocol": {
            "subsets": [str(s).upper() for s in subsets],
            "prediction_time": "End of each published test trajectory",
            "label_isolation": "RUL labels are used only in this benchmark module, never by production diagnosis code.",
            "sensor_channels": list(CMAPSS_SENSOR_NAMES),
            "operating_settings": 3,
            "model": "deterministic public turbofan health-index/RUL baseline",
            "threshold_tuning": "No C-MAPSS test RUL labels are used to tune thresholds or model parameters.",
        },
        "summary": _summarize(cases),
        "by_subset": by_subset,
        "cases": [asdict(row) for row in cases],
        "failures": failures,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "turbofan_cmapss_benchmark.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if cases:
            with (out / "turbofan_cmapss_cases.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(cases[0]).keys()))
                writer.writeheader()
                writer.writerows(asdict(row) for row in cases)
    return payload
