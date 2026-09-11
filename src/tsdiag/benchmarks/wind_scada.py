from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from ..datasets.wind_care import (
    CARE_DOI,
    CARE_SAMPLE_PERIOD_MINUTES,
    care_normal_mask,
    care_sensor_frame,
    load_care_event,
    load_care_event_info,
    validate_care_layout,
)
from ..domains.wind_scada_runner import WindScadaDiagnosticPipeline
from ..evaluation.wind_scada import evaluate_wind_event, summarize_wind_events


@dataclass
class WindBenchmarkEventResult:
    event_id: int
    wind_farm: str
    asset_id: str
    true_label: str
    event_detected: bool
    max_criticality: int
    alarm_fraction: float
    residual_alarm_fraction: float
    drift_alarm_fraction: float
    fused_alarm_fraction: float
    event_window_recall: float | None
    first_detection_index: int | None
    lead_time_minutes: float | None
    affected_channels: list[str]
    change_point_count: int
    physics_finding_count: int
    confidence: float
    tool_call_count: int
    runtime_seconds: float
    n_train: int
    n_prediction: int
    n_channels: int


def _ids(values: Iterable[int] | None, available: Iterable[int]) -> list[int]:
    available_ids = [int(v) for v in available]
    if values is None:
        return available_ids
    requested = {int(v) for v in values}
    return [v for v in available_ids if v in requested]


def _aligned_sensor_matrices(event_data):
    train_sensor, _, train_timestamps = care_sensor_frame(event_data.train)
    pred_sensor, _, pred_timestamps = care_sensor_frame(event_data.prediction)
    common = [name for name in train_sensor.columns if name in pred_sensor.columns]
    if not common:
        raise ValueError(f"event {event_data.event.event_id} has no aligned numeric SCADA channels")
    return (
        train_sensor[common].to_numpy(dtype=float),
        pred_sensor[common].to_numpy(dtype=float),
        common,
        None if train_timestamps is None else train_timestamps.to_numpy(),
        None if pred_timestamps is None else pred_timestamps.to_numpy(),
    )


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_summary_markdown(payload: dict, path: Path) -> None:
    summary = payload["summary"]
    cfg = payload["configuration"]
    lines = [
        "# CARE v6 Wind-SCADA Benchmark",
        "",
        f"Source: CARE to Compare v6 (`{CARE_DOI}`).",
        "",
        "## Primary results",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Successful events | {summary.get('successful_events', 0)} |",
        f"| Failed events | {summary.get('failed_events', 0)} |",
        f"| Event recall | {_fmt(summary.get('event_recall'))} |",
        f"| Event precision | {_fmt(summary.get('event_precision'))} |",
        f"| Event F1 | {_fmt(summary.get('event_f1'))} |",
        f"| Normal-event false alarm rate | {_fmt(summary.get('normal_event_false_alarm_rate'))} |",
        f"| Mean anomaly-window recall | {_fmt(summary.get('mean_event_window_recall'))} |",
        f"| Mean early-warning lead time (min) | {_fmt(summary.get('mean_lead_time_minutes'), 1)} |",
        f"| Median early-warning lead time (min) | {_fmt(summary.get('median_lead_time_minutes'), 1)} |",
        f"| Mean max CARE-style criticality | {_fmt(summary.get('mean_max_criticality'), 2)} |",
        f"| Mean anomaly-event criticality | {_fmt(summary.get('mean_anomaly_event_criticality'), 2)} |",
        f"| Mean normal-event criticality | {_fmt(summary.get('mean_normal_event_criticality'), 2)} |",
        f"| Mean residual alarm fraction | {_fmt(summary.get('mean_residual_alarm_fraction'))} |",
        f"| Mean CUSUM alarm fraction | {_fmt(summary.get('mean_drift_alarm_fraction'))} |",
        f"| Mean fused alarm fraction | {_fmt(summary.get('mean_fused_alarm_fraction'))} |",
        f"| Mean tool calls / event | {_fmt(summary.get('mean_tool_calls'), 2)} |",
        f"| Mean runtime / event (s) | {_fmt(summary.get('mean_runtime_seconds'), 3)} |",
        f"| Total runtime (s) | {_fmt(summary.get('total_runtime_seconds'), 2)} |",
        "",
        "## Configuration",
        "",
        f"- Healthy operating regimes: {cfg['n_regimes']}",
        f"- Residual threshold: {cfg['residual_threshold']}",
        f"- Residual persistence: {cfg['persistence']}",
        f"- CUSUM drift allowance: {cfg['cusum_drift']}",
        f"- CUSUM threshold: {cfg['cusum_threshold']}",
        f"- CARE-style criticality threshold: {cfg['criticality_threshold']}",
        "- Event labels are used only for evaluation. Training rows are filtered using operator/status healthy labels.",
        "- Physics checks are verification evidence; missing anonymized turbine specifications disable only the corresponding bound.",
        "",
        "## Event failures",
        "",
    ]
    failures = payload.get("failures", [])
    if failures:
        lines.extend([f"- Event {row['event_id']}: `{row['error']}`" for row in failures])
    else:
        lines.append("No event-level benchmark failures.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_care_benchmark(
    data_dir: str | Path,
    *,
    event_ids: Iterable[int] | None = None,
    wind_farm: str | None = None,
    n_regimes: int = 4,
    residual_threshold: float = 3.5,
    persistence: int = 3,
    cusum_drift: float = 0.25,
    cusum_threshold: float = 8.0,
    cusum_hold_samples: int = 6,
    criticality_threshold: int = 72,
    output_dir: str | Path | None = None,
) -> dict:
    root = Path(data_dir)
    layout = validate_care_layout(root)
    if not all(layout.values()) and wind_farm is None:
        raise FileNotFoundError(f"incomplete CARE layout: {layout}")
    if wind_farm is not None and not layout.get(str(wind_farm).upper(), False):
        raise FileNotFoundError(f"CARE Wind Farm {wind_farm} is not available")

    info = load_care_event_info(root, wind_farm=wind_farm)
    selected_ids = _ids(event_ids, info["event_id"].astype(int).tolist())
    event_rows: list[WindBenchmarkEventResult] = []
    evaluation_rows = []
    failures = []
    total_started = perf_counter()

    for event_id in selected_ids:
        try:
            event_data = load_care_event(root, event_id, statistics=("avg",))
            if event_data.event.is_anomaly is None:
                raise ValueError("event_label is unavailable")
            train, prediction, channel_names, train_ts, pred_ts = _aligned_sensor_matrices(event_data)
            healthy_train = care_normal_mask(event_data.train)

            pipeline = WindScadaDiagnosticPipeline(
                n_regimes=n_regimes,
                residual_threshold=residual_threshold,
                persistence=persistence,
                cusum_drift=cusum_drift,
                cusum_threshold=cusum_threshold,
                cusum_hold_samples=cusum_hold_samples,
            )
            started = perf_counter()
            result = pipeline.run(
                train,
                prediction,
                channel_names,
                train_timestamps=train_ts,
                prediction_timestamps=pred_ts,
                healthy_train_mask=healthy_train,
            )
            runtime = perf_counter() - started

            if event_data.event.wind_farm == "A":
                pred_normal = np.ones(len(prediction), dtype=bool)
            else:
                pred_normal = care_normal_mask(event_data.prediction)

            evaluation = evaluate_wind_event(
                event_id=event_id,
                is_anomaly_event=bool(event_data.event.is_anomaly),
                alarm_mask=result.alarm_mask,
                timestamps=pred_ts,
                event_start=event_data.event.event_start,
                event_end=event_data.event.event_end,
                normal_mask=pred_normal,
                criticality_threshold=criticality_threshold,
                sample_period_minutes=CARE_SAMPLE_PERIOD_MINUTES,
            )
            evaluation_rows.append(evaluation)
            cp = result.artifacts.get("residual_changepoint", {})
            physics = result.artifacts.get("physics_consistency", {})
            fusion = result.artifacts.get("fusion", {})
            event_rows.append(
                WindBenchmarkEventResult(
                    event_id=event_id,
                    wind_farm=event_data.event.wind_farm,
                    asset_id=event_data.event.asset_id,
                    true_label="anomaly" if event_data.event.is_anomaly else "normal",
                    event_detected=evaluation.event_detected,
                    max_criticality=evaluation.max_criticality,
                    alarm_fraction=evaluation.alarm_fraction,
                    residual_alarm_fraction=float(fusion.get("residual_alarm_fraction", 0.0)),
                    drift_alarm_fraction=float(fusion.get("drift_alarm_fraction", 0.0)),
                    fused_alarm_fraction=float(fusion.get("fused_alarm_fraction", 0.0)),
                    event_window_recall=evaluation.event_window_recall,
                    first_detection_index=evaluation.first_detection_index,
                    lead_time_minutes=evaluation.lead_time_minutes,
                    affected_channels=result.affected_channels,
                    change_point_count=len(cp.get("change_points", [])),
                    physics_finding_count=len(physics.get("verification_findings", [])),
                    confidence=result.confidence,
                    tool_call_count=len(result.tool_trace),
                    runtime_seconds=float(runtime),
                    n_train=int(train.shape[0]),
                    n_prediction=int(prediction.shape[0]),
                    n_channels=int(train.shape[1]),
                )
            )
        except Exception as exc:
            failures.append({"event_id": int(event_id), "error": f"{type(exc).__name__}: {exc}"})

    total_runtime = perf_counter() - total_started
    summary = summarize_wind_events(evaluation_rows)
    summary.update(
        {
            "successful_events": len(event_rows),
            "failed_events": len(failures),
            "mean_runtime_seconds": float(np.mean([r.runtime_seconds for r in event_rows])) if event_rows else None,
            "total_runtime_seconds": float(total_runtime),
            "mean_tool_calls": float(np.mean([r.tool_call_count for r in event_rows])) if event_rows else None,
            "total_tool_calls": int(sum(r.tool_call_count for r in event_rows)),
            "mean_residual_alarm_fraction": float(np.mean([r.residual_alarm_fraction for r in event_rows])) if event_rows else None,
            "mean_drift_alarm_fraction": float(np.mean([r.drift_alarm_fraction for r in event_rows])) if event_rows else None,
            "mean_fused_alarm_fraction": float(np.mean([r.fused_alarm_fraction for r in event_rows])) if event_rows else None,
        }
    )

    payload = {
        "source": f"CARE to Compare v6 ({CARE_DOI})",
        "data_dir": str(root),
        "configuration": {
            "statistics": ["avg"],
            "n_regimes": int(n_regimes),
            "residual_threshold": float(residual_threshold),
            "persistence": int(persistence),
            "cusum_drift": float(cusum_drift),
            "cusum_threshold": float(cusum_threshold),
            "cusum_hold_samples": int(cusum_hold_samples),
            "criticality_threshold": int(criticality_threshold),
        },
        "summary": summary,
        "events": [asdict(row) for row in event_rows],
        "failures": failures,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "wind_scada_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_summary_markdown(payload, out / "care_v6_summary.md")
        rows = [asdict(row) for row in event_rows]
        if rows:
            with (out / "wind_scada_event_table.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                for row in rows:
                    row = dict(row)
                    row["affected_channels"] = "|".join(row["affected_channels"])
                    writer.writerow(row)

    return payload


def _parse_event_ids(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    out = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                out.append(int(token))
    return out


def main():
    parser = argparse.ArgumentParser(description="Run the CARE-to-Compare Wind-SCADA benchmark")
    parser.add_argument("--data-dir", required=True, help="CARE v6 extracted directory or official ZIP archive")
    parser.add_argument("--events", nargs="*", help="optional event IDs")
    parser.add_argument("--wind-farm", choices=["A", "B", "C"])
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--residual-threshold", type=float, default=3.5)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--cusum-drift", type=float, default=0.25)
    parser.add_argument("--cusum-threshold", type=float, default=8.0)
    parser.add_argument("--cusum-hold-samples", type=int, default=6)
    parser.add_argument("--criticality-threshold", type=int, default=72)
    parser.add_argument("--output-dir", default="outputs/wind_benchmark")
    args = parser.parse_args()

    result = run_care_benchmark(
        args.data_dir,
        event_ids=_parse_event_ids(args.events),
        wind_farm=args.wind_farm,
        n_regimes=args.n_regimes,
        residual_threshold=args.residual_threshold,
        persistence=args.persistence,
        cusum_drift=args.cusum_drift,
        cusum_threshold=args.cusum_threshold,
        cusum_hold_samples=args.cusum_hold_samples,
        criticality_threshold=args.criticality_threshold,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2))
    if result["failures"]:
        print("Event failures:")
        print(json.dumps(result["failures"], indent=2))


if __name__ == "__main__":
    main()
