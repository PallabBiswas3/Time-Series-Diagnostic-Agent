from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..datasets.tep import (
    TEP_CHANNEL_NAMES,
    TEP_FAULTS,
    TEP_LOCALIZATION_PROXIES,
    TEP_SAMPLE_PERIOD_MIN,
    TEP_TEST_FAULT_START,
    download_braatz_tep,
    load_tep_dat,
    load_tep_reference,
)
from ..domains.process_runner import ProcessDiagnosticPipeline
from ..evaluation.reporting import (
    write_detection_plots,
    write_fault_table,
    write_json_report,
    write_markdown_summary,
)
from ..tools import (
    MonitoringConfig,
    calibrate_monitoring_config,
    pre_post_shift_evidence,
    run_monitoring_method,
    standardize_against_normal,
)


@dataclass
class TEPFaultMethodResult:
    method: str
    fault_id: int
    description: str
    fault_type: str
    n_samples: int
    calibration_alpha: float
    calibration_min_consecutive: int
    calibration_lags: int
    validation_false_alarm_rate: float | None
    pre_fault_false_alarm_rate: float
    post_fault_detection_rate: float | None
    first_detection_index: int | None
    detection_delay_samples: int | None
    detection_delay_minutes: float | None
    max_score_post_fault: float | None
    selected_diagnostic_channels: list[str]
    predicted_root_cause: str | None
    diagnostic_confidence: float | None
    localization_proxy_targets: list[str]
    localization_hit: bool | None
    predicted_fault_label: str | None
    abstain_reason: str | None


@dataclass
class TEPBenchmarkResult:
    source: str
    reference_shape: tuple[int, int]
    fault_start_index: int
    channel_count: int
    methods: list[str]
    calibration: dict[str, dict[str, Any]]
    faults: list[TEPFaultMethodResult]
    summary: dict[str, Any]
    artifacts: dict[str, str]


def _parse_ids(values: list[str] | None, default):
    if not values:
        return list(default)
    out = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                out.append(int(token))
    return out


def _parse_float_grid(value: str | None, default: Iterable[float]) -> tuple[float, ...]:
    if not value:
        return tuple(float(v) for v in default)
    return tuple(float(v.strip()) for v in value.split(",") if v.strip())


def _parse_int_grid(value: str | None, default: Iterable[int]) -> tuple[int, ...]:
    if not value:
        return tuple(int(v) for v in default)
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def _method_config_to_dict(cfg: MonitoringConfig) -> dict[str, Any]:
    return {
        "method": cfg.method,
        "alpha": cfg.alpha,
        "min_consecutive": cfg.min_consecutive,
        "variance_target": cfg.variance_target,
        "lags": cfg.lags,
        "validation_false_alarm_rate": cfg.validation_false_alarm_rate,
    }


def _evaluate_alarm(alarm_mask, score, fault_id: int) -> dict[str, Any]:
    alarm = np.asarray(alarm_mask, dtype=bool)
    split = min(TEP_TEST_FAULT_START, len(alarm))
    pre_far = float(np.mean(alarm[:split])) if split else 0.0
    score_arr = np.asarray(score, dtype=float)

    if int(fault_id) == 0:
        first_detection = int(np.flatnonzero(alarm)[0]) if np.any(alarm) else None
        return {
            "pre_fault_false_alarm_rate": pre_far,
            "post_fault_detection_rate": None,
            "first_detection_index": first_detection,
            "detection_delay_samples": None,
            "detection_delay_minutes": None,
            "max_score_post_fault": None,
        }

    post = alarm[split:]
    post_rate = float(np.mean(post)) if len(post) else 0.0
    post_indices = np.flatnonzero(post)
    first_detection = int(split + post_indices[0]) if post_indices.size else None
    delay = None if first_detection is None else int(first_detection - split)
    post_score = score_arr[split:]
    max_score = None if post_score.size == 0 else float(np.nanmax(post_score))
    return {
        "pre_fault_false_alarm_rate": pre_far,
        "post_fault_detection_rate": post_rate,
        "first_detection_index": first_detection,
        "detection_delay_samples": delay,
        "detection_delay_minutes": None if delay is None else delay * TEP_SAMPLE_PERIOD_MIN,
        "max_score_post_fault": max_score,
    }


def _candidate_channels(reference: np.ndarray, current: np.ndarray, alarm_mask, *, top_k: int) -> list[int]:
    standardized = standardize_against_normal(current, reference)
    shift = pre_post_shift_evidence(
        standardized["standardized_signal"],
        TEP_CHANNEL_NAMES,
        alarm_mask=alarm_mask,
    )
    name_to_idx = {name: i for i, name in enumerate(TEP_CHANNEL_NAMES)}
    ranked = [name_to_idx[name] for name in shift["ranked_variables"] if name in name_to_idx]
    return ranked[: min(max(2, int(top_k)), current.shape[1])]


def _summarize(rows: list[TEPFaultMethodResult], methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        group = [r for r in rows if r.method == method]
        faulty = [r for r in group if r.fault_id != 0]
        post_rates = [r.post_fault_detection_rate for r in faulty if r.post_fault_detection_rate is not None]
        delays = [r.detection_delay_samples for r in faulty if r.detection_delay_samples is not None]
        localization = [r for r in faulty if r.localization_hit is not None]
        summary[method] = {
            "mean_pre_fault_false_alarm_rate": float(np.mean([r.pre_fault_false_alarm_rate for r in group])) if group else None,
            "mean_post_fault_detection_rate": float(np.mean(post_rates)) if post_rates else None,
            "faults_with_any_post_fault_detection": int(sum((r.post_fault_detection_rate or 0.0) > 0 for r in faulty)),
            "fault_count": len(faulty),
            "mean_detection_delay_samples": float(np.mean(delays)) if delays else None,
            "mean_detection_delay_minutes": float(np.mean(delays) * TEP_SAMPLE_PERIOD_MIN) if delays else None,
            "proxy_localization_accuracy": float(np.mean([bool(r.localization_hit) for r in localization])) if localization else None,
            "proxy_localization_cases": len(localization),
        }
    if "pca" in summary and "dpca" in summary:
        pca_far = summary["pca"]["mean_pre_fault_false_alarm_rate"]
        dpca_far = summary["dpca"]["mean_pre_fault_false_alarm_rate"]
        pca_dr = summary["pca"]["mean_post_fault_detection_rate"]
        dpca_dr = summary["dpca"]["mean_post_fault_detection_rate"]
        summary["comparison"] = {
            "dpca_false_alarm_reduction": None if pca_far is None or dpca_far is None else float(pca_far - dpca_far),
            "dpca_detection_rate_delta": None if pca_dr is None or dpca_dr is None else float(dpca_dr - pca_dr),
        }
    return summary


def _table_rows(results: list[TEPFaultMethodResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        rows.append(
            {
                "method": r.method,
                "fault_id": r.fault_id,
                "fault_type": r.fault_type,
                "pre_fault_false_alarm_rate": round(r.pre_fault_false_alarm_rate, 6),
                "post_fault_detection_rate": None if r.post_fault_detection_rate is None else round(r.post_fault_detection_rate, 6),
                "detection_delay_samples": r.detection_delay_samples,
                "detection_delay_minutes": r.detection_delay_minutes,
                "predicted_root_cause": r.predicted_root_cause,
                "diagnostic_confidence": r.diagnostic_confidence,
                "localization_hit": r.localization_hit,
                "predicted_fault_label": r.predicted_fault_label,
            }
        )
    return rows


class TEPBenchmark:
    """Run calibrated PCA/DPCA benchmarks on canonical Braatz TEP files.

    Calibration uses held-out normal-operation data only, so false-alarm control is
    tuned without using fault labels. Detection is evaluated on the standard TEP
    test convention: first 160 observations normal, fault active from index 160.
    """

    def __init__(
        self,
        *,
        methods: Iterable[str] = ("pca", "dpca"),
        variance_target: float = 0.95,
        target_false_alarm_rate: float = 0.05,
        alpha_grid: Iterable[float] = (0.99, 0.995, 0.9975, 0.999, 0.9995),
        persistence_grid: Iterable[int] = (1, 2, 3, 5),
        lags_grid: Iterable[int] = (1, 2, 3),
        diagnostic_top_k: int = 8,
        maxlag: int = 1,
    ):
        self.methods = tuple(m.lower() for m in methods)
        self.variance_target = variance_target
        self.target_false_alarm_rate = target_false_alarm_rate
        self.alpha_grid = tuple(alpha_grid)
        self.persistence_grid = tuple(persistence_grid)
        self.lags_grid = tuple(lags_grid)
        self.diagnostic_top_k = diagnostic_top_k
        self.maxlag = maxlag

    def _calibrate(self, standardized_reference: np.ndarray) -> dict[str, dict[str, Any]]:
        out = {}
        for method in self.methods:
            calibrated = calibrate_monitoring_config(
                standardized_reference,
                method=method,
                variance_target=self.variance_target,
                target_false_alarm_rate=self.target_false_alarm_rate,
                alpha_grid=self.alpha_grid,
                persistence_grid=self.persistence_grid,
                lags_grid=self.lags_grid,
            )
            out[method] = {
                "config": calibrated["config"],
                "trials": calibrated["trials"],
            }
        return out

    def run(
        self,
        data_dir: str | Path,
        *,
        fault_ids: Iterable[int] = range(0, 22),
        diagnostic_faults: Iterable[int] = tuple(TEP_LOCALIZATION_PROXIES),
        diagnostic_method: str = "dpca",
        output_dir: str | Path | None = None,
        write_plots: bool = True,
    ) -> TEPBenchmarkResult:
        data_dir = Path(data_dir)
        reference = load_tep_reference(data_dir)
        standardized_reference = standardize_against_normal(reference, reference)["standardized_reference"]
        calibration = self._calibrate(standardized_reference)
        diagnostic_faults = set(int(x) for x in diagnostic_faults)
        diagnostic_method = diagnostic_method.lower()
        if diagnostic_method not in self.methods:
            diagnostic_method = self.methods[0]

        results: list[TEPFaultMethodResult] = []
        for fault_id in [int(x) for x in fault_ids]:
            current = load_tep_dat(data_dir / f"d{fault_id:02d}_te.dat")
            standardized = standardize_against_normal(current, reference)
            method_outputs: dict[str, dict[str, Any]] = {}

            for method in self.methods:
                cfg = calibration[method]["config"]
                monitored = run_monitoring_method(
                    standardized["standardized_signal"],
                    standardized["standardized_reference"],
                    cfg,
                )
                method_outputs[method] = monitored
                eval_row = _evaluate_alarm(monitored["alarm_mask"], monitored["combined_score"], fault_id)

                selected_names: list[str] = []
                predicted_root = None
                confidence = None
                localization_hit = None
                predicted_label = None
                abstain_reason = None
                proxies = list(TEP_LOCALIZATION_PROXIES.get(fault_id, ()))

                if method == diagnostic_method and fault_id in diagnostic_faults and fault_id != 0:
                    candidate_idx = _candidate_channels(
                        reference,
                        current,
                        monitored["alarm_mask"],
                        top_k=self.diagnostic_top_k,
                    )
                    selected_names = [TEP_CHANNEL_NAMES[i] for i in candidate_idx]
                    subset_current = current[:, candidate_idx]
                    subset_reference = reference[:, candidate_idx]
                    catalog = {
                        proxy: f"IDV({fault_id}): {TEP_FAULTS[fault_id]['description']}"
                        for proxy in proxies
                        if proxy in selected_names
                    }
                    pipeline = ProcessDiagnosticPipeline(
                        variance_target=self.variance_target,
                        control_alpha=cfg.alpha,
                        maxlag=self.maxlag,
                        onset_persistence=max(2, cfg.min_consecutive),
                    )
                    diagnosis = pipeline.run(
                        subset_current,
                        subset_reference,
                        selected_names,
                        fault_catalog=catalog or None,
                    )
                    predicted_root = diagnosis.root_cause
                    confidence = diagnosis.confidence
                    predicted_label = diagnosis.fault_label
                    abstain_reason = diagnosis.abstain_reason
                    localization_hit = predicted_root in proxies if predicted_root and proxies else False

                results.append(
                    TEPFaultMethodResult(
                        method=method,
                        fault_id=fault_id,
                        description=TEP_FAULTS[fault_id]["description"],
                        fault_type=TEP_FAULTS[fault_id]["type"],
                        n_samples=int(current.shape[0]),
                        calibration_alpha=float(cfg.alpha),
                        calibration_min_consecutive=int(cfg.min_consecutive),
                        calibration_lags=int(cfg.lags),
                        validation_false_alarm_rate=cfg.validation_false_alarm_rate,
                        selected_diagnostic_channels=selected_names,
                        predicted_root_cause=predicted_root,
                        diagnostic_confidence=confidence,
                        localization_proxy_targets=proxies,
                        localization_hit=localization_hit,
                        predicted_fault_label=predicted_label,
                        abstain_reason=abstain_reason,
                        **eval_row,
                    )
                )

        summary = _summarize(results, list(self.methods))
        artifacts: dict[str, str] = {}
        if output_dir is not None:
            out = Path(output_dir)
            rows = _table_rows(results)
            artifacts["fault_table_csv"] = str(write_fault_table(rows, out / "tep_fault_table.csv"))
            artifacts["summary_md"] = str(write_markdown_summary(summary, rows, out / "tep_summary.md"))
            if write_plots:
                plot_paths = write_detection_plots(rows, out / "plots")
                artifacts["plots"] = ",".join(str(p) for p in plot_paths)

        return TEPBenchmarkResult(
            source="Braatz Tennessee Eastman Process archive",
            reference_shape=tuple(int(x) for x in reference.shape),
            fault_start_index=TEP_TEST_FAULT_START,
            channel_count=len(TEP_CHANNEL_NAMES),
            methods=list(self.methods),
            calibration={k: {"config": _method_config_to_dict(v["config"]), "trials": v["trials"]} for k, v in calibration.items()},
            faults=results,
            summary=summary,
            artifacts=artifacts,
        )


def main():
    parser = argparse.ArgumentParser(description="Run calibrated Tennessee Eastman Process benchmark")
    parser.add_argument("--data-dir", default="data/tep_braatz")
    parser.add_argument("--download", action="store_true", help="download required public Braatz files first")
    parser.add_argument("--faults", nargs="*", help="fault IDs, e.g. --faults 0 1 4 6 or --faults 0,1,4,6")
    parser.add_argument("--diagnostic-faults", nargs="*", help="fault IDs for expensive root-cause analysis")
    parser.add_argument("--methods", nargs="*", default=["pca", "dpca"], help="monitoring methods: pca dpca")
    parser.add_argument("--diagnostic-method", default="dpca")
    parser.add_argument("--diagnostic-top-k", type=int, default=8)
    parser.add_argument("--maxlag", type=int, default=1)
    parser.add_argument("--variance-target", type=float, default=0.95)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--alpha-grid", default="0.99,0.995,0.9975,0.999,0.9995")
    parser.add_argument("--persistence-grid", default="1,2,3,5")
    parser.add_argument("--lags-grid", default="1,2,3")
    parser.add_argument("--output", default="outputs/tep_benchmark.json")
    parser.add_argument("--artifact-dir", default="outputs/tep_benchmark")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    fault_ids = _parse_ids(args.faults, range(0, 22))
    diagnostic_faults = _parse_ids(args.diagnostic_faults, TEP_LOCALIZATION_PROXIES.keys())
    data_dir = Path(args.data_dir)

    if args.download:
        download_braatz_tep(data_dir, fault_ids=[0], splits=["train"])
        download_braatz_tep(data_dir, fault_ids=fault_ids, splits=["test"])

    benchmark = TEPBenchmark(
        methods=args.methods,
        variance_target=args.variance_target,
        target_false_alarm_rate=args.target_far,
        alpha_grid=_parse_float_grid(args.alpha_grid, (0.99, 0.995, 0.9975, 0.999, 0.9995)),
        persistence_grid=_parse_int_grid(args.persistence_grid, (1, 2, 3, 5)),
        lags_grid=_parse_int_grid(args.lags_grid, (1, 2, 3)),
        diagnostic_top_k=args.diagnostic_top_k,
        maxlag=args.maxlag,
    )
    result = benchmark.run(
        data_dir,
        fault_ids=fault_ids,
        diagnostic_faults=diagnostic_faults,
        diagnostic_method=args.diagnostic_method,
        output_dir=args.artifact_dir,
        write_plots=not args.no_plots,
    )

    output = Path(args.output)
    write_json_report(result, output)
    print(json.dumps(result.summary, indent=2))
    print("Calibration:")
    print(json.dumps({k: v["config"] for k, v in result.calibration.items()}, indent=2, default=str))
    print(f"saved: {output}")
    for key, value in result.artifacts.items():
        print(f"artifact {key}: {value}")


if __name__ == "__main__":
    main()
