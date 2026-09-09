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
from ..tools import contribution_analysis, pca_monitoring, standardize_against_normal


@dataclass
class TEPFaultResult:
    fault_id: int
    description: str
    fault_type: str
    n_samples: int
    pre_fault_false_alarm_rate: float
    post_fault_detection_rate: float | None
    first_detection_index: int | None
    detection_delay_samples: int | None
    detection_delay_minutes: float | None
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
    faults: list[TEPFaultResult]
    summary: dict[str, float | int | None]


def _pca_alarm(reference: np.ndarray, current: np.ndarray, *, variance_target=0.95, alpha=0.99):
    standardized = standardize_against_normal(current, reference)
    monitored = pca_monitoring(
        standardized["standardized_signal"],
        standardized["standardized_reference"],
        variance_target=variance_target,
        alpha=alpha,
    )
    return standardized, monitored


def _candidate_channels(reference: np.ndarray, current: np.ndarray, alarm_mask, *, top_k: int) -> list[int]:
    standardized, monitored = _pca_alarm(reference, current)
    contrib = contribution_analysis(
        standardized["standardized_signal"],
        monitored["pca_state"],
        alarm_mask,
    )
    ranked = [int(i) for i in np.asarray(contrib["ranked_indices"]).ravel()]
    return ranked[: min(max(2, int(top_k)), current.shape[1])]


class TEPBenchmark:
    """Benchmark the deterministic process pipeline on canonical Braatz TEP files.

    Detection is evaluated on every selected fault using the standard convention:
    the first 160 test observations are normal and the fault is active from sample
    161 (zero-based index 160). Root-cause/localization is evaluated separately
    because the public TEP archive does not provide canonical per-fault measured-
    variable root-cause labels. For a conservative subset, engineering proxy targets
    are reported explicitly and scored as proxy localization rather than ground truth.
    """

    def __init__(
        self,
        *,
        variance_target: float = 0.95,
        control_alpha: float = 0.99,
        diagnostic_top_k: int = 8,
        maxlag: int = 1,
    ):
        self.variance_target = variance_target
        self.control_alpha = control_alpha
        self.diagnostic_top_k = diagnostic_top_k
        self.maxlag = maxlag

    def run(
        self,
        data_dir: str | Path,
        *,
        fault_ids: Iterable[int] = range(0, 22),
        diagnostic_faults: Iterable[int] = tuple(TEP_LOCALIZATION_PROXIES),
    ) -> TEPBenchmarkResult:
        data_dir = Path(data_dir)
        reference = load_tep_reference(data_dir)
        diagnostic_faults = set(int(x) for x in diagnostic_faults)
        results: list[TEPFaultResult] = []

        for fault_id in [int(x) for x in fault_ids]:
            test_path = data_dir / f"d{fault_id:02d}_te.dat"
            current = load_tep_dat(test_path)
            standardized, monitored = _pca_alarm(
                reference,
                current,
                variance_target=self.variance_target,
                alpha=self.control_alpha,
            )
            alarm = np.asarray(monitored["alarm_mask"], dtype=bool)
            split = min(TEP_TEST_FAULT_START, len(alarm))
            pre_far = float(np.mean(alarm[:split])) if split else 0.0

            if fault_id == 0:
                post_rate = None
                first_detection = int(np.flatnonzero(alarm)[0]) if np.any(alarm) else None
                delay = None
            else:
                post = alarm[split:]
                post_rate = float(np.mean(post)) if len(post) else 0.0
                post_indices = np.flatnonzero(post)
                first_detection = int(split + post_indices[0]) if post_indices.size else None
                delay = None if first_detection is None else int(first_detection - split)

            selected_names: list[str] = []
            predicted_root = None
            confidence = None
            localization_hit = None
            predicted_label = None
            abstain_reason = None
            proxies = list(TEP_LOCALIZATION_PROXIES.get(fault_id, ()))

            if fault_id in diagnostic_faults and fault_id != 0:
                candidate_idx = _candidate_channels(
                    reference,
                    current,
                    alarm,
                    top_k=self.diagnostic_top_k,
                )
                selected_names = [TEP_CHANNEL_NAMES[i] for i in candidate_idx]
                subset_current = current[:, candidate_idx]
                subset_reference = reference[:, candidate_idx]

                # Only provide the benchmark fault's conservative engineering proxies
                # to avoid giving the diagnostic pipeline the true class itself.
                catalog = {
                    proxy: f"IDV({fault_id}): {TEP_FAULTS[fault_id]['description']}"
                    for proxy in proxies
                    if proxy in selected_names
                }

                pipeline = ProcessDiagnosticPipeline(
                    variance_target=self.variance_target,
                    control_alpha=self.control_alpha,
                    maxlag=self.maxlag,
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
                localization_hit = (
                    predicted_root in proxies if predicted_root is not None and proxies else False
                )

            results.append(
                TEPFaultResult(
                    fault_id=fault_id,
                    description=TEP_FAULTS[fault_id]["description"],
                    fault_type=TEP_FAULTS[fault_id]["type"],
                    n_samples=int(current.shape[0]),
                    pre_fault_false_alarm_rate=pre_far,
                    post_fault_detection_rate=post_rate,
                    first_detection_index=first_detection,
                    detection_delay_samples=delay,
                    detection_delay_minutes=None if delay is None else delay * TEP_SAMPLE_PERIOD_MIN,
                    selected_diagnostic_channels=selected_names,
                    predicted_root_cause=predicted_root,
                    diagnostic_confidence=confidence,
                    localization_proxy_targets=proxies,
                    localization_hit=localization_hit,
                    predicted_fault_label=predicted_label,
                    abstain_reason=abstain_reason,
                )
            )

        faulty = [r for r in results if r.fault_id != 0]
        localization = [r for r in faulty if r.localization_hit is not None]
        detected_delays = [r.detection_delay_samples for r in faulty if r.detection_delay_samples is not None]
        summary = {
            "mean_pre_fault_false_alarm_rate": float(np.mean([r.pre_fault_false_alarm_rate for r in results])),
            "mean_post_fault_detection_rate": float(np.mean([r.post_fault_detection_rate for r in faulty if r.post_fault_detection_rate is not None])) if faulty else None,
            "faults_with_any_post_fault_detection": int(sum((r.post_fault_detection_rate or 0.0) > 0 for r in faulty)),
            "fault_count": len(faulty),
            "mean_detection_delay_samples": float(np.mean(detected_delays)) if detected_delays else None,
            "mean_detection_delay_minutes": float(np.mean(detected_delays) * TEP_SAMPLE_PERIOD_MIN) if detected_delays else None,
            "proxy_localization_accuracy": float(np.mean([bool(r.localization_hit) for r in localization])) if localization else None,
            "proxy_localization_cases": len(localization),
        }

        return TEPBenchmarkResult(
            source="Braatz Tennessee Eastman Process archive",
            reference_shape=tuple(int(x) for x in reference.shape),
            fault_start_index=TEP_TEST_FAULT_START,
            channel_count=len(TEP_CHANNEL_NAMES),
            faults=results,
            summary=summary,
        )


def _parse_ids(values: list[str] | None, default):
    if not values:
        return list(default)
    out = []
    for value in values:
        for token in value.split(","):
            out.append(int(token))
    return out


def main():
    parser = argparse.ArgumentParser(description="Run the canonical Tennessee Eastman Process benchmark")
    parser.add_argument("--data-dir", default="data/tep_braatz")
    parser.add_argument("--download", action="store_true", help="download required public Braatz files first")
    parser.add_argument("--faults", nargs="*", help="fault IDs, e.g. --faults 0 1 4 6 or --faults 0,1,4,6")
    parser.add_argument("--diagnostic-faults", nargs="*", help="fault IDs for expensive root-cause analysis")
    parser.add_argument("--diagnostic-top-k", type=int, default=8)
    parser.add_argument("--maxlag", type=int, default=1)
    parser.add_argument("--output", default="outputs/tep_benchmark.json")
    args = parser.parse_args()

    fault_ids = _parse_ids(args.faults, range(0, 22))
    diagnostic_faults = _parse_ids(args.diagnostic_faults, TEP_LOCALIZATION_PROXIES.keys())
    data_dir = Path(args.data_dir)

    if args.download:
        # d00 training reference plus selected test files are sufficient.
        download_braatz_tep(data_dir, fault_ids=[0], splits=["train"])
        download_braatz_tep(data_dir, fault_ids=fault_ids, splits=["test"])

    benchmark = TEPBenchmark(diagnostic_top_k=args.diagnostic_top_k, maxlag=args.maxlag)
    result = benchmark.run(data_dir, fault_ids=fault_ids, diagnostic_faults=diagnostic_faults)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    print(json.dumps(result.summary, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
