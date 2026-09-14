from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..datasets.transformer_sgah import SGAH_CHANNEL_NAMES, SGAH_CLASSES, SgahEvent, load_sgah_events
from ..domains.runners import TransformerDiagnosticPipeline
from ..tools.transformer_rules import fit_transformer_rule_reference, transformer_electrical_features, transformer_rule_diagnosis


@dataclass
class SgahCase:
    class_id: int
    label: str
    event_id: int
    true_transformer_fault: bool
    predicted_transformer_fault: bool
    decision: str
    abstained: bool
    confidence: float
    verification_status: str | None
    runtime_seconds: float


def _split(events: list[SgahEvent]) -> tuple[list[SgahEvent], list[SgahEvent], list[SgahEvent]]:
    n = len(events)
    train_end = max(1, int(np.floor(0.60 * n)))
    dev_end = max(train_end + 1, int(np.floor(0.80 * n))) if n >= 3 else n
    dev_end = min(dev_end, n)
    return events[:train_end], events[train_end:dev_end], events[dev_end:]


def _metrics_from_predictions(events: list[SgahEvent], predicted: np.ndarray) -> dict:
    y = np.asarray([event.class_id == 4 for event in events], dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(predicted & y)); fn = int(np.sum((~predicted) & y))
    fp = int(np.sum(predicted & (~y))); tn = int(np.sum((~predicted) & (~y)))
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    specificity = 0.0 if tn + fp == 0 else tn / (tn + fp)
    normal = np.asarray([event.class_id == 5 for event in events], dtype=bool)
    competing = np.asarray([event.class_id in (1, 2, 3) for event in events], dtype=bool)
    return {
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "normal_false_positive_rate": float(np.mean(predicted[normal])) if np.any(normal) else 0.0,
        "competing_fault_false_positive_rate": float(np.mean(predicted[competing])) if np.any(competing) else 0.0,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def _calibrate_rule_threshold(reference, dev_events: list[SgahEvent]) -> tuple[float, dict]:
    candidates = np.arange(2.0, 6.01, 0.25)
    rows = []
    for threshold in candidates:
        predicted = np.asarray([
            transformer_rule_diagnosis(event.signal_matrix, reference, threshold=float(threshold))["transformer_fault"]
            for event in dev_events
        ], dtype=bool)
        row = {"threshold": float(threshold), **_metrics_from_predictions(dev_events, predicted)}
        rows.append(row)
    feasible = [
        row for row in rows
        if row["recall"] >= 0.60
        and row["normal_false_positive_rate"] <= 0.15
        and row["competing_fault_false_positive_rate"] <= 0.30
    ]
    if feasible:
        best = max(feasible, key=lambda row: (row["balanced_accuracy"], row["recall"], row["threshold"]))
    else:
        def violation(row):
            return max(0.60-row["recall"],0.0)+max(row["normal_false_positive_rate"]-0.15,0.0)+max(row["competing_fault_false_positive_rate"]-0.30,0.0)
        best = max(rows, key=lambda row: (-violation(row), row["balanced_accuracy"], row["threshold"]))
    return float(best["threshold"]), best


def _binary_metrics(rows: list[SgahCase]) -> dict:
    tp = sum(r.true_transformer_fault and r.predicted_transformer_fault for r in rows)
    fn = sum(r.true_transformer_fault and not r.predicted_transformer_fault for r in rows)
    fp = sum((not r.true_transformer_fault) and r.predicted_transformer_fault for r in rows)
    tn = sum((not r.true_transformer_fault) and not r.predicted_transformer_fault for r in rows)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    specificity = None if tn + fp == 0 else tn / (tn + fp)
    precision = None if tp + fp == 0 else tp / (tp + fp)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    balanced = None if recall is None or specificity is None else 0.5 * (recall + specificity)
    supported = sum(r.verification_status == "SUPPORTED" for r in rows)
    return {
        "case_count": len(rows), "positive_count": tp + fn, "negative_count": tn + fp,
        "true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn,
        "recall": recall, "specificity": specificity, "precision": precision, "f1": f1,
        "balanced_accuracy": balanced, "coverage": 1.0 if rows else None, "abstention_rate": 0.0 if rows else None,
        "verification_support_rate": None if not rows else supported / len(rows),
        "mean_runtime_seconds": None if not rows else float(np.mean([r.runtime_seconds for r in rows])),
    }


def run_sgah_transformer_benchmark(data_dir: str | Path, *, output_dir: str | Path | None = None) -> dict:
    root = Path(data_dir)
    train_by_class: dict[int, list[SgahEvent]] = {}
    dev_events: list[SgahEvent] = []
    test_events: list[SgahEvent] = []
    manifest = []
    for class_id, label in SGAH_CLASSES.items():
        events = load_sgah_events(root / f"{class_id}-data.csv", class_id)
        train, dev, test = _split(events)
        if not train or not dev or not test:
            raise ValueError(f"SGAH class {class_id} does not have enough whole events for frozen split")
        train_by_class[class_id] = train
        dev_events.extend(dev)
        test_events.extend(test)
        manifest.append({"class_id": class_id, "label": label, "event_count": len(events), "train_count": len(train), "dev_count": len(dev), "test_count": len(test)})

    # Physics baseline learns only a robust envelope of normal electrical operation.
    # No fault classifier is fit. Rules remain explicit and auditable.
    normal_reference_rows = [transformer_electrical_features(event.signal_matrix) for event in train_by_class[5]]
    reference = fit_transformer_rule_reference(normal_reference_rows)
    rule_threshold, dev_calibration = _calibrate_rule_threshold(reference, dev_events)

    cases: list[SgahCase] = []
    failures: list[dict] = []
    pipeline = TransformerDiagnosticPipeline()
    for event in test_events:
        try:
            started = perf_counter()
            result = pipeline.run(
                event.signal_matrix,
                sampling_rate_hz=1.0,
                sensor_positions=list(SGAH_CHANNEL_NAMES),
                transformer_rule_reference=reference,
                transformer_rule_threshold=rule_threshold,
                allow_ml_fallback=False,
            )
            runtime = perf_counter() - started
            predicted = bool(result.decision == "diagnose" and any(h.label == "main_transformer_fault" for h in result.hypotheses))
            verification_status = str(result.verification[0].status) if result.verification else None
            cases.append(SgahCase(event.class_id,event.label,event.event_id,event.class_id==4,predicted,result.decision,bool(result.abstained),float(result.confidence),verification_status,float(runtime)))
        except Exception as exc:
            failures.append({"class_id": event.class_id, "event_id": event.event_id, "error": f"{type(exc).__name__}: {exc}"})

    normal_rows = [row for row in cases if row.class_id == 5]
    competing_rows = [row for row in cases if row.class_id in (1,2,3)]
    summary = _binary_metrics(cases)
    summary.update({
        "normal_false_positive_rate": None if not normal_rows else sum(row.predicted_transformer_fault for row in normal_rows)/len(normal_rows),
        "competing_fault_false_positive_rate": None if not competing_rows else sum(row.predicted_transformer_fault for row in competing_rows)/len(competing_rows),
    })
    by_class = {str(class_id): {"label": SGAH_CLASSES[class_id], **_binary_metrics([row for row in cases if row.class_id == class_id])} for class_id in SGAH_CLASSES}

    payload = {
        "source": "smartlab-hfut/SGAH-datasets (State Grid Corporation of China)",
        "protocol": {
            "upstream_commit": "bbe1020e3fade83f7861657bb3eaea41c25ec0c9",
            "samples_per_event": 100,
            "split": "Per class, contiguous whole-event 60% train / 20% development / 20% frozen test; no row-level splitting.",
            "positive_class": "main_transformer_fault (class 4)",
            "negative_controls": "classes 1,2,3 competing grid faults plus class 5 normal",
            "primary_method": "deterministic transformer protection rules; no ML classifier",
            "normal_reference": "Robust median/MAD envelope fitted only from class-5 normal training events.",
            "rule_features": [
                "three-phase voltage/current RMS", "positive/negative/zero sequence ratios", "phase imbalance",
                "voltage/current transient ratios", "voltage/current THD", "apparent impedance",
                "impedance phase spread", "power phase imbalance"
            ],
            "rule_logic": "Transformer candidate requires a strong disturbance plus internal impedance/harmonic/transient evidence; dominant sequence/asymmetry evidence is treated as an external-grid-fault signature.",
            "threshold_calibration": "Single robust-sigma rule threshold selected on development events only against the frozen target gates; test labels excluded.",
            "rule_threshold": rule_threshold,
            "ml_fallback": False,
        },
        "manifest": manifest,
        "development": {"rule_threshold": rule_threshold, "dev_at_calibrated_threshold": dev_calibration},
        "normal_reference": {"center": reference.center, "scale": reference.scale},
        "summary": summary,
        "by_class": by_class,
        "cases": [asdict(row) for row in cases],
        "failures": failures,
    }
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "transformer_sgah_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if cases:
            with (out / "transformer_sgah_cases.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(cases[0]).keys())); writer.writeheader(); writer.writerows(asdict(row) for row in cases)
    return payload
