from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FAILURE_CATEGORIES = (
    "DOWNSTREAM_OVER_WEIGHTING",
    "EARLY_NOISE_ONSET",
    "CATALOG_AMBIGUITY",
    "WEAK_SIGNAL",
)


def _candidate_map(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        str(row["variable"]): {
            key: float(row.get(key, 0.0) or 0.0)
            for key in (
                "contribution_score",
                "pre_post_shift_score",
                "onset_earliness_score",
                "topology_upstreamness_score",
                "fault_type_agreement_score",
                "catalog_root_prior_score",
            )
        }
        for row in record.get("candidates", [])
        if "variable" in row
    }


def _score_with_weights(features: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(float(weights.get(name, 0.0)) * float(value) for name, value in features.items()))


def _best_true_root(expected_roots: Iterable[str], feature_map, weights):
    choices = []
    for root in expected_roots:
        if root in feature_map:
            choices.append((root, _score_with_weights(feature_map[root], weights)))
    if not choices:
        return None, None
    choices.sort(key=lambda item: item[1], reverse=True)
    return choices[0]


def _failure_category(
    true_features: dict[str, float],
    wrong_features: dict[str, float],
    root_support_gap: float,
) -> str:
    if true_features.get("pre_post_shift_score", 0.0) < 0.20:
        return "WEAK_SIGNAL"
    if (
        wrong_features.get("onset_earliness_score", 0.0)
        - true_features.get("onset_earliness_score", 0.0)
        >= 0.30
    ):
        return "EARLY_NOISE_ONSET"
    if (
        abs(
            wrong_features.get("catalog_root_prior_score", 0.0)
            - true_features.get("catalog_root_prior_score", 0.0)
        ) <= 0.10
        and abs(
            wrong_features.get("fault_type_agreement_score", 0.0)
            - true_features.get("fault_type_agreement_score", 0.0)
        ) <= 0.15
    ):
        return "CATALOG_AMBIGUITY"
    if (
        wrong_features.get("contribution_score", 0.0)
        + wrong_features.get("pre_post_shift_score", 0.0)
        > true_features.get("contribution_score", 0.0)
        + true_features.get("pre_post_shift_score", 0.0)
        and true_features.get("topology_upstreamness_score", 0.0)
        >= wrong_features.get("topology_upstreamness_score", 0.0)
    ):
        return "DOWNSTREAM_OVER_WEIGHTING"
    if root_support_gap < 0.08:
        return "CATALOG_AMBIGUITY"
    return "DOWNSTREAM_OVER_WEIGHTING"


def analyze_root_cause_errors(
    calibration_records: Iterable[dict[str, Any]],
    held_out_predictions: Iterable[dict[str, Any]],
    weights: dict[str, float],
) -> dict[str, Any]:
    records = {int(row["fault_id"]): row for row in calibration_records}
    predictions = {int(row["fault_id"]): row for row in held_out_predictions}
    errors = []

    for fault_id, record in sorted(records.items()):
        expected = [str(x) for x in record.get("expected_roots", [])]
        if not expected or fault_id not in predictions:
            continue
        prediction = predictions[fault_id]
        raw_top = prediction.get("raw_top_root")
        feature_map = _candidate_map(record)
        if not raw_top or raw_top not in feature_map:
            continue

        ranked = prediction.get("ranking", [])
        true_candidates = [row for row in ranked if row.get("variable") in expected]
        true_candidates.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        true_row = true_candidates[0] if true_candidates else None
        if true_row is None:
            true_root, true_score = _best_true_root(expected, feature_map, weights)
            if true_root is None:
                continue
            true_rank = None
        else:
            true_root = str(true_row["variable"])
            true_score = float(true_row.get("score", 0.0))
            true_rank = next(
                (i + 1 for i, row in enumerate(ranked) if row.get("variable") == true_root),
                None,
            )

        wrong = str(raw_top)
        wrong_score = float(ranked[0].get("score", 0.0)) if ranked else _score_with_weights(feature_map[wrong], weights)
        if wrong in expected:
            continue

        true_features = feature_map.get(true_root, {})
        wrong_features = feature_map.get(wrong, {})
        gap = float(wrong_score - float(true_score))
        category = _failure_category(true_features, wrong_features, gap)
        errors.append(
            {
                "fault_id": fault_id,
                "true_root": true_root,
                "true_root_rank": true_rank,
                "top_wrong_variable": wrong,
                "root_support_gap": gap,
                "primary_failure_cause": category,
                "true_root_score": float(true_score),
                "top_wrong_score": wrong_score,
                "true_root_components": true_features,
                "top_wrong_components": wrong_features,
                "abstained": bool(prediction.get("abstained")),
                "score_margin": float(prediction.get("score_margin", 0.0) or 0.0),
            }
        )

    counts = Counter(row["primary_failure_cause"] for row in errors)
    return {
        "error_count": len(errors),
        "category_counts": {name: int(counts.get(name, 0)) for name in FAILURE_CATEGORIES},
        "errors": errors,
    }


def write_error_analysis_json(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def write_error_analysis_markdown(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TEP Root-Cause Error Analysis",
        "",
        "## Error taxonomy",
        "",
    ]
    for category in FAILURE_CATEGORIES:
        lines.append(f"- **{category}**: {report.get('category_counts', {}).get(category, 0)}")
    lines.extend(
        [
            "",
            "## Misdiagnosed faults",
            "",
            "| Fault | True root | True rank | Top wrong variable | Support gap | Primary failure cause |",
            "| ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in report.get("errors", []):
        rank = "-" if row.get("true_root_rank") is None else str(row["true_root_rank"])
        lines.append(
            f"| {row['fault_id']} | {row['true_root']} | {rank} | {row['top_wrong_variable']} | "
            f"{row['root_support_gap']:.4f} | {row['primary_failure_cause']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
