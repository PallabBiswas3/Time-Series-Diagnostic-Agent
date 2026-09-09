from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _normalize_scores(score_map: dict[str, float]) -> dict[str, float]:
    values = [max(0.0, float(v)) for v in score_map.values()]
    scale = max(values) if values else 0.0
    if scale <= 1e-12:
        return {str(k): 0.0 for k in score_map}
    return {str(k): max(0.0, float(v)) / scale for k, v in score_map.items()}


def _score_from_rank(ranked_variables: Iterable[str], decay: float = 0.82) -> dict[str, float]:
    return {str(name): float(decay**i) for i, name in enumerate(ranked_variables)}


def _catalog_records(fault_catalog: Any) -> list[dict[str, Any]]:
    if fault_catalog is None:
        return []
    if isinstance(fault_catalog, dict) and "faults" in fault_catalog:
        return [r for r in fault_catalog["faults"] if isinstance(r, dict)]
    if isinstance(fault_catalog, list):
        return [r for r in fault_catalog if isinstance(r, dict)]
    return []


def _expected_roots(record: dict[str, Any]) -> list[str]:
    roots = record.get("expected_roots")
    if roots:
        return [str(x) for x in roots]
    root = record.get("root_cause") or record.get("root_variable")
    return [str(root)] if root else []


def _choose_best_root(roots: list[str], score_map: dict[str, float]) -> str | None:
    if not roots:
        return None
    primary = roots[0]
    primary_score = score_map.get(primary, 0.0)
    if primary_score > 0.05:
        return primary
    return max(roots, key=lambda root: score_map.get(root, 0.0))


def rank_tep_fault_catalog(
    ranked_variables: Iterable[str],
    fault_catalog: Any,
    *,
    variable_scores: dict[str, float] | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    """Match observed abnormal variables to the TEP fault catalog.

    The scoring is transparent and conservative: root priors are only one term;
    affected-variable overlap must also support the fault. Unknown IDV(16-20)-type
    records with no expected roots are skipped because they cannot provide a
    physical root-cause prior.
    """
    ranked = [str(v) for v in ranked_variables]
    rank_scores = _score_from_rank(ranked)
    score_map = _normalize_scores({**rank_scores, **(variable_scores or {})})
    top_set = set(ranked[: max(1, int(top_k))])

    rows: list[dict[str, Any]] = []
    for record in _catalog_records(fault_catalog):
        roots = _expected_roots(record)
        affected = [str(x) for x in record.get("affected_variables", [])]
        if not roots and not affected:
            continue

        root_scores = [score_map.get(root, 0.0) for root in roots]
        root_score = max(root_scores) if root_scores else 0.0
        best_root = _choose_best_root(roots, score_map)

        affected_scores = [score_map.get(var, 0.0) for var in affected]
        affected_mean = float(np.mean(affected_scores)) if affected_scores else 0.0
        affected_hits = sorted(top_set & set(affected))
        affected_overlap = len(affected_hits) / max(1, min(len(affected), len(top_set)))

        # Fault-catalog match: root is important, but overlap in the expected
        # propagation neighbourhood prevents single-variable leakage.
        catalog_score = 0.45 * root_score + 0.35 * affected_overlap + 0.20 * affected_mean
        rows.append(
            {
                "fault_id": record.get("fault_id"),
                "fault_label": record.get("fault_label") or record.get("label"),
                "description": record.get("description"),
                "fault_type": record.get("type"),
                "subsystem": record.get("subsystem"),
                "score": float(catalog_score),
                "best_root": best_root,
                "expected_roots": roots,
                "affected_hits": affected_hits,
                "root_score": float(root_score),
                "affected_overlap": float(affected_overlap),
                "affected_mean_score": float(affected_mean),
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"fault_catalog_ranking": rows, "best_match": rows[0] if rows else None}


def knowledge_guided_root_cause_decision(
    generic_ranking: list[dict[str, Any]],
    fault_catalog: Any,
    *,
    shift_scores: dict[str, float] | None = None,
    contribution_scores: dict[str, float] | None = None,
    onset_order: Iterable[str] | None = None,
    threshold: float = 0.18,
) -> dict[str, Any]:
    """Fuse generic root-cause ranking with a TEP catalog/topology prior.

    This implements the research idea used by process-fault papers: do not trust
    a pure contribution plot or pure data-driven edge graph alone; regularize it
    with process knowledge and known fault mechanisms.
    """
    generic_score = {str(r["variable"]): float(r.get("score", 0.0)) for r in (generic_ranking or []) if "variable" in r}
    score_map: dict[str, float] = {}
    for source, weight in ((generic_score, 0.35), (shift_scores or {}, 0.35), (contribution_scores or {}, 0.20), (_score_from_rank(onset_order or []), 0.10)):
        normalized = _normalize_scores({str(k): float(v) for k, v in source.items()})
        for key, value in normalized.items():
            score_map[key] = score_map.get(key, 0.0) + weight * value

    ranked_variables = [k for k, _ in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)]
    catalog = rank_tep_fault_catalog(ranked_variables, fault_catalog, variable_scores=score_map)
    best_catalog = catalog["best_match"]

    generic_best = generic_ranking[0] if generic_ranking else None
    generic_root = str(generic_best["variable"]) if generic_best else None
    generic_conf = float(generic_best.get("score", 0.0)) if generic_best else 0.0

    if best_catalog and best_catalog["score"] >= threshold and best_catalog.get("best_root"):
        root = best_catalog["best_root"]
        confidence = float(np.clip(0.55 * best_catalog["score"] + 0.45 * score_map.get(root, 0.0), 0.0, 1.0))
        return {
            "root_cause": root,
            "fault_label": best_catalog["fault_label"],
            "fault_id": best_catalog["fault_id"],
            "confidence": confidence,
            "decision_source": "tep_fault_catalog",
            "catalog_match": best_catalog,
            "catalog_ranking": catalog["fault_catalog_ranking"],
            "ranked_variables": ranked_variables,
        }

    return {
        "root_cause": generic_root,
        "fault_label": None if generic_root is None else f"Process anomaly rooted at {generic_root}",
        "fault_id": None,
        "confidence": float(np.clip(generic_conf, 0.0, 1.0)),
        "decision_source": "generic_ranking",
        "catalog_match": best_catalog,
        "catalog_ranking": catalog["fault_catalog_ranking"],
        "ranked_variables": ranked_variables,
    }
