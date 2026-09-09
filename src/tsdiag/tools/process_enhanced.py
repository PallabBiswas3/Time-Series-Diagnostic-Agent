from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

import numpy as np


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def _normalize_edge(edge: Any) -> dict[str, Any]:
    if isinstance(edge, dict):
        cause = edge.get("cause") or edge.get("source") or edge.get("from")
        effect = edge.get("effect") or edge.get("target") or edge.get("to")
        if cause is None or effect is None:
            raise ValueError("edge dictionaries must define cause/source/from and effect/target/to")
        out = dict(edge)
        out["cause"] = str(cause)
        out["effect"] = str(effect)
        return out
    if isinstance(edge, (tuple, list)) and len(edge) >= 2:
        return {"cause": str(edge[0]), "effect": str(edge[1])}
    raise TypeError("edges must be mappings or (cause, effect) pairs")


def _extract_edges(filtered_causal_graph: Any) -> list[dict[str, Any]]:
    if not filtered_causal_graph:
        return []
    raw = filtered_causal_graph.get("edges", []) if isinstance(filtered_causal_graph, dict) else filtered_causal_graph
    return [_normalize_edge(edge) for edge in raw]


def pre_post_shift_evidence(
    signal_matrix,
    channel_names=None,
    *,
    alarm_mask=None,
    baseline_fraction: float = 0.2,
    post_window: int | None = None,
) -> dict[str, Any]:
    """Rank variables by robust pre/post shift magnitude.

    This is useful for TEP-like process faults where root causes often create a
    sustained mean/variance deviation. The split is estimated from the first alarm
    when available; otherwise it falls back to the first baseline_fraction of the
    record.
    """
    x = _as_2d(signal_matrix)
    n, d = x.shape
    names = list(channel_names) if channel_names is not None else [f"ch{i}" for i in range(d)]
    if len(names) != d:
        raise ValueError("channel_names length mismatch")

    split = max(10, int(round(n * baseline_fraction)))
    if alarm_mask is not None:
        alarm = np.asarray(alarm_mask, dtype=bool).ravel()
        if alarm.shape[0] != n:
            raise ValueError("alarm_mask length mismatch")
        idx = np.flatnonzero(alarm)
        if idx.size and idx[0] >= 10:
            split = int(idx[0])
    split = min(max(10, split), n - 2)

    pre = x[:split]
    if post_window is None:
        post = x[split:]
    else:
        post = x[split : min(n, split + int(post_window))]
    if post.size == 0:
        post = x[split:]

    center = np.nanmedian(pre, axis=0)
    mad = np.nanmedian(np.abs(pre - center), axis=0)
    scale = np.where(1.4826 * mad > 1e-12, 1.4826 * mad, np.nanstd(pre, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    shift = np.abs(np.nanmedian(post, axis=0) - center) / scale
    var_ratio = np.nanstd(post, axis=0) / (np.nanstd(pre, axis=0) + 1e-12)
    score = shift + 0.25 * np.abs(np.log(np.maximum(var_ratio, 1e-12)))
    max_score = float(np.nanmax(score)) if score.size else 0.0
    normalized = score / (max_score + 1e-12)
    order = np.argsort(normalized)[::-1]

    return {
        "split_index": split,
        "shift_scores": {names[i]: float(normalized[i]) for i in range(d)},
        "raw_shift_z": {names[i]: float(shift[i]) for i in range(d)},
        "variance_ratio": {names[i]: float(var_ratio[i]) for i in range(d)},
        "ranked_variables": [names[int(i)] for i in order],
    }


def _paths_from(root: str, edges: list[dict[str, Any]], max_depth: int = 5) -> list[list[str]]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge["cause"]].append(edge["effect"])
    paths: list[list[str]] = []
    queue = deque([(root, [root])])
    while queue:
        node, path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for nxt in adjacency.get(node, []):
            if nxt in path:
                continue
            new_path = path + [nxt]
            paths.append(new_path)
            queue.append((nxt, new_path))
    return paths


def root_cause_rank_enhanced(
    suspect_variables: Iterable[Any],
    filtered_causal_graph: Any,
    onset_order: Iterable[str],
    *,
    variable_contributions=None,
    shift_scores: dict[str, float] | None = None,
    channel_names=None,
    max_path_depth: int = 5,
) -> dict[str, Any]:
    """Improved transparent root-cause ranking for process data.

    Compared with the first baseline, this adds sustained pre/post shift evidence
    and uses Granger edge significance as a soft strength score when p-values are
    available. This is still a deterministic baseline, not a causal proof.
    """
    edges = _extract_edges(filtered_causal_graph)
    names = list(channel_names) if channel_names is not None else None

    suspects: set[str] = set()
    for item in suspect_variables or []:
        if isinstance(item, (int, np.integer)) and names is not None and 0 <= int(item) < len(names):
            suspects.add(str(names[int(item)]))
        else:
            suspects.add(str(item))

    nodes = sorted(
        {edge["cause"] for edge in edges}
        | {edge["effect"] for edge in edges}
        | suspects
        | {str(x) for x in (onset_order or [])}
        | set((shift_scores or {}).keys())
    )
    if not nodes:
        return {"root_cause_ranking": [], "propagation_paths": [], "confidence": 0.0}

    outgoing = defaultdict(float)
    incoming = defaultdict(float)
    for edge in edges:
        p = edge.get("p_value")
        strength = 1.0
        if p is not None:
            strength = min(3.0, max(0.0, -np.log10(max(float(p), 1e-12)))) / 3.0
        outgoing[edge["cause"]] += strength
        incoming[edge["effect"]] += strength

    onset_position = {str(name): i for i, name in enumerate(onset_order or [])}
    denom_onset = max(1, len(onset_position) - 1)
    max_out = max([outgoing[n] for n in nodes] + [1.0])
    max_in = max([incoming[n] for n in nodes] + [1.0])

    contribution_map: dict[str, float] = {}
    if variable_contributions is not None:
        arr = np.asarray(variable_contributions, dtype=float).ravel()
        if names is not None and arr.size == len(names):
            contribution_map = {str(names[i]): float(arr[i]) for i in range(len(names))}
    max_contrib = max(contribution_map.values()) if contribution_map else 1.0
    max_contrib = max(max_contrib, 1e-12)

    rows = []
    for node in nodes:
        contribution_score = max(0.0, contribution_map.get(node, 0.0)) / max_contrib
        suspect_score = 1.0 if node in suspects else 0.0
        evidence_score = max(suspect_score, contribution_score)
        onset_score = 1.0 - onset_position[node] / denom_onset if node in onset_position else 0.0
        outgoing_score = outgoing[node] / max_out
        upstream_score = 1.0 - incoming[node] / max_in
        shift_score = max(0.0, float((shift_scores or {}).get(node, 0.0)))

        if edges:
            score = (
                0.25 * evidence_score
                + 0.25 * onset_score
                + 0.20 * outgoing_score
                + 0.10 * upstream_score
                + 0.20 * shift_score
            )
        else:
            score = 0.40 * evidence_score + 0.30 * onset_score + 0.30 * shift_score

        rows.append(
            {
                "variable": node,
                "score": float(score),
                "contribution_score": float(contribution_score),
                "suspect_score": float(suspect_score),
                "onset_score": float(onset_score),
                "outgoing_score": float(outgoing_score),
                "upstream_score": float(upstream_score),
                "shift_score": float(shift_score),
                "outgoing_strength": float(outgoing[node]),
                "incoming_strength": float(incoming[node]),
            }
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    second = rows[1]["score"] if len(rows) > 1 else 0.0
    separation = max(0.0, best["score"] - second)
    confidence = float(np.clip(0.70 * best["score"] + 0.30 * separation, 0.0, 1.0))
    return {
        "root_cause_ranking": rows,
        "propagation_paths": _paths_from(best["variable"], edges, max_depth=max_path_depth),
        "confidence": confidence,
        "ranking_method": "enhanced_contribution_onset_granger_shift",
    }
