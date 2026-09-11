from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .root_rank_calibration import cross_validate_root_ranker as _cross_validate_root_ranker


DEFAULT_ROOT_CANDIDATE_LIMIT = 24


def screen_root_candidates(
    records: Iterable[dict[str, Any]],
    *,
    candidate_limit: int = DEFAULT_ROOT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """Apply label-blind pre-ranking candidate screening.

    Candidate generation and calibrated ranking are deliberately separated. The
    post-mortem record may contain every global TEP catalog root so an omitted true
    root can still be audited, but the ranker only receives the strongest
    data-driven candidates according to pre/post shift evidence. This prevents a
    large catalog from turning calibration into a topology-prior lookup while never
    using the held-out fault label.
    """
    limit = max(1, int(candidate_limit))
    screened: list[dict[str, Any]] = []
    for record in records:
        row = deepcopy(record)
        candidates = list(row.get("candidates", []))
        candidates.sort(
            key=lambda item: (
                float(item.get("pre_post_shift_score", 0.0) or 0.0),
                float(item.get("contribution_score", 0.0) or 0.0),
                float(item.get("onset_earliness_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        row["candidates"] = candidates[:limit]
        row["candidate_screening"] = {
            "method": "label_blind_shift_screen",
            "candidate_limit": limit,
            "pre_screen_count": len(candidates),
            "post_screen_count": min(limit, len(candidates)),
        }
        screened.append(row)
    return screened


def cross_validate_root_ranker_screened(
    records: Iterable[dict[str, Any]],
    *,
    candidate_limit: int = DEFAULT_ROOT_CANDIDATE_LIMIT,
    **kwargs,
) -> dict[str, Any]:
    """Run fault-level CV after a fixed, label-blind candidate screen."""
    screened = screen_root_candidates(records, candidate_limit=candidate_limit)
    result = _cross_validate_root_ranker(screened, **kwargs)
    result["candidate_screening"] = {
        "method": "label_blind_shift_screen",
        "candidate_limit": int(candidate_limit),
    }
    return result
