from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResidualCUSUMConfig:
    drift: float = 0.25
    threshold: float = 8.0
    reset_after_alarm: bool = True
    hold_samples: int = 6
    two_sided: bool = True


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def residual_cusum(
    normalized_residuals,
    *,
    timestamps=None,
    config: ResidualCUSUMConfig | None = None,
) -> dict[str, Any]:
    """Sequential two-sided CUSUM for low-amplitude persistent residual shifts.

    Inputs are normalized healthy-model residuals. The detector accumulates
    departures larger than ``drift`` and emits sample indices, optional timestamps,
    and a short post-change hold mask suitable for fusion into an event alarm stream.
    """
    cfg = config or ResidualCUSUMConfig()
    x = _as_2d(normalized_residuals)
    x = np.where(np.isfinite(x), x, 0.0)
    n, d = x.shape

    ts = None
    if timestamps is not None:
        ts = np.asarray(timestamps)
        if ts.shape[0] != n:
            raise ValueError("timestamps length mismatch")

    pos = np.zeros(d, dtype=float)
    neg = np.zeros(d, dtype=float)
    pos_history = np.zeros((n, d), dtype=float)
    neg_history = np.zeros((n, d), dtype=float)
    point_mask = np.zeros((n, d), dtype=bool)

    for i in range(n):
        row = x[i]
        pos = np.maximum(0.0, pos + row - float(cfg.drift))
        if cfg.two_sided:
            neg = np.maximum(0.0, neg - row - float(cfg.drift))
        else:
            neg[:] = 0.0
        triggered = (pos >= float(cfg.threshold)) | (neg >= float(cfg.threshold))
        point_mask[i] = triggered
        pos_history[i] = pos
        neg_history[i] = neg
        if cfg.reset_after_alarm and np.any(triggered):
            pos[triggered] = 0.0
            neg[triggered] = 0.0

    aggregate_points = np.any(point_mask, axis=1)
    hold = np.zeros(n, dtype=bool)
    hold_samples = max(1, int(cfg.hold_samples))
    for idx in np.flatnonzero(aggregate_points):
        hold[idx:min(n, idx + hold_samples)] = True

    strength = np.maximum(pos_history, neg_history)
    score = np.nanmax(strength, axis=1) if d else np.zeros(n)
    change_points = np.flatnonzero(aggregate_points).astype(int).tolist()
    channel_change_points = {
        int(j): np.flatnonzero(point_mask[:, j]).astype(int).tolist()
        for j in range(d)
        if np.any(point_mask[:, j])
    }

    def _serialize_timestamp(value):
        if isinstance(value, np.datetime64):
            return str(value)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    change_point_timestamps = [] if ts is None else [_serialize_timestamp(ts[i]) for i in change_points]
    channel_change_point_timestamps = {}
    if ts is not None:
        channel_change_point_timestamps = {
            int(channel): [_serialize_timestamp(ts[i]) for i in indices]
            for channel, indices in channel_change_points.items()
        }

    return {
        "change_points": change_points,
        "change_point_timestamps": change_point_timestamps,
        "channel_change_points": channel_change_points,
        "channel_change_point_timestamps": channel_change_point_timestamps,
        "change_point_mask": aggregate_points,
        "alarm_mask": hold,
        "cusum_scores": score,
        "positive_cusum": pos_history,
        "negative_cusum": neg_history,
        "configuration": {
            "drift": float(cfg.drift),
            "threshold": float(cfg.threshold),
            "reset_after_alarm": bool(cfg.reset_after_alarm),
            "hold_samples": hold_samples,
            "two_sided": bool(cfg.two_sided),
        },
    }
