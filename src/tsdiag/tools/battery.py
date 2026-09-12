from __future__ import annotations

import numpy as np


def _as_history(cycle_index, capacity_ah) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(cycle_index, dtype=float).ravel()
    y = np.asarray(capacity_ah, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("cycle_index and capacity_ah must have equal length")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    return x[order], y[order]


def smooth_capacity_history(capacity_ah, *, window: int = 5) -> np.ndarray:
    y = np.asarray(capacity_ah, dtype=float).ravel()
    if y.size == 0:
        return y.copy()
    w = max(1, int(window))
    if w == 1 or y.size < 3:
        return y.copy()
    half = w // 2
    out = np.empty_like(y)
    for i in range(y.size):
        lo = max(0, i - half)
        hi = min(y.size, i + half + 1)
        out[i] = float(np.median(y[lo:hi]))
    return out


def capacity_health_features(
    cycle_index,
    capacity_ah,
    *,
    nominal_capacity_ah: float = 2.0,
    smoothing_window: int = 5,
    slope_window: int = 20,
) -> dict:
    x, y = _as_history(cycle_index, capacity_ah)
    if x.size < 2:
        return {
            "cycle_count": int(x.size),
            "current_capacity_ah": None if x.size == 0 else float(y[-1]),
            "soh": None if x.size == 0 else float(y[-1] / nominal_capacity_ah),
            "smoothed_capacity_ah": y.tolist(),
            "local_slope_ah_per_cycle": None,
            "global_slope_ah_per_cycle": None,
            "knee_cycle": None,
            "knee_score": None,
        }

    smooth = smooth_capacity_history(y, window=smoothing_window)
    global_slope = float(np.polyfit(x, smooth, 1)[0])
    n_local = min(max(3, int(slope_window)), x.size)
    local_slope = float(np.polyfit(x[-n_local:], smooth[-n_local:], 1)[0])

    knee_cycle = None
    knee_score = None
    if x.size >= 12:
        min_side = max(4, min(10, x.size // 4))
        best_score = 0.0
        best_cycle = None
        for split in range(min_side, x.size - min_side):
            left_slope = float(np.polyfit(x[:split], smooth[:split], 1)[0])
            right_slope = float(np.polyfit(x[split:], smooth[split:], 1)[0])
            acceleration = max(0.0, abs(right_slope) - abs(left_slope))
            scale = max(abs(global_slope), 1e-6)
            score = float(acceleration / scale)
            if score > best_score:
                best_score = score
                best_cycle = int(round(x[split]))
        if best_cycle is not None and best_score >= 0.35:
            knee_cycle = best_cycle
            knee_score = float(best_score)

    return {
        "cycle_count": int(x.size),
        "current_capacity_ah": float(y[-1]),
        "soh": float(y[-1] / nominal_capacity_ah),
        "smoothed_capacity_ah": smooth.tolist(),
        "local_slope_ah_per_cycle": local_slope,
        "global_slope_ah_per_cycle": global_slope,
        "knee_cycle": knee_cycle,
        "knee_score": knee_score,
    }


def estimate_capacity_rul(
    cycle_index,
    capacity_ah,
    *,
    eol_capacity_ah: float = 1.4,
    nominal_capacity_ah: float = 2.0,
    smoothing_window: int = 5,
    slope_window: int = 20,
    minimum_observations: int = 20,
    maximum_projection_cycles: int = 500,
) -> dict:
    x, y = _as_history(cycle_index, capacity_ah)
    features = capacity_health_features(
        x,
        y,
        nominal_capacity_ah=nominal_capacity_ah,
        smoothing_window=smoothing_window,
        slope_window=slope_window,
    )
    if x.size < minimum_observations:
        return {
            **features,
            "predicted_eol_cycle": None,
            "remaining_useful_life_cycles": None,
            "uncertainty_cycles": None,
            "risk": None,
            "abstained": True,
            "abstain_reason": f"Fewer than {minimum_observations} discharge observations are available.",
            "method": "robust_local_capacity_extrapolation",
        }

    smooth = np.asarray(features["smoothed_capacity_ah"], dtype=float)
    current_cycle = float(x[-1])
    current_capacity = float(smooth[-1])
    if current_capacity <= eol_capacity_ah:
        return {
            **features,
            "predicted_eol_cycle": current_cycle,
            "remaining_useful_life_cycles": 0.0,
            "uncertainty_cycles": 0.0,
            "risk": 1.0,
            "abstained": False,
            "abstain_reason": None,
            "method": "robust_local_capacity_extrapolation",
        }

    n_local = min(max(3, int(slope_window)), x.size)
    x_local = x[-n_local:]
    y_local = smooth[-n_local:]
    slope, intercept = np.polyfit(x_local, y_local, 1)
    slope = float(slope)
    intercept = float(intercept)
    if slope >= -1e-5:
        return {
            **features,
            "predicted_eol_cycle": None,
            "remaining_useful_life_cycles": None,
            "uncertainty_cycles": None,
            "risk": None,
            "abstained": True,
            "abstain_reason": "Observed capacity trajectory does not show a stable degrading slope.",
            "method": "robust_local_capacity_extrapolation",
        }

    projected_eol = float((eol_capacity_ah - intercept) / slope)
    rul = float(max(projected_eol - current_cycle, 0.0))
    if rul > float(maximum_projection_cycles):
        return {
            **features,
            "predicted_eol_cycle": None,
            "remaining_useful_life_cycles": None,
            "uncertainty_cycles": None,
            "risk": None,
            "abstained": True,
            "abstain_reason": "Projected EOL lies beyond the configured credible extrapolation horizon.",
            "method": "robust_local_capacity_extrapolation",
        }

    fitted = slope * x_local + intercept
    residual_std = float(np.std(y_local - fitted, ddof=1)) if x_local.size > 2 else 0.0
    slope_mag = max(abs(slope), 1e-6)
    uncertainty_cycles = float(np.clip(1.96 * residual_std / slope_mag, 1.0, maximum_projection_cycles))

    capacity_margin = max(current_capacity - eol_capacity_ah, 0.0)
    nominal_margin = max(nominal_capacity_ah - eol_capacity_ah, 1e-6)
    risk = float(np.clip(1.0 - capacity_margin / nominal_margin, 0.0, 1.0))
    if features["knee_cycle"] is not None and current_cycle >= float(features["knee_cycle"]):
        risk = float(np.clip(risk + 0.10, 0.0, 1.0))

    return {
        **features,
        "predicted_eol_cycle": projected_eol,
        "remaining_useful_life_cycles": rul,
        "uncertainty_cycles": uncertainty_cycles,
        "risk": risk,
        "abstained": False,
        "abstain_reason": None,
        "method": "robust_local_capacity_extrapolation",
    }
