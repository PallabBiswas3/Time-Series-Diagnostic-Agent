from __future__ import annotations

import numpy as np


def _history(cycle_index, capacity_ah) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(cycle_index, dtype=float).ravel()
    y = np.asarray(capacity_ah, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("cycle_index and capacity_ah must have equal length")
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size:
        order = np.argsort(x)
        x, y = x[order], y[order]
    return x, y


def smooth_capacity(capacity_ah, window: int = 5) -> np.ndarray:
    y = np.asarray(capacity_ah, dtype=float).ravel()
    if y.size < 3 or window <= 1:
        return y.copy()
    radius = max(1, int(window) // 2)
    return np.asarray([
        np.median(y[max(0, i-radius):min(len(y), i+radius+1)])
        for i in range(len(y))
    ], dtype=float)


def capacity_features(cycle_index, capacity_ah, *, nominal_capacity_ah=2.0, slope_window=20) -> dict:
    x, y = _history(cycle_index, capacity_ah)
    smooth = smooth_capacity(y)
    if x.size < 2:
        return {
            "cycle_count": int(x.size),
            "current_capacity_ah": None if not x.size else float(y[-1]),
            "soh": None if not x.size else float(y[-1] / nominal_capacity_ah),
            "local_slope_ah_per_cycle": None,
            "global_slope_ah_per_cycle": None,
            "smoothed_capacity_ah": smooth.tolist(),
        }
    n = min(max(3, int(slope_window)), x.size)
    return {
        "cycle_count": int(x.size),
        "current_capacity_ah": float(y[-1]),
        "soh": float(y[-1] / nominal_capacity_ah),
        "local_slope_ah_per_cycle": float(np.polyfit(x[-n:], smooth[-n:], 1)[0]),
        "global_slope_ah_per_cycle": float(np.polyfit(x, smooth, 1)[0]),
        "smoothed_capacity_ah": smooth.tolist(),
    }


def estimate_capacity_eol(
    cycle_index,
    capacity_ah,
    *,
    eol_capacity_ah=1.4,
    nominal_capacity_ah=2.0,
    minimum_observations=20,
    slope_window=20,
    maximum_projection_cycles=500,
    confidence_z=1.96,
) -> dict:
    """Causal capacity-trajectory EOL estimate with an uncertainty interval.

    Only observations supplied in ``cycle_index``/``capacity_ah`` are used. The
    interval is later scored against right-censoring bounds; the predictor never
    receives those future censoring bounds.
    """
    x, y = _history(cycle_index, capacity_ah)
    features = capacity_features(
        x, y, nominal_capacity_ah=nominal_capacity_ah, slope_window=slope_window
    )
    base = {**features, "method": "local_linear_capacity_interval"}
    if x.size < minimum_observations:
        return {**base, "abstained": True, "abstain_reason": "insufficient_history"}

    smooth = np.asarray(features["smoothed_capacity_ah"], dtype=float)
    current_cycle = float(x[-1])
    current_capacity = float(smooth[-1])
    if current_capacity <= eol_capacity_ah:
        return {
            **base,
            "abstained": False,
            "abstain_reason": None,
            "predicted_eol_cycle": current_cycle,
            "eol_interval_cycles": [current_cycle, current_cycle],
            "remaining_useful_life_cycles": 0.0,
            "uncertainty_cycles": 0.0,
            "risk": 1.0,
        }

    n = min(max(3, int(slope_window)), x.size)
    xl, yl = x[-n:], smooth[-n:]
    slope, intercept = np.polyfit(xl, yl, 1)
    slope, intercept = float(slope), float(intercept)
    if slope >= -1e-5:
        return {**base, "abstained": True, "abstain_reason": "non_degrading_or_unstable_slope"}

    eol = float((eol_capacity_ah - intercept) / slope)
    rul = float(eol - current_cycle)
    if rul < 0 or rul > maximum_projection_cycles:
        return {**base, "abstained": True, "abstain_reason": "projection_outside_credible_horizon"}

    fitted = slope * xl + intercept
    residual = yl - fitted
    residual_std = float(np.std(residual, ddof=1)) if len(residual) > 2 else 0.0
    uncertainty = float(np.clip(confidence_z * residual_std / max(abs(slope), 1e-6), 1.0, maximum_projection_cycles))
    lower_eol = max(current_cycle, eol - uncertainty)
    upper_eol = eol + uncertainty

    margin = max(current_capacity - eol_capacity_ah, 0.0)
    full_margin = max(nominal_capacity_ah - eol_capacity_ah, 1e-6)
    risk = float(np.clip(1.0 - margin / full_margin, 0.0, 1.0))
    return {
        **base,
        "abstained": False,
        "abstain_reason": None,
        "predicted_eol_cycle": eol,
        "eol_interval_cycles": [float(lower_eol), float(upper_eol)],
        "remaining_useful_life_cycles": rul,
        "uncertainty_cycles": uncertainty,
        "risk": risk,
    }
