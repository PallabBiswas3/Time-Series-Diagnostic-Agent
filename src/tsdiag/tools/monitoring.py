from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input contains NaN or Inf")
    return arr


def apply_alarm_persistence(alarm_mask, min_consecutive: int = 1) -> np.ndarray:
    """Keep alarms only after a run persists for min_consecutive samples.

    This is a simple false-alarm-control baseline. It suppresses isolated spikes but
    preserves sustained process deviations.
    """
    alarm = np.asarray(alarm_mask, dtype=bool).ravel()
    k = max(1, int(min_consecutive))
    if k <= 1:
        return alarm.copy()
    out = np.zeros_like(alarm)
    run = 0
    for i, active in enumerate(alarm):
        if active:
            run += 1
            if run >= k:
                out[i] = True
        else:
            run = 0
    return out


def make_lagged_matrix(signal_matrix, lags: int = 1) -> np.ndarray:
    """Construct a dynamic/lagged matrix [x_t, x_{t-1}, ..., x_{t-lags}]."""
    x = _as_2d(signal_matrix)
    p = max(0, int(lags))
    if p == 0:
        return x.copy()
    if x.shape[0] <= p:
        raise ValueError("not enough samples for requested lags")
    rows = []
    for lag in range(p + 1):
        rows.append(x[p - lag : x.shape[0] - lag])
    return np.hstack(rows)


def _choose_n_components(reference: np.ndarray, variance_target: float) -> int:
    full = PCA().fit(reference)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    return min(max(1, n_components), min(reference.shape))


def _monitoring_core(
    current_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    variance_target: float = 0.95,
    alpha: float = 0.99,
    min_consecutive: int = 1,
) -> dict[str, Any]:
    cur = _as_2d(current_features)
    ref = _as_2d(reference_features)
    if cur.shape[1] != ref.shape[1]:
        raise ValueError("current and reference dimensions differ")

    n_components = _choose_n_components(ref, variance_target)
    pca = PCA(n_components=n_components).fit(ref)

    ref_scores = pca.transform(ref)
    cur_scores = pca.transform(cur)
    ref_recon = pca.inverse_transform(ref_scores)
    cur_recon = pca.inverse_transform(cur_scores)

    ref_spe = np.sum((ref - ref_recon) ** 2, axis=1)
    spe = np.sum((cur - cur_recon) ** 2, axis=1)

    score_mu = ref_scores.mean(axis=0)
    score_cov = np.atleast_2d(np.cov(ref_scores, rowvar=False)) + 1e-9 * np.eye(n_components)
    inv_cov = np.linalg.pinv(score_cov)
    ref_t2 = np.einsum("ij,jk,ik->i", ref_scores - score_mu, inv_cov, ref_scores - score_mu)
    t2 = np.einsum("ij,jk,ik->i", cur_scores - score_mu, inv_cov, cur_scores - score_mu)

    limits = {
        "spe": float(np.quantile(ref_spe, alpha)),
        "t2": float(np.quantile(ref_t2, alpha)),
    }
    raw_alarm = (spe > limits["spe"]) | (t2 > limits["t2"])
    alarm = apply_alarm_persistence(raw_alarm, min_consecutive)
    score = np.maximum(t2 / (limits["t2"] + 1e-12), spe / (limits["spe"] + 1e-12))

    return {
        "scores": cur_scores,
        "t2": t2,
        "spe": spe,
        "combined_score": score,
        "control_limits": limits,
        "raw_alarm_mask": raw_alarm,
        "alarm_mask": alarm,
        "pca_state": {
            "components": pca.components_,
            "mean": pca.mean_,
            "explained_variance": pca.explained_variance_,
            "n_components": n_components,
        },
        "calibration": {
            "alpha": float(alpha),
            "min_consecutive": int(min_consecutive),
            "variance_target": float(variance_target),
        },
    }


def calibrated_pca_monitoring(
    standardized_signal,
    normal_reference,
    *,
    variance_target: float = 0.95,
    alpha: float = 0.99,
    min_consecutive: int = 1,
) -> dict[str, Any]:
    return _monitoring_core(
        standardized_signal,
        normal_reference,
        variance_target=variance_target,
        alpha=alpha,
        min_consecutive=min_consecutive,
    )


def dpca_monitoring(
    standardized_signal,
    normal_reference,
    *,
    lags: int = 2,
    variance_target: float = 0.95,
    alpha: float = 0.99,
    min_consecutive: int = 1,
) -> dict[str, Any]:
    """Dynamic PCA baseline using lagged variables.

    DPCA often reduces false alarms in autocorrelated process data because the PCA
    model sees short temporal context rather than isolated samples.
    """
    x = _as_2d(standardized_signal)
    ref = _as_2d(normal_reference)
    p = max(0, int(lags))
    cur_features = make_lagged_matrix(x, p)
    ref_features = make_lagged_matrix(ref, p)
    core = _monitoring_core(
        cur_features,
        ref_features,
        variance_target=variance_target,
        alpha=alpha,
        min_consecutive=min_consecutive,
    )

    pad = np.zeros(p, dtype=bool)
    pad_float = np.full(p, np.nan)
    core["raw_alarm_mask"] = np.r_[pad, core["raw_alarm_mask"]]
    core["alarm_mask"] = np.r_[pad, core["alarm_mask"]]
    core["t2"] = np.r_[pad_float, core["t2"]]
    core["spe"] = np.r_[pad_float, core["spe"]]
    core["combined_score"] = np.r_[pad_float, core["combined_score"]]
    core["dpca_lags"] = p
    core["calibration"]["method"] = "dpca"
    core["calibration"]["lags"] = p
    return core


@dataclass(frozen=True)
class MonitoringConfig:
    method: str
    alpha: float
    min_consecutive: int
    variance_target: float = 0.95
    lags: int = 0
    validation_false_alarm_rate: float | None = None


def run_monitoring_method(
    standardized_signal,
    standardized_reference,
    config: MonitoringConfig,
) -> dict[str, Any]:
    method = config.method.lower()
    if method == "pca":
        result = calibrated_pca_monitoring(
            standardized_signal,
            standardized_reference,
            variance_target=config.variance_target,
            alpha=config.alpha,
            min_consecutive=config.min_consecutive,
        )
    elif method == "dpca":
        result = dpca_monitoring(
            standardized_signal,
            standardized_reference,
            lags=config.lags,
            variance_target=config.variance_target,
            alpha=config.alpha,
            min_consecutive=config.min_consecutive,
        )
    else:
        raise ValueError("method must be 'pca' or 'dpca'")
    result["calibration"]["method"] = method
    return result


def calibrate_monitoring_config(
    standardized_reference,
    *,
    method: str = "pca",
    variance_target: float = 0.95,
    target_false_alarm_rate: float = 0.05,
    calibration_fraction: float = 0.35,
    alpha_grid: Iterable[float] = (0.99, 0.995, 0.9975, 0.999, 0.9995),
    persistence_grid: Iterable[int] = (1, 2, 3, 5),
    lags_grid: Iterable[int] = (0, 1, 2, 3),
) -> dict[str, Any]:
    """Tune monitoring thresholds on held-out healthy data.

    This targets false-alarm control only. It does not use fault labels, so it is a
    fair normal-operation calibration stage.
    """
    ref = _as_2d(standardized_reference)
    if ref.shape[0] < 40:
        raise ValueError("need at least 40 healthy samples for calibration")

    frac = min(max(float(calibration_fraction), 0.15), 0.5)
    split = max(20, int(round(ref.shape[0] * (1.0 - frac))))
    split = min(split, ref.shape[0] - 10)
    fit_ref = ref[:split]
    val_ref = ref[split:]

    method = method.lower()
    lags_candidates = [0] if method == "pca" else [max(1, int(v)) for v in lags_grid]
    trials = []
    for lags in lags_candidates:
        if method == "dpca" and min(len(fit_ref), len(val_ref)) <= lags + 5:
            continue
        for alpha in alpha_grid:
            for persistence in persistence_grid:
                cfg = MonitoringConfig(
                    method=method,
                    alpha=float(alpha),
                    min_consecutive=int(persistence),
                    variance_target=float(variance_target),
                    lags=int(lags),
                )
                try:
                    monitored = run_monitoring_method(val_ref, fit_ref, cfg)
                    far = float(np.nanmean(np.asarray(monitored["alarm_mask"], dtype=bool)))
                except Exception:
                    continue
                trials.append({"config": cfg, "validation_false_alarm_rate": far})

    if not trials:
        raise RuntimeError("no valid monitoring calibration trials")

    target = float(target_false_alarm_rate)

    def key(row):
        far = row["validation_false_alarm_rate"]
        over = far > target
        # Prefer settings close to the target from below. If every setting is over,
        # choose the lowest false-alarm rate.
        return (over, abs(far - target) if not over else far, row["config"].min_consecutive)

    best = sorted(trials, key=key)[0]
    cfg = best["config"]
    chosen = MonitoringConfig(
        method=cfg.method,
        alpha=cfg.alpha,
        min_consecutive=cfg.min_consecutive,
        variance_target=cfg.variance_target,
        lags=cfg.lags,
        validation_false_alarm_rate=best["validation_false_alarm_rate"],
    )
    return {
        "config": chosen,
        "trials": [
            {
                "method": row["config"].method,
                "alpha": row["config"].alpha,
                "min_consecutive": row["config"].min_consecutive,
                "lags": row["config"].lags,
                "validation_false_alarm_rate": row["validation_false_alarm_rate"],
            }
            for row in trials
        ],
    }
