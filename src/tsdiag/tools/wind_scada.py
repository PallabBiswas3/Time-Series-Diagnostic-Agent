from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import KMeans


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def scada_quality_check(signal_matrix, channel_names, timestamps=None) -> dict[str, Any]:
    x = _as_2d(signal_matrix)
    names = [str(v) for v in channel_names]
    if len(names) != x.shape[1]:
        raise ValueError("channel_names length mismatch")

    flags = []
    missingness = {}
    for i, name in enumerate(names):
        col = x[:, i]
        missing = float(np.mean(~np.isfinite(col)))
        missingness[name] = missing
        finite = col[np.isfinite(col)]
        if missing > 0:
            flags.append({"channel": name, "flag": "missing_values", "fraction": missing})
        if finite.size and np.nanstd(finite) < 1e-12:
            flags.append({"channel": name, "flag": "constant_channel"})

    irregularity = None
    if timestamps is not None:
        ts = np.asarray(timestamps)
        if ts.shape[0] != x.shape[0]:
            raise ValueError("timestamps length mismatch")
        try:
            t = ts.astype("datetime64[ns]").astype("int64")
            dt = np.diff(t).astype(float)
            if dt.size:
                median = np.median(dt)
                irregularity = float(np.mean(np.abs(dt - median) > max(abs(median) * 0.05, 1.0)))
                if irregularity > 0.01:
                    flags.append({"flag": "irregular_sampling", "fraction": irregularity})
        except Exception:
            flags.append({"flag": "unparseable_timestamps"})

    return {
        "quality_flags": flags,
        "missingness": missingness,
        "sampling_irregularity": irregularity,
    }


def detect_wind_operating_regimes(
    signal_matrix,
    channel_names=None,
    *,
    driver_indices: Iterable[int] | None = None,
    operating_regime_labels=None,
    n_regimes: int = 4,
    random_state: int = 0,
) -> dict[str, Any]:
    """Cluster healthy operation into comparable regimes.

    Wind-SCADA is strongly regime dependent. This stage prevents startup, partial
    load and rated-power operation from being compared as if they were identical.
    `channel_names` is accepted for domain-contract compatibility; regime drivers
    can be supplied explicitly by index or inferred from high-variance channels.
    """
    x = _as_2d(signal_matrix)
    if channel_names is not None and len(channel_names) != x.shape[1]:
        raise ValueError("channel_names length mismatch")
    if operating_regime_labels is not None:
        labels = np.asarray(operating_regime_labels)
        if labels.shape[0] != x.shape[0]:
            raise ValueError("operating_regime_labels length mismatch")
    else:
        if driver_indices is None:
            variances = np.nanvar(x, axis=0)
            driver_indices = np.argsort(variances)[::-1][: min(4, x.shape[1])]
        cols = np.asarray(list(driver_indices), dtype=int)
        if cols.size == 0:
            raise ValueError("at least one regime driver is required")
        drivers = x[:, cols]
        med = np.nanmedian(drivers, axis=0)
        scale = np.nanmedian(np.abs(drivers - med), axis=0) * 1.4826
        scale = np.where(scale < 1e-12, np.nanstd(drivers, axis=0), scale)
        scale = np.where(scale < 1e-12, 1.0, scale)
        z = np.nan_to_num((drivers - med) / scale)
        k = max(1, min(int(n_regimes), len(x)))
        labels = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit_predict(z)

    descriptions = {}
    for value in np.unique(labels):
        mask = labels == value
        descriptions[int(value)] = {
            "samples": int(np.sum(mask)),
            "mean": np.nanmean(x[mask], axis=0).tolist(),
        }
    return {"regime_ids": labels, "regime_descriptions": descriptions}


@dataclass
class WindNormalBehaviorState:
    target_indices: list[int]
    predictor_indices: list[int]
    models: dict[tuple[int, int], HistGradientBoostingRegressor]
    global_models: dict[int, HistGradientBoostingRegressor]
    residual_scale: np.ndarray
    regime_ids: list[int]


def _valid_rows(x: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(x), axis=1)


def fit_wind_normal_behavior_model(
    normal_reference,
    reference_regime_ids,
    *,
    target_indices: Iterable[int] | None = None,
    predictor_indices: Iterable[int] | None = None,
    max_train_samples_per_regime: int = 20000,
    random_state: int = 0,
) -> WindNormalBehaviorState:
    ref = _as_2d(normal_reference)
    regimes = np.asarray(reference_regime_ids)
    if regimes.shape[0] != ref.shape[0]:
        raise ValueError("reference_regime_ids length mismatch")

    targets = list(range(ref.shape[1])) if target_indices is None else [int(i) for i in target_indices]
    predictors = list(range(ref.shape[1])) if predictor_indices is None else [int(i) for i in predictor_indices]
    if not targets or not predictors:
        raise ValueError("normal behavior model requires targets and predictors")

    models: dict[tuple[int, int], HistGradientBoostingRegressor] = {}
    global_models: dict[int, HistGradientBoostingRegressor] = {}
    expected_ref = np.full((ref.shape[0], len(targets)), np.nan)

    for target_pos, target in enumerate(targets):
        feature_cols = [j for j in predictors if j != target]
        if not feature_cols:
            feature_cols = predictors

        valid = _valid_rows(ref[:, feature_cols]) & np.isfinite(ref[:, target])
        if np.sum(valid) < 10:
            raise ValueError(f"insufficient finite reference samples for target {target}")
        global_model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_state,
        )
        global_model.fit(ref[valid][:, feature_cols], ref[valid, target])
        global_models[target] = global_model

        for regime in np.unique(regimes):
            mask = valid & (regimes == regime)
            idx = np.flatnonzero(mask)
            if idx.size < 30:
                continue
            if idx.size > max_train_samples_per_regime:
                sample_positions = np.linspace(0, idx.size - 1, max_train_samples_per_regime).astype(int)
                idx = idx[sample_positions]
            model = HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=random_state,
            )
            model.fit(ref[idx][:, feature_cols], ref[idx, target])
            models[(int(regime), target)] = model

        for regime in np.unique(regimes):
            mask = regimes == regime
            valid_mask = mask & _valid_rows(ref[:, feature_cols])
            if not np.any(valid_mask):
                continue
            model = models.get((int(regime), target), global_model)
            expected_ref[valid_mask, target_pos] = model.predict(ref[valid_mask][:, feature_cols])

    residuals = ref[:, targets] - expected_ref
    residual_scale = np.nanmedian(np.abs(residuals - np.nanmedian(residuals, axis=0)), axis=0) * 1.4826
    fallback = np.nanstd(residuals, axis=0)
    residual_scale = np.where(residual_scale < 1e-12, fallback, residual_scale)
    residual_scale = np.where(residual_scale < 1e-12, 1.0, residual_scale)

    return WindNormalBehaviorState(
        target_indices=targets,
        predictor_indices=predictors,
        models=models,
        global_models=global_models,
        residual_scale=residual_scale,
        regime_ids=[int(v) for v in np.unique(regimes)],
    )


def predict_wind_normal_behavior(
    signal_matrix,
    regime_ids,
    state: WindNormalBehaviorState,
) -> dict[str, Any]:
    x = _as_2d(signal_matrix)
    regimes = np.asarray(regime_ids)
    if regimes.shape[0] != x.shape[0]:
        raise ValueError("regime_ids length mismatch")

    expected = np.full((x.shape[0], len(state.target_indices)), np.nan)
    for target_pos, target in enumerate(state.target_indices):
        feature_cols = [j for j in state.predictor_indices if j != target]
        if not feature_cols:
            feature_cols = state.predictor_indices
        valid_features = _valid_rows(x[:, feature_cols])
        for regime in np.unique(regimes):
            mask = (regimes == regime) & valid_features
            if not np.any(mask):
                continue
            model = state.models.get((int(regime), target), state.global_models[target])
            expected[mask, target_pos] = model.predict(x[mask][:, feature_cols])

    observed = x[:, state.target_indices]
    residuals = observed - expected
    normalized = residuals / state.residual_scale
    uncertainty = np.tile(state.residual_scale, (x.shape[0], 1))
    return {
        "expected_signal": expected,
        "observed_signal": observed,
        "residuals": residuals,
        "normalized_residuals": normalized,
        "model_uncertainty": uncertainty,
        "target_indices": list(state.target_indices),
    }


def normal_behavior_model(
    signal_matrix,
    regime_ids,
    *,
    normal_reference=None,
    reference_regime_ids=None,
    target_indices: Iterable[int] | None = None,
    predictor_indices: Iterable[int] | None = None,
    physics_model=None,
    ambient_conditions=None,
) -> dict[str, Any]:
    """One-shot domain-contract wrapper around the fit/predict NBM API."""
    x = _as_2d(signal_matrix)
    ref = x if normal_reference is None else _as_2d(normal_reference)
    ref_regimes = np.asarray(regime_ids if reference_regime_ids is None else reference_regime_ids)
    state = fit_wind_normal_behavior_model(
        ref,
        ref_regimes,
        target_indices=target_indices,
        predictor_indices=predictor_indices,
    )
    prediction = predict_wind_normal_behavior(x, regime_ids, state)
    return {
        "expected_signal": prediction["expected_signal"],
        "model_uncertainty": prediction["model_uncertainty"],
        "normal_behavior_state": state,
        "normalized_residuals": prediction["normalized_residuals"],
    }


def wind_residual_anomaly_detection(
    normalized_residuals,
    *,
    threshold: float = 3.5,
    persistence: int = 3,
) -> dict[str, Any]:
    z = np.abs(_as_2d(normalized_residuals))
    per_channel = z >= float(threshold)
    score = np.nanmax(z, axis=1)
    raw_alarm = np.any(per_channel, axis=1)

    k = max(1, int(persistence))
    alarm = np.zeros_like(raw_alarm)
    run = 0
    for i, active in enumerate(raw_alarm):
        run = run + 1 if active else 0
        if run >= k:
            alarm[i] = True

    return {
        "anomaly_scores": score,
        "channel_alarm_mask": per_channel,
        "raw_alarm_mask": raw_alarm,
        "alarm_mask": alarm,
        "threshold": float(threshold),
        "persistence": k,
    }
