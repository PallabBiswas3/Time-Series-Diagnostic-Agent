from __future__ import annotations

import numpy as np


def _as_2d(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def rolling_statistics(residuals, timestamps=None, window: int = 20):
    x = _as_2d(residuals)
    if window < 3:
        raise ValueError("window must be >= 3")
    n, d = x.shape
    means = np.full((n, d), np.nan)
    stds = np.full((n, d), np.nan)
    slopes = np.full((n, d), np.nan)
    for i in range(window - 1, n):
        seg = x[i - window + 1:i + 1]
        t = np.arange(window, dtype=float)
        means[i] = np.nanmean(seg, axis=0)
        stds[i] = np.nanstd(seg, axis=0)
        for j in range(d):
            col = seg[:, j]
            finite = np.isfinite(col)
            if finite.sum() >= 3:
                slopes[i, j] = np.polyfit(t[finite], col[finite], 1)[0]
    return {"rolling_mean": means, "rolling_std": stds, "rolling_slope": slopes}


def cross_sensor_relationships(signal_matrix, channel_names=None, relationship_pairs=None, reference_correlations=None):
    x = _as_2d(signal_matrix)
    names = channel_names or [f"ch{i}" for i in range(x.shape[1])]
    corr = np.corrcoef(x, rowvar=False)
    pairs = relationship_pairs
    if pairs is None:
        pairs = [(i, j) for i in range(x.shape[1]) for j in range(i + 1, x.shape[1])]
    relationships = []
    shifts = []
    for a, b in pairs:
        ia = names.index(a) if isinstance(a, str) else int(a)
        ib = names.index(b) if isinstance(b, str) else int(b)
        value = float(corr[ia, ib])
        item = {"pair": [names[ia], names[ib]], "correlation": value}
        if reference_correlations is not None:
            ref = np.asarray(reference_correlations, dtype=float)
            shift = float(abs(value - ref[ia, ib]))
            item["shift"] = shift
            shifts.append(item)
        relationships.append(item)
    return {"correlations": relationships, "relationship_shift_scores": shifts}


def change_point_detection(residuals, timestamps=None, min_size: int = 20, z_threshold: float = 4.0, regime_ids=None, variance_model=None):
    """Deterministic robust mean-shift detector using left/right median contrasts.

    This is intentionally dependency-light and serves as the baseline against
    which more advanced PELT/BOCPD methods can later be compared.
    """
    x = _as_2d(residuals)
    n, d = x.shape
    if n < 2 * min_size + 1:
        return {"change_points": [], "change_magnitudes": []}
    scale = np.nanmedian(np.abs(x - np.nanmedian(x, axis=0)), axis=0) * 1.4826
    scale = np.where(scale < 1e-9, np.nanstd(x, axis=0) + 1e-9, scale)
    candidates = []
    for i in range(min_size, n - min_size):
        left = np.nanmedian(x[i - min_size:i], axis=0)
        right = np.nanmedian(x[i:i + min_size], axis=0)
        score = np.max(np.abs(right - left) / scale)
        if score >= z_threshold:
            candidates.append((i, float(score), np.abs(right - left)))
    # Non-maximum suppression: keep strongest point within one min_size neighborhood.
    selected = []
    for idx, score, mag in sorted(candidates, key=lambda z: z[1], reverse=True):
        if all(abs(idx - kept[0]) >= min_size for kept in selected):
            selected.append((idx, score, mag))
    selected.sort(key=lambda z: z[0])
    cps = []
    mags = []
    for idx, score, mag in selected:
        time_value = timestamps[idx] if timestamps is not None else idx
        cps.append({"index": int(idx), "time": time_value, "score": score})
        mags.append({"index": int(idx), "per_channel": np.asarray(mag).tolist(), "max_magnitude": float(np.max(mag))})
    return {"change_points": cps, "change_magnitudes": mags}
