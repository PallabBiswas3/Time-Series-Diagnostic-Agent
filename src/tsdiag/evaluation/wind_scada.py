from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class WindEventEvaluation:
    event_id: int
    is_anomaly_event: bool
    event_detected: bool
    max_criticality: int
    alarm_fraction: float
    event_window_recall: float | None
    event_window_precision: float | None
    first_detection_index: int | None
    lead_time_samples: int | None
    lead_time_minutes: float | None


def calculate_criticality(
    alarm_mask,
    normal_mask=None,
    *,
    initial: int = 0,
    maximum: int = 1000,
) -> np.ndarray:
    """CARE-style bounded cumulative criticality.

    Criticality increases by one on a detected anomaly while the turbine is in an
    operator-normal state, decreases by one on a non-alarm normal timestamp, and
    remains unchanged for already-abnormal operating states.
    """
    alarm = np.asarray(alarm_mask, dtype=bool).ravel()
    if normal_mask is None:
        normal = np.ones_like(alarm, dtype=bool)
    else:
        normal = np.asarray(normal_mask, dtype=bool).ravel()
        if normal.shape != alarm.shape:
            raise ValueError("normal_mask length mismatch")
    out = np.empty(alarm.size, dtype=int)
    value = int(initial)
    for i, (flag, is_normal) in enumerate(zip(alarm, normal)):
        delta = 1 if is_normal and flag else (-1 if is_normal and not flag else 0)
        value = max(0, min(int(maximum), value + delta))
        out[i] = value
    return out


def event_window_mask(timestamps, event_start, event_end) -> np.ndarray:
    ts = np.asarray(timestamps).astype("datetime64[ns]")
    if event_start is None or event_end is None:
        return np.zeros(ts.shape[0], dtype=bool)
    start = np.datetime64(event_start, "ns")
    end = np.datetime64(event_end, "ns")
    return (ts >= start) & (ts <= end)


def evaluate_wind_event(
    *,
    event_id: int,
    is_anomaly_event: bool,
    alarm_mask,
    timestamps=None,
    event_start=None,
    event_end=None,
    normal_mask=None,
    criticality_threshold: int = 72,
    sample_period_minutes: float = 10.0,
) -> WindEventEvaluation:
    alarm = np.asarray(alarm_mask, dtype=bool).ravel()
    normal = np.ones_like(alarm) if normal_mask is None else np.asarray(normal_mask, dtype=bool).ravel()
    if normal.shape != alarm.shape:
        raise ValueError("normal_mask length mismatch")

    criticality = calculate_criticality(alarm, normal)
    max_criticality = int(np.max(criticality)) if criticality.size else 0
    detected = bool(max_criticality >= int(criticality_threshold))
    alarm_fraction = float(np.mean(alarm[normal])) if np.any(normal) else 0.0

    window_recall = None
    window_precision = None
    first_detection = None
    lead_samples = None
    lead_minutes = None

    if timestamps is not None and event_start is not None and event_end is not None:
        window = event_window_mask(timestamps, event_start, event_end)
        evaluable = normal.copy()
        truth = window & bool(is_anomaly_event)
        if np.any(evaluable):
            y_true = truth[evaluable].astype(int)
            y_pred = alarm[evaluable].astype(int)
            if is_anomaly_event:
                window_recall = float(recall_score(y_true, y_pred, zero_division=0))
                window_precision = float(precision_score(y_true, y_pred, zero_division=0))

        threshold_hits = np.flatnonzero(criticality >= int(criticality_threshold))
        if threshold_hits.size:
            first_detection = int(threshold_hits[0])
            if is_anomaly_event:
                ts = np.asarray(timestamps).astype("datetime64[ns]")
                end = np.datetime64(event_end, "ns")
                detected_time = ts[first_detection]
                delta_minutes = float((end - detected_time) / np.timedelta64(1, "m"))
                lead_minutes = delta_minutes
                lead_samples = int(round(delta_minutes / float(sample_period_minutes)))

    return WindEventEvaluation(
        event_id=int(event_id),
        is_anomaly_event=bool(is_anomaly_event),
        event_detected=detected,
        max_criticality=max_criticality,
        alarm_fraction=alarm_fraction,
        event_window_recall=window_recall,
        event_window_precision=window_precision,
        first_detection_index=first_detection,
        lead_time_samples=lead_samples,
        lead_time_minutes=lead_minutes,
    )


def summarize_wind_events(rows: Iterable[WindEventEvaluation | dict]) -> dict:
    records = [asdict(row) if isinstance(row, WindEventEvaluation) else dict(row) for row in rows]
    if not records:
        return {
            "event_count": 0,
            "event_recall": None,
            "event_precision": None,
            "normal_event_false_alarm_rate": None,
            "event_f1": None,
            "mean_lead_time_minutes": None,
            "mean_max_criticality": None,
        }

    truth = np.asarray([bool(row["is_anomaly_event"]) for row in records], dtype=bool)
    pred = np.asarray([bool(row["event_detected"]) for row in records], dtype=bool)
    normal = ~truth
    leads = [
        float(row["lead_time_minutes"])
        for row in records
        if row.get("lead_time_minutes") is not None and bool(row["is_anomaly_event"])
    ]
    coverage = [
        float(row["event_window_recall"])
        for row in records
        if row.get("event_window_recall") is not None and bool(row["is_anomaly_event"])
    ]
    max_criticality = [float(row.get("max_criticality", 0.0)) for row in records]
    anomaly_criticality = [
        float(row.get("max_criticality", 0.0))
        for row in records
        if bool(row["is_anomaly_event"])
    ]
    normal_criticality = [
        float(row.get("max_criticality", 0.0))
        for row in records
        if not bool(row["is_anomaly_event"])
    ]
    return {
        "event_count": len(records),
        "anomaly_event_count": int(np.sum(truth)),
        "normal_event_count": int(np.sum(normal)),
        "event_recall": float(recall_score(truth, pred, zero_division=0)),
        "event_precision": float(precision_score(truth, pred, zero_division=0)),
        "event_f1": float(f1_score(truth, pred, zero_division=0)),
        "normal_event_false_alarm_rate": float(np.mean(pred[normal])) if np.any(normal) else None,
        "mean_event_window_recall": float(np.mean(coverage)) if coverage else None,
        "mean_lead_time_minutes": float(np.mean(leads)) if leads else None,
        "median_lead_time_minutes": float(np.median(leads)) if leads else None,
        "mean_max_criticality": float(np.mean(max_criticality)) if max_criticality else None,
        "mean_anomaly_event_criticality": float(np.mean(anomaly_criticality)) if anomaly_criticality else None,
        "mean_normal_event_criticality": float(np.mean(normal_criticality)) if normal_criticality else None,
    }
