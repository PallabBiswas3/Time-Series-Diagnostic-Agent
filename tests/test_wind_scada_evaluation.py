import numpy as np

from tsdiag.evaluation.wind_scada import (
    calculate_criticality,
    evaluate_wind_event,
    summarize_wind_events,
)


def test_criticality_increases_and_decays_only_on_normal_operation():
    alarm = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
    normal = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    assert calculate_criticality(alarm, normal).tolist() == [1, 2, 1, 1, 1, 0]


def test_event_evaluation_reports_early_detection():
    timestamps = np.arange(
        np.datetime64("2024-01-01T00:00"),
        np.datetime64("2024-01-01T03:20"),
        np.timedelta64(10, "m"),
    )
    alarm = np.zeros(len(timestamps), dtype=bool)
    alarm[5:15] = True
    result = evaluate_wind_event(
        event_id=4,
        is_anomaly_event=True,
        alarm_mask=alarm,
        timestamps=timestamps,
        event_start=np.datetime64("2024-01-01T01:00"),
        event_end=np.datetime64("2024-01-01T02:30"),
        criticality_threshold=3,
        sample_period_minutes=10,
    )
    assert result.event_detected is True
    assert result.first_detection_index == 7
    assert result.lead_time_minutes == 80.0
    assert result.event_window_recall is not None


def test_event_summary_separates_recall_and_false_alarms():
    rows = [
        {"event_id": 1, "is_anomaly_event": True, "event_detected": True, "lead_time_minutes": 60.0, "event_window_recall": 0.8},
        {"event_id": 2, "is_anomaly_event": True, "event_detected": False, "lead_time_minutes": None, "event_window_recall": 0.2},
        {"event_id": 3, "is_anomaly_event": False, "event_detected": False, "lead_time_minutes": None, "event_window_recall": None},
        {"event_id": 4, "is_anomaly_event": False, "event_detected": True, "lead_time_minutes": None, "event_window_recall": None},
    ]
    summary = summarize_wind_events(rows)
    assert summary["event_recall"] == 0.5
    assert summary["normal_event_false_alarm_rate"] == 0.5
    assert summary["mean_event_window_recall"] == 0.5
