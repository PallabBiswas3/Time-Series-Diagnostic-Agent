from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from tsdiag.datasets.wind_care import (
    care_normal_mask,
    load_care_event,
    validate_care_layout,
)
from tsdiag.tools.wind_scada import (
    detect_wind_operating_regimes,
    fit_wind_normal_behavior_model,
    predict_wind_normal_behavior,
    scada_quality_check,
    wind_residual_anomaly_detection,
)


def _write_care_fixture(root: Path):
    for wf in ("A", "B", "C"):
        folder = root / f"Wind Farm {wf}"
        (folder / "datasets").mkdir(parents=True)
        pd.DataFrame(
            [{
                "event_id": 1 if wf == "A" else 100 + ord(wf),
                "asset_id": 0,
                "event_start": "2022-01-02",
                "event_end": "2022-01-03",
                "event_label": "anomaly" if wf == "A" else "normal",
            }]
        ).to_csv(folder / "event_info.csv", sep=";", index=False)
        pd.DataFrame(
            [
                {"sensor_name": "wind_speed", "statistics_type": "average"},
                {"sensor_name": "power", "statistics_type": "average"},
                {"sensor_name": "temp", "statistics_type": "average"},
            ]
        ).to_csv(folder / "feature_description.csv", sep=";", index=False)

    frame = pd.DataFrame(
        {
            "id": range(8),
            "train_test": ["train"] * 5 + ["prediction"] * 3,
            "time_stamp": pd.date_range("2022-01-01", periods=8, freq="10min"),
            "asset_id": [0] * 8,
            "status_type_id": [0, 0, 0, 1, 0, 0, 4, 0],
            "wind_speed": np.linspace(4, 11, 8),
            "power": np.linspace(50, 900, 8),
            "temp": np.linspace(30, 40, 8),
        }
    )
    frame.to_csv(root / "Wind Farm A" / "datasets" / "1.csv", sep=";", index=False)


def test_care_loader_uses_train_prediction_split(tmp_path):
    _write_care_fixture(tmp_path)
    layout = validate_care_layout(tmp_path)
    assert layout == {"A": True, "B": True, "C": True}

    event = load_care_event(tmp_path, 1)
    assert event.event.is_anomaly is True
    assert len(event.train) == 5
    assert len(event.prediction) == 3
    assert {"wind_speed", "power", "temp"}.issubset(event.train.columns)
    assert care_normal_mask(event.train).tolist() == [True, True, True, False, True]


def test_care_loader_streams_official_style_zip(tmp_path):
    fixture = tmp_path / "care"
    _write_care_fixture(fixture)
    archive = Path(shutil.make_archive(str(tmp_path / "CARE_To_Compare"), "zip", fixture))
    assert validate_care_layout(archive) == {"A": True, "B": True, "C": True}
    event = load_care_event(archive, 1)
    assert event.event.is_anomaly is True
    assert len(event.train) == 5
    assert len(event.prediction) == 3


def test_scada_quality_check_detects_irregular_sampling():
    x = np.column_stack([np.arange(5.0), np.ones(5)])
    ts = np.array([
        "2022-01-01T00:00",
        "2022-01-01T00:10",
        "2022-01-01T00:20",
        "2022-01-01T00:50",
        "2022-01-01T01:00",
    ], dtype="datetime64[m]")
    result = scada_quality_check(x, ["wind", "constant"], ts)
    assert result["sampling_irregularity"] > 0
    assert any(flag.get("flag") == "constant_channel" for flag in result["quality_flags"])


def test_regime_aware_normal_behavior_flags_persistent_shift():
    rng = np.random.default_rng(4)
    n_ref = 500
    wind_ref = rng.uniform(3, 14, n_ref)
    regime_ref = (wind_ref >= 8).astype(int)
    power_ref = np.where(regime_ref == 0, 15 * wind_ref**2, 900 + 5 * wind_ref) + rng.normal(0, 15, n_ref)
    temp_ref = 25 + 0.015 * power_ref + rng.normal(0, 0.5, n_ref)
    ref = np.column_stack([wind_ref, power_ref, temp_ref])

    current = ref[:160].copy()
    current_regime = regime_ref[:160].copy()
    current[100:, 2] += 8.0

    state = fit_wind_normal_behavior_model(
        ref,
        regime_ref,
        target_indices=[2],
        predictor_indices=[0, 1, 2],
    )
    predicted = predict_wind_normal_behavior(current, current_regime, state)
    detected = wind_residual_anomaly_detection(
        predicted["normalized_residuals"],
        threshold=3.0,
        persistence=3,
    )

    assert detected["alarm_mask"][:90].mean() < 0.10
    assert detected["alarm_mask"][110:].mean() > 0.50


def test_out_of_sample_residual_scale_stays_sane_on_healthy_holdout():
    rng = np.random.default_rng(17)
    n_ref = 800
    wind_ref = rng.uniform(3.0, 14.0, n_ref)
    power_ref = 22.0 * wind_ref**2 + rng.normal(0.0, 30.0, n_ref)
    temp_ref = 20.0 + 0.012 * power_ref + np.sin(wind_ref) + rng.normal(0.0, 0.7, n_ref)
    ref = np.column_stack([wind_ref, power_ref, temp_ref])
    regimes = (wind_ref >= 8.0).astype(int)

    n_test = 250
    wind_test = rng.uniform(3.0, 14.0, n_test)
    power_test = 22.0 * wind_test**2 + rng.normal(0.0, 30.0, n_test)
    temp_test = 20.0 + 0.012 * power_test + np.sin(wind_test) + rng.normal(0.0, 0.7, n_test)
    test = np.column_stack([wind_test, power_test, temp_test])
    test_regimes = (wind_test >= 8.0).astype(int)

    state = fit_wind_normal_behavior_model(
        ref,
        regimes,
        target_indices=[2],
        predictor_indices=[0, 1, 2],
    )
    prediction = predict_wind_normal_behavior(test, test_regimes, state)
    z = np.abs(prediction["normalized_residuals"][:, 0])
    assert np.nanmedian(z) < 1.5
    assert np.mean(z >= 3.5) < 0.10


def test_oos_residual_calibration_recenters_systematic_healthy_model_bias():
    rng = np.random.default_rng(23)
    n_ref = 700
    wind = rng.uniform(3.0, 14.0, n_ref)
    power = 18.0 * wind**2 + rng.normal(0.0, 20.0, n_ref)
    # Add a nonlinear/noisy target so finite-tree prediction has a measurable
    # out-of-sample residual offset rather than assuming exact zero centering.
    temp = 24.0 + 0.01 * power + 0.5 * np.sin(1.7 * wind) + rng.normal(0.25, 0.8, n_ref)
    ref = np.column_stack([wind, power, temp])
    regimes = (wind >= 8.0).astype(int)

    state = fit_wind_normal_behavior_model(
        ref,
        regimes,
        target_indices=[2],
        predictor_indices=[0, 1, 2],
    )

    prediction = predict_wind_normal_behavior(ref, regimes, state)
    z = prediction["normalized_residuals"][:, 0]
    # Calibration should carry and subtract the healthy residual center rather
    # than letting a non-zero model bias feed a sequential drift detector.
    assert np.isfinite(state.residual_center[0])
    assert abs(np.nanmedian(z)) < 0.5


def test_operating_regime_detection_returns_requested_number():
    rng = np.random.default_rng(8)
    x = np.r_[
        rng.normal([4, 100], [0.3, 10], size=(80, 2)),
        rng.normal([8, 600], [0.3, 20], size=(80, 2)),
        rng.normal([12, 1000], [0.3, 15], size=(80, 2)),
    ]
    result = detect_wind_operating_regimes(x, driver_indices=[0, 1], n_regimes=3)
    assert len(np.unique(result["regime_ids"])) == 3
