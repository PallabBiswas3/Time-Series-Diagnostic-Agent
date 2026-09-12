from __future__ import annotations

import numpy as np

from tsdiag.domains.battery_runner import BatteryPrognosticPipeline
from tsdiag.tools.battery import capacity_health_features, estimate_capacity_rul


def test_capacity_features_detect_degrading_slope_and_knee_candidate():
    x = np.arange(1, 121, dtype=float)
    y = 2.0 - 0.0015 * x
    y[x >= 70] -= 0.0025 * (x[x >= 70] - 69)
    features = capacity_health_features(x, y)
    assert features["local_slope_ah_per_cycle"] < 0.0
    assert features["global_slope_ah_per_cycle"] < 0.0
    assert features["knee_cycle"] is not None


def test_rul_estimator_abstains_on_too_short_or_non_degrading_history():
    short = estimate_capacity_rul(np.arange(1, 10), np.linspace(2.0, 1.98, 9))
    assert short["abstained"] is True

    flat_x = np.arange(1, 50)
    flat_y = np.full_like(flat_x, 1.95, dtype=float)
    flat = estimate_capacity_rul(flat_x, flat_y)
    assert flat["abstained"] is True
    assert flat["remaining_useful_life_cycles"] is None


def test_pipeline_uses_only_prefix_and_returns_structured_prognosis():
    x = np.arange(1, 101, dtype=float)
    y = 2.0 - 0.0045 * x
    result = BatteryPrognosticPipeline().run(x, y, battery_id="synthetic")
    assert result.domain == "battery"
    assert result.task == "prognosis"
    assert result.prognosis is not None
    assert result.prognosis.remaining_useful_life is not None
    assert result.to_dict()["schema_version"] == "1.0"


def test_future_values_do_not_change_prefix_prediction():
    full_x = np.arange(1, 151, dtype=float)
    full_y = 2.0 - 0.0035 * full_x
    prefix_x = full_x[:80]
    prefix_y = full_y[:80]

    first = BatteryPrognosticPipeline().run(prefix_x, prefix_y, battery_id="cell")
    altered_future = full_y.copy()
    altered_future[80:] = np.linspace(1.9, 0.5, altered_future.size - 80)
    second = BatteryPrognosticPipeline().run(full_x[:80], altered_future[:80], battery_id="cell")

    assert first.prognosis is not None and second.prognosis is not None
    assert first.prognosis.remaining_useful_life == second.prognosis.remaining_useful_life
    assert first.prognosis.details["predicted_eol_cycle"] == second.prognosis.details["predicted_eol_cycle"]
