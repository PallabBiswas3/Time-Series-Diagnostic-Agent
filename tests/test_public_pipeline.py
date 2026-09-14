import numpy as np

from tsdiag import PIPELINE_VERSION, DiagnosticPipeline, DiagnosticResult, diagnose, get_domain_pack


def test_missing_required_metadata_returns_structured_abstention():
    result = diagnose("bearing", signal=np.ones(128))
    assert isinstance(result, DiagnosticResult)
    assert result.decision == "abstain"
    assert "sampling_rate_hz" in result.abstain_reason


def test_bearing_runs_through_public_pipeline():
    fs = 8000.0
    t = np.arange(0, 1.5, 1 / fs)
    x = (1 + 0.8 * np.sin(2 * np.pi * 80 * t)) * np.sin(2 * np.pi * 1800 * t)
    result = diagnose("bearing", signal=x, sampling_rate_hz=fs,
                      fault_frequencies={"BPFO": 80.0, "BPFI": 125.0, "BSF": 55.0, "FTF": 10.0})
    assert result.domain == "bearing"
    assert result.decision in {"diagnose", "abstain"}
    assert result.tool_trace


def test_process_adapter_returns_cross_domain_contract():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(180, 3))
    cur = rng.normal(size=(100, 3))
    result = diagnose("process", signal_matrix=cur, normal_reference=ref,
                      channel_names=["pressure", "flow", "level"], sampling_rate_hz=1.0, maxlag=1)
    assert isinstance(result, DiagnosticResult)
    assert result.domain == "process"
    assert result.metadata["pipeline_version"] == PIPELINE_VERSION


def test_wind_scada_pipeline_localizes_persistent_shift():
    rng = np.random.default_rng(5)
    ref = rng.normal(scale=0.2, size=(80, 3))
    cur = rng.normal(scale=0.2, size=(100, 3))
    cur[40:, 1] += 2.0
    result = diagnose("wind_scada", signal_matrix=cur, normal_reference=ref,
                      channel_names=["power", "gearbox_temp", "wind"], timestamps=np.arange(100))
    assert result.decision == "diagnose"
    assert result.localization.channels == ["gearbox_temp"]
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("wind_scada").tool_names())


def test_battery_pipeline_localizes_outlying_cell():
    rng = np.random.default_rng(6)
    voltage = 3.7 + rng.normal(scale=0.003, size=(80, 5))
    temperature = 30 + rng.normal(scale=0.1, size=(80, 5))
    voltage[:, 3] -= 0.12
    result = diagnose("battery", cell_voltage=voltage, cell_temperature=temperature,
                      cell_ids=[f"cell_{i}" for i in range(5)], timestamps=np.arange(80), cell_anomaly_threshold=1.5)
    assert result.decision == "diagnose"
    assert result.localization.components == ["cell_3"]
    assert result.prognosis is not None
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("battery").tool_names())


def test_turbofan_pipeline_returns_health_and_prognosis():
    rng = np.random.default_rng(7)
    cycles = np.arange(1, 121)
    signal = np.column_stack([0.02 * cycles + rng.normal(scale=.05, size=120), rng.normal(size=120)])
    result = diagnose("turbofan", signal_matrix=signal, channel_names=["temperature", "noise"], cycle_index=cycles)
    assert result.decision in {"diagnose", "monitor"}
    assert result.prognosis is not None
    assert result.localization.channels
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("turbofan").tool_names())


def test_transformer_pipeline_completes_and_abstains_without_classifier_when_abnormal():
    fs = 4000.0
    t = np.arange(0, 1, 1/fs)
    base = np.sin(2*np.pi*300*t)
    impulses = np.zeros_like(t)
    impulses[::100] = 8
    signal = np.column_stack([base + impulses, .8*base + impulses])
    result = diagnose("transformer", signal_matrix=signal, sampling_rate_hz=fs, sensor_positions=["tank_a", "tank_b"])
    assert result.domain == "transformer"
    assert result.decision in {"monitor", "abstain"}
    assert result.tool_trace[-1].tool == "transformer_decision"
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("transformer").tool_names())


def test_dispatcher_exposes_frozen_version():
    assert DiagnosticPipeline.version == "1.0.0"


def test_unsupported_domain_task_returns_structured_abstention():
    result = diagnose("battery", task="remaining_useful_life", cell_ids=["c1"], timestamps=np.arange(10))
    assert result.decision == "abstain"
    assert "Unsupported task" in result.abstain_reason
