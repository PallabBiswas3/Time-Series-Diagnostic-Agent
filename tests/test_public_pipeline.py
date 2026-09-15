import numpy as np
import json

from tsdiag import PIPELINE_VERSION, DiagnosticPipeline, DiagnosticResult, diagnose, get_domain_pack
from tsdiag.domains import default_domain_tool_registry
from tsdiag.execution import ExecutionTrace


def _trace_names(result):
    trace = result.tool_trace
    if isinstance(trace, ExecutionTrace):
        return [step.tool for step in trace.flatten()]
    rows = []
    def visit(step):
        rows.append(step.tool)
        for child in getattr(step, "children", []): visit(child)
    for step in trace: visit(step)
    return rows


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
    assert all(step.duration_seconds is not None for step in result.tool_trace)
    assert result.tool_trace[-1].tool == "bearing_evidence_fusion"


def test_process_uses_decomposed_cross_domain_workflow():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(180, 3))
    cur = rng.normal(size=(100, 3))
    result = diagnose("process", signal_matrix=cur, normal_reference=ref,
                      channel_names=["pressure", "flow", "level"], sampling_rate_hz=1.0, maxlag=1)
    assert isinstance(result, DiagnosticResult)
    assert result.domain == "process"
    assert result.metadata["pipeline_version"] == PIPELINE_VERSION
    assert all(step.duration_seconds is not None for step in result.tool_trace)
    assert result.tool_trace[0].tool == "standardize_against_normal"
    assert result.tool_trace[-1].tool == "process_diagnosis"
    assert result.metadata["workflow_version"] == "3.0"
    assert result.metadata["policy_version"] == "process-policy-v3"


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
    assert result.metadata["workflow_version"] == "2.0"
    assert result.metadata["policy_version"] == "battery-pack-policy-v2"


def test_turbofan_pipeline_returns_health_and_prognosis():
    rng = np.random.default_rng(7)
    cycles = np.arange(1, 121)
    signal = np.column_stack([0.02 * cycles + rng.normal(scale=.05, size=120), rng.normal(size=120)])
    result = diagnose("turbofan", signal_matrix=signal, channel_names=["temperature", "noise"], cycle_index=cycles)
    assert result.decision in {"diagnose", "monitor"}
    assert result.prognosis is not None
    assert result.localization.channels
    assert result.tool_trace[0].tool == "turbofan_analysis"
    assert [step.tool for step in result.tool_trace[0].children] == list(get_domain_pack("turbofan").tool_names())


def test_transformer_pipeline_uses_decomposed_workflow():
    fs = 4000.0
    t = np.arange(0, 1, 1/fs)
    base = np.sin(2*np.pi*300*t)
    impulses = np.zeros_like(t)
    impulses[::100] = 8
    signal = np.column_stack([base + impulses, .8*base + impulses])
    result = diagnose("transformer", signal_matrix=signal, sampling_rate_hz=fs, sensor_positions=["tank_a", "tank_b"])
    assert result.domain == "transformer"
    assert result.decision in {"monitor", "abstain"}
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("transformer").tool_names())
    assert result.tool_trace[-1].tool == "transformer_decision"
    assert result.metadata["workflow_version"] == "2.0"
    assert result.metadata["policy_version"] == "transformer-policy-v2"


def test_dispatcher_exposes_current_version():
    assert DiagnosticPipeline.version == PIPELINE_VERSION == "1.1.0"


def test_unsupported_domain_task_returns_structured_abstention():
    result = diagnose("battery", task="remaining_useful_life", cell_ids=["c1"], timestamps=np.arange(10))
    assert result.decision == "abstain"
    assert "Unsupported task" in result.abstain_reason


def test_every_domain_contract_has_a_registered_callable():
    registry = default_domain_tool_registry()
    for domain in ("bearing", "process", "wind_scada", "battery", "turbofan", "transformer"):
        assert set(get_domain_pack(domain).tool_names()) <= set(registry.names(domain))


def test_new_domain_trace_steps_are_timed_and_json_safe():
    rng = np.random.default_rng(30)
    result = diagnose("battery", cell_voltage=3.7+rng.normal(scale=.003,size=(40,4)),
                      cell_temperature=30+rng.normal(scale=.1,size=(40,4)),
                      cell_ids=["a","b","c","d"], timestamps=np.arange(40))
    assert all(step.duration_seconds is not None and step.duration_seconds >= 0 for step in result.tool_trace)
    assert [step.tool for step in result.tool_trace] == list(get_domain_pack("battery").tool_names())
    json.loads(result.to_json())


def test_step_failure_is_isolated_with_failed_step_trace():
    def broken_model(**kwargs):
        raise RuntimeError("model unavailable")
    result = diagnose("wind_scada", signal_matrix=np.ones((40,2)), channel_names=["a","b"],
                      timestamps=np.arange(40), physics_model=broken_model)
    assert result.decision == "abstain"
    assert result.tool_trace[-1].tool == "normal_behavior_model"
    assert result.tool_trace[-1].status == "error"
    assert result.tool_trace[-1].duration_seconds is not None
