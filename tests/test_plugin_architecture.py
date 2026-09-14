from types import SimpleNamespace

import numpy as np

from tsdiag.contracts import DiagnosticRequest
from tsdiag.execution import ExecutionTrace, Step, Workflow, WorkflowExecutor
from tsdiag.models import ToolTraceStep
from tsdiag.pipeline import diagnose
from tsdiag.registry import domain_registry, model_registry, policy_registry


def test_execution_trace_preserves_order_and_named_lookup():
    trace = ExecutionTrace([
        ToolTraceStep("analysis", status="ok"),
        ToolTraceStep("verification", status="warning"),
        ToolTraceStep("analysis", status="ok"),
    ])
    assert [step.tool for step in trace.steps] == ["analysis", "verification", "analysis"]
    assert trace.get("analysis") is trace.steps[-1]
    assert trace.get("analysis", occurrence=0) is trace.steps[0]
    assert trace.require("verification").status == "warning"


def test_workflow_executor_respects_dependencies_and_named_trace():
    workflow = Workflow(steps=(
        Step("first", lambda state: {"x": 2}),
        Step("second", lambda state: {"y": state["x"] + 3}, depends_on=("first",)),
    ))
    state, trace = WorkflowExecutor().run(workflow, {})
    assert state["y"] == 5
    assert trace.require("first").status == "ok"
    assert trace.require("second").outputs_summary["output_keys"] == ["y"]


def test_structured_wind_request_uses_plugin_policy(monkeypatch):
    import tsdiag.domains.wind_scada_plugin as plugin_module

    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, train_matrix, prediction_matrix, channel_names, **kwargs):
            n = len(prediction_matrix)
            residual = np.zeros(n, dtype=bool)
            drift = np.zeros(n, dtype=bool)
            residual[2:8] = True
            drift[2:8] = True
            return SimpleNamespace(
                affected_channels=[channel_names[-1]],
                confidence=0.8,
                alarm_mask=residual | drift,
                anomaly_scores=np.ones(n),
                tool_trace=["legacy-internal-step"],
                artifacts={
                    "anomaly_detection": {"alarm_mask": residual},
                    "residual_changepoint": {"alarm_mask": drift, "change_points": [2]},
                    "regime_assignment": {"out_of_distribution_mask": np.zeros(n, dtype=bool)},
                    "physics_consistency": {"verification_findings": [{"channel": channel_names[-1]}]},
                    "fusion": {"fused_alarm_fraction": float(np.mean(residual | drift))},
                },
            )

    monkeypatch.setattr(plugin_module, "WindScadaDiagnosticPipeline", FakePipeline)

    result = diagnose(DiagnosticRequest(
        domain="wind_scada",
        task="condition_monitoring",
        inputs={
            "train_matrix": np.ones((80, 3)),
            "prediction_matrix": np.ones((20, 3)),
            "channel_names": ["wind_speed", "power", "temperature"],
            "evaluable_mask": np.ones(20, dtype=bool),
            "event_minimum_corroborated_run": 6,
        },
    ))

    assert result.decision == "diagnose"
    assert result.detection.abnormal is True
    assert result.uncertainty is None
    assert result.metadata["workflow_version"] == "1.0"
    assert result.metadata["policy_version"] == "1.0"
    assert result.metadata["model_version"] == "wind-nbm-regime-v1"
    assert result.tool_trace.require("event_decision").evidence_ids == ["wind-event-evidence"]


def test_all_six_domain_plugins_are_registered_after_diagnose():
    diagnose(DiagnosticRequest(domain="bearing", task="fault_diagnosis", inputs={}))
    assert domain_registry.names() == (
        "battery", "bearing", "process", "transformer", "turbofan", "wind_scada"
    )
    for domain in domain_registry.names():
        assert "default" in policy_registry.refs(domain)


def test_versioned_policy_ref_and_model_version_provenance():
    rng = np.random.default_rng(91)
    voltage = 3.7 + rng.normal(scale=0.003, size=(40, 4))
    temperature = 30 + rng.normal(scale=0.1, size=(40, 4))
    model_registry.register(
        "battery-health-test",
        version="test-v1",
        metadata={"purpose": "unit-test provenance only"},
        replace=True,
    )

    result = diagnose(DiagnosticRequest(
        domain="battery",
        task="anomaly_localization",
        policy_ref="compat-1.0",
        model_refs={"health_model": "battery-health-test"},
        inputs={
            "cell_voltage": voltage,
            "cell_temperature": temperature,
            "cell_ids": ["a", "b", "c", "d"],
            "timestamps": np.arange(40),
        },
    ))

    assert result.decision in {"diagnose", "monitor"}
    # Battery now has a task-aware plugin workflow (pack diagnosis + prognosis),
    # while this request intentionally selects the frozen compatibility policy.
    assert result.metadata["workflow_version"] == "1.0"
    assert result.metadata["policy_version"] == "compat-1.0"
    assert result.metadata["policy_ref"] == "compat-1.0"
    assert result.metadata["resolved_model_versions"] == {"health_model": "test-v1"}
    assert result.metadata["compatibility_adapter"] is True
    assert result.metadata["outer_workflow_trace"][0]["step"] == "battery_analysis"
    assert len(result.tool_trace) > 1


def test_structured_battery_prognosis_uses_task_specific_policy():
    cycles = np.arange(1, 61, dtype=float)
    capacity = 2.0 - 0.006 * cycles
    result = diagnose(DiagnosticRequest(
        domain="battery",
        task="prognosis",
        policy_ref="capacity-prognosis-policy-v1",
        inputs={
            "cycle_index": cycles,
            "capacity_ah": capacity,
            "battery_id": "fixture",
            "eol_capacity_ah": 1.4,
        },
    ))

    assert result.task == "prognosis"
    assert result.prognosis is not None
    assert result.metadata["workflow_version"] == "1.0"
    assert result.metadata["policy_version"] == "capacity-prognosis-policy-v1"
    assert result.metadata["outer_workflow_trace"][0]["step"] == "battery_prognosis"
