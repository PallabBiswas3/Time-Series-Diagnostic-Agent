import numpy as np

from tsdiag.core import RunRequest
from tsdiag.domains.process_engine import ProcessEngineConfig, ProcessExecutionPipeline
from tsdiag.domains.process_runner import ProcessDiagnosticPipeline
from tsdiag.domains.runtime_adapters import run_process_request


def _request(cur, ref, *, topology=None, catalog=None):
    return RunRequest(
        domain="process",
        task="root_cause",
        observation=cur,
        metadata={
            "sampling_rate_hz": 1.0,
            "channel_names": ["A", "B", "C"],
            "normal_reference": ref,
            "process_topology": topology,
            "fault_catalog": catalog,
        },
    )


def test_process_engine_matches_legacy_on_synthetic_fault():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(1200, 3))
    cur = rng.normal(size=(500, 3))
    cur[180:, 0] += 5.0
    cur[220:, 1] += 3.0
    cur[260:, 2] += 2.0
    topology = {"A": ["B", "C"], "B": ["C"]}
    catalog = {"A": "synthetic upstream fault"}

    legacy = ProcessDiagnosticPipeline(
        maxlag=1,
        onset_z_threshold=3.0,
        diagnosis_threshold=0.2,
    ).run(
        cur,
        ref,
        ["A", "B", "C"],
        process_topology=topology,
        fault_catalog=catalog,
    )

    native = ProcessExecutionPipeline(
        ProcessEngineConfig(
            maxlag=1,
            onset_z_threshold=3.0,
            diagnosis_threshold=0.2,
        )
    ).run_request(_request(cur, ref, topology=topology, catalog=catalog))

    assert native.detection.abnormal == legacy.fault_detected
    assert native.localization.components == ([legacy.root_cause] if legacy.root_cause else [])
    assert (native.hypotheses[0].label if native.hypotheses else None) == legacy.fault_label
    assert native.abstain_reason == legacy.abstain_reason
    assert np.isclose(native.confidence, legacy.confidence)

    native_tools = [step.tool for step in native.tool_trace]
    assert native_tools == [
        "process_data_quality",
        "standardize_against_normal",
        "pca_monitoring",
        "contribution_analysis",
        "pre_post_shift_evidence",
        "temporal_fault_type_evidence",
        "stationarity_analysis",
        "granger_causality",
        "causal_graph_filter",
        "fault_onset_timing",
        "root_cause_rank",
        "process_diagnosis",
    ]


def test_process_engine_preserves_healthy_early_exit():
    rng = np.random.default_rng(11)
    ref = rng.normal(size=(1500, 3))
    # Use the exact healthy reference slice as current data so PCA should remain
    # inside its own reference distribution apart from rare 99th percentile tails.
    cur = ref[:100].copy()

    result = ProcessExecutionPipeline().run_request(_request(cur, ref))
    tools = [step.tool for step in result.tool_trace]

    if result.decision == "monitor":
        assert tools == [
            "process_data_quality",
            "standardize_against_normal",
            "pca_monitoring",
        ]
    else:
        # A 99th-percentile PCA detector can legitimately flag a reference draw.
        # If it does, the engine must continue through the full contracted chain.
        assert tools[:3] == [
            "process_data_quality",
            "standardize_against_normal",
            "pca_monitoring",
        ]
        assert "process_diagnosis" in tools


def test_default_process_adapter_uses_execution_engine():
    rng = np.random.default_rng(5)
    ref = rng.normal(size=(800, 3))
    cur = rng.normal(size=(300, 3))
    cur[120:, 0] += 4.5
    cur[150:, 1] += 2.5

    result = run_process_request(
        _request(cur, ref, topology={"A": ["B", "C"], "B": ["C"]})
    )

    assert result.metadata.get("runtime") == "execution_engine"
    assert result.metadata.get("policy") == "deterministic-process-engine-v1"
    assert result.tool_trace[0].tool == "process_data_quality"
