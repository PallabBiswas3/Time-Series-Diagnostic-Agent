from __future__ import annotations

from types import SimpleNamespace

from tsdiag.core import RunContext, RunRequest
from tsdiag.domains.runtime_adapters import run_bearing_request, run_process_request
from tsdiag.models import DetectionResult, DiagnosticHypothesis, DiagnosticResult
from tsdiag.verification import DataQualityVerifier, OODVerifier, PhysicalBoundsVerifier


def test_bearing_adapter_abstains_when_sampling_rate_is_missing():
    result = run_bearing_request(RunRequest(domain="bearing", task="fault_diagnosis", observation=[1.0, 2.0]))
    assert result.abstained
    assert "sampling_rate_hz" in result.abstain_reason


def test_process_adapter_converts_legacy_result_to_diagnostic_result():
    class FakePipeline:
        def run(self, *args, **kwargs):
            return SimpleNamespace(
                fault_detected=True,
                fault_label="fault-1",
                root_cause="XMEAS(1)",
                affected_variables=["XMEAS(2)"],
                propagation_paths=[["XMEAS(1)", "XMEAS(2)"]],
                confidence=0.8,
                abstain_reason=None,
                tool_trace=["pca_monitoring", "root_cause_rank_enhanced"],
                artifacts={"pca_monitoring": {}, "root_cause_rank": {}},
            )

    request = RunRequest(
        domain="process",
        task="root_cause",
        observation=[[0.0, 1.0], [1.0, 2.0]],
        metadata={
            "sampling_rate_hz": 1.0,
            "channel_names": ["XMEAS(1)", "XMEAS(2)"],
            "normal_reference": [[0.0, 0.0], [0.1, 0.1]],
        },
    )
    result = run_process_request(request, pipeline=FakePipeline())
    assert result.decision == "diagnose"
    assert result.localization.components == ["XMEAS(1)"]
    assert result.hypotheses[0].label == "fault-1"
    assert result.evidence[0].evidence_id == "process-root-cause"


def test_bearing_adapter_preserves_pipeline_result():
    expected = DiagnosticResult(
        domain="bearing",
        task="fault_diagnosis",
        decision="monitor",
        detection=DetectionResult(abnormal=False),
        confidence=0.7,
    )

    class FakePipeline:
        def run(self, *args, **kwargs):
            return expected

    result = run_bearing_request(
        RunRequest(domain="bearing", task="fault_diagnosis", observation=[1.0], metadata={"sampling_rate_hz": 1000.0}),
        pipeline=FakePipeline(),
    )
    assert result is expected


def test_standard_verifiers_cover_quality_ood_and_physical_bounds():
    hypothesis = DiagnosticHypothesis(label="fault", score=0.8)
    context = RunContext.from_request(
        RunRequest(
            domain="bearing",
            task="fault_diagnosis",
            observation=[1.0],
            metadata={
                "quality_flags": ["minor_clipping"],
                "ood_score": 0.2,
                "temperature_c": 80.0,
                "physical_bounds": {"temperature_c": (-40.0, 180.0)},
            },
        )
    )
    assert DataQualityVerifier().verify(hypothesis, context).status == "SUPPORTED"
    assert OODVerifier().verify(hypothesis, context).status == "SUPPORTED"
    assert PhysicalBoundsVerifier().verify(hypothesis, context).status == "SUPPORTED"


def test_ood_verifier_contradicts_out_of_regime_case():
    hypothesis = DiagnosticHypothesis(label="fault", score=0.8)
    context = RunContext.from_request(
        RunRequest(domain="process", task="fault_diagnosis", observation=[1.0], metadata={"ood_score": 0.95})
    )
    assert OODVerifier(threshold=0.8).verify(hypothesis, context).status == "CONTRADICTED"
