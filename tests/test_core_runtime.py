from __future__ import annotations

from tsdiag.contracts import DataKind, DomainPack, TaskKind, ToolContract, ToolRegistry
from tsdiag.core import (
    DeterministicRouter,
    DiagnosticRuntime,
    EvidenceStore,
    ExecutionEngine,
    RunContext,
    RunRequest,
    ToolOutcome,
)
from tsdiag.models import DiagnosticHypothesis, Evidence, VerificationResult
from tsdiag.verification import VerificationSuite


def _pack() -> DomainPack:
    return DomainPack(
        key="synthetic",
        title="Synthetic",
        data_kind=DataKind.MULTIVARIATE_SERIES,
        tasks=(TaskKind.FAULT_DIAGNOSIS,),
        required_metadata=("sampling_rate_hz",),
        optional_metadata=("asset_id",),
        tools=(
            ToolContract(
                name="feature",
                purpose="Extract a feature",
                required_inputs=("observation",),
                outputs=("feature_value",),
            ),
            ToolContract(
                name="decision",
                purpose="Convert feature to evidence",
                required_inputs=("feature_value",),
                outputs=("decision_score",),
            ),
        ),
        outputs=("decision_score",),
    )


def test_run_context_keeps_observation_and_metadata_separate():
    request = RunRequest(
        domain="synthetic",
        task="fault_diagnosis",
        observation=[1.0, 2.0],
        metadata={"sampling_rate_hz": 100.0},
    )
    context = RunContext.from_request(request)
    assert context.get("observation") == [1.0, 2.0]
    assert context.get("sampling_rate_hz") == 100.0


def test_deterministic_router_preserves_domain_pack_order():
    pack = _pack()
    context = RunContext.from_request(
        RunRequest(
            domain="synthetic",
            task="fault_diagnosis",
            observation=[1.0],
            metadata={"sampling_rate_hz": 10.0},
        )
    )
    plan = DeterministicRouter().plan(pack, context)
    assert plan.ready
    assert plan.tool_sequence == ("feature", "decision")


def test_execution_engine_propagates_outputs_evidence_and_trace():
    pack = _pack()
    registry = ToolRegistry()
    registry.register("feature", lambda observation: {"feature_value": sum(observation)})
    registry.register(
        "decision",
        lambda feature_value: ToolOutcome(
            outputs={"decision_score": feature_value / 10.0},
            evidence=[Evidence(source="decision", statement="score computed", kind="model")],
        ),
    )
    context = RunContext.from_request(
        RunRequest(
            domain="synthetic",
            task="fault_diagnosis",
            observation=[2.0, 3.0],
            metadata={"sampling_rate_hz": 10.0},
        )
    )
    plan = DeterministicRouter().plan(pack, context)
    result = ExecutionEngine(registry).run(pack, plan, context)

    assert not result.abstained
    assert result.context.get("feature_value") == 5.0
    assert result.context.get("decision_score") == 0.5
    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_id == "evidence-0001"
    assert result.evidence[0].provenance["tool"] == "decision"
    assert [row.tool for row in result.trace] == ["feature", "decision"]


def test_engine_abstains_before_execution_when_metadata_is_missing():
    pack = _pack()
    context = RunContext.from_request(
        RunRequest(domain="synthetic", task="fault_diagnosis", observation=[1.0])
    )
    plan = DeterministicRouter().plan(pack, context)
    result = ExecutionEngine(ToolRegistry()).run(pack, plan, context)
    assert result.abstained
    assert "sampling_rate_hz" in result.abstain_reason


def test_evidence_store_rejects_duplicate_ids():
    store = EvidenceStore()
    store.add(Evidence(source="a", statement="one", evidence_id="fixed"))
    try:
        store.add(Evidence(source="b", statement="two", evidence_id="fixed"))
    except ValueError as exc:
        assert "duplicate evidence_id" in str(exc)
    else:
        raise AssertionError("duplicate evidence id should fail")


def test_diagnostic_runtime_uses_swappable_router(monkeypatch):
    pack = _pack()
    monkeypatch.setattr("tsdiag.core.runtime.get_domain_pack", lambda _: pack)
    registry = ToolRegistry()
    registry.register("feature", lambda observation: {"feature_value": len(observation)})
    registry.register("decision", lambda feature_value: {"decision_score": feature_value})
    runtime = DiagnosticRuntime(registry)
    result = runtime.run(
        RunRequest(
            domain="synthetic",
            task="fault_diagnosis",
            observation=[1, 2, 3],
            metadata={"sampling_rate_hz": 10.0},
        )
    )
    assert result.context.get("decision_score") == 3


class _ContradictingVerifier:
    name = "physics"

    def verify(self, hypothesis, context):
        return VerificationResult(
            hypothesis=hypothesis.label,
            status="CONTRADICTED",
            reason="violates synthetic invariant",
            verifier=self.name,
        )


def test_verification_is_independent_and_can_contradict_hypothesis():
    suite = VerificationSuite((_ContradictingVerifier(),))
    context = RunContext.from_request(
        RunRequest(domain="synthetic", task="fault_diagnosis", observation=[1])
    )
    results = suite.run([DiagnosticHypothesis(label="fault", score=0.9)], context)
    assert suite.contradicted(results)
