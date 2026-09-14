from __future__ import annotations

from typing import Any

from .models import DiagnosticResult


RESULT_CONTRACT_VERSION = "1.0"


def standardize_result(
    result: DiagnosticResult,
    *,
    pipeline_version: str = "1.0.0",
) -> DiagnosticResult:
    """Apply common cross-domain reporting semantics without changing science.

    Domain pipelines remain responsible for detection, localization, hypotheses,
    verification and prognosis. This function only normalizes the public result
    envelope so downstream evaluation, orchestration and agent policies can rely
    on the same metadata/evidence structure for every domain.
    """
    result.schema_version = RESULT_CONTRACT_VERSION

    if result.uncertainty is None:
        result.uncertainty = max(0.0, min(1.0, 1.0 - float(result.confidence)))

    if result.decision == "abstain":
        result.abstained = True
    elif result.abstained:
        result.decision = "abstain"

    for index, evidence in enumerate(result.evidence):
        if not evidence.evidence_id:
            evidence.evidence_id = f"{result.domain}-evidence-{index + 1}"
        evidence.provenance.setdefault("domain", result.domain)
        evidence.provenance.setdefault("task", result.task)
        evidence.provenance.setdefault("source", evidence.source)

    executed_tools = [step.tool for step in result.tool_trace if step.status == "ok"]
    failed_tools = [step.tool for step in result.tool_trace if step.status == "error"]
    skipped_tools = [step.tool for step in result.tool_trace if step.status == "skipped"]

    result.metadata.setdefault("pipeline_version", pipeline_version)
    result.metadata["contract"] = {
        "name": "DiagnosticResult",
        "schema_version": RESULT_CONTRACT_VERSION,
    }
    result.metadata["outcome"] = {
        "decision": result.decision,
        "abstained": bool(result.abstained),
        "abstain_reason": result.abstain_reason,
        "confidence": float(result.confidence),
        "uncertainty": None if result.uncertainty is None else float(result.uncertainty),
    }
    result.metadata["evidence_summary"] = {
        "count": len(result.evidence),
        "kinds": sorted({row.kind for row in result.evidence}),
        "sources": sorted({row.source for row in result.evidence}),
        "verification_count": len(result.verification),
    }
    result.metadata["execution_summary"] = {
        "executed_tools": executed_tools,
        "failed_tools": failed_tools,
        "skipped_tools": skipped_tools,
        "tool_count": len(result.tool_trace),
    }
    result.metadata["result_summary"] = {
        "abnormal": result.detection.abnormal,
        "detection_method": result.detection.method,
        "top_hypothesis": result.hypotheses[0].label if result.hypotheses else None,
        "localized_components": list(result.localization.components),
        "localized_channels": list(result.localization.channels),
        "has_prognosis": result.prognosis is not None,
    }

    result.validate()
    return result


def result_summary(result: DiagnosticResult) -> dict[str, Any]:
    """Compact JSON-safe summary for demos, benchmark tables and dashboards."""
    standardized = standardize_result(result)
    return {
        "domain": standardized.domain,
        "task": standardized.task,
        "decision": standardized.decision,
        "abnormal": standardized.detection.abnormal,
        "label": standardized.hypotheses[0].label if standardized.hypotheses else None,
        "confidence": float(standardized.confidence),
        "uncertainty": standardized.uncertainty,
        "abstained": standardized.abstained,
        "abstain_reason": standardized.abstain_reason,
        "components": list(standardized.localization.components),
        "channels": list(standardized.localization.channels),
        "evidence_count": len(standardized.evidence),
        "verification_count": len(standardized.verification),
        "tool_count": len(standardized.tool_trace),
        "prognosis": None if standardized.prognosis is None else standardized.prognosis.details,
    }
