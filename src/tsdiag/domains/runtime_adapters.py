from __future__ import annotations

from typing import Any

import numpy as np

from ..core.context import RunRequest
from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    ToolTraceStep,
)
from .bearing_runner import BearingDiagnosticPipeline
from .process_runner import ProcessDiagnosticPipeline


def run_bearing_request(
    request: RunRequest,
    *,
    pipeline: BearingDiagnosticPipeline | None = None,
) -> DiagnosticResult:
    """Bridge a canonical RunRequest to the existing bearing pipeline."""

    request.validate()
    if request.domain != "bearing":
        raise ValueError("run_bearing_request requires domain='bearing'")

    metadata = dict(request.metadata)
    fs = metadata.get("sampling_rate_hz")
    if fs is None:
        return _metadata_abstention(request, "sampling_rate_hz")

    fault_frequencies = {
        key: float(metadata[key])
        for key in ("BPFO", "BPFI", "BSF", "FTF")
        if metadata.get(key) is not None
    }
    pipeline = pipeline or BearingDiagnosticPipeline()
    return pipeline.run(
        request.observation,
        float(fs),
        fault_frequencies=fault_frequencies,
        shaft_rate_hz=metadata.get("shaft_rate_hz"),
        channel_name=str(metadata.get("channel_name", "ch0")),
        operating_condition=metadata.get("operating_condition"),
    )


def run_process_request(
    request: RunRequest,
    *,
    pipeline: ProcessDiagnosticPipeline | None = None,
) -> DiagnosticResult:
    """Bridge a canonical RunRequest to the existing process/TEP pipeline."""

    request.validate()
    if request.domain != "process":
        raise ValueError("run_process_request requires domain='process'")

    metadata = dict(request.metadata)
    for name in ("sampling_rate_hz", "channel_names", "normal_reference"):
        if metadata.get(name) is None:
            return _metadata_abstention(request, name)

    pipeline = pipeline or ProcessDiagnosticPipeline()
    legacy = pipeline.run(
        request.observation,
        metadata["normal_reference"],
        metadata["channel_names"],
        process_topology=metadata.get("process_topology"),
        fault_catalog=metadata.get("fault_catalog"),
        timestamps=metadata.get("timestamps"),
    )

    evidence: list[Evidence] = []
    if legacy.root_cause is not None:
        evidence.append(
            Evidence(
                source="process_pipeline",
                statement=f"Root-cause candidate: {legacy.root_cause}",
                score=float(np.clip(legacy.confidence, 0.0, 1.0)),
                evidence_id="process-root-cause",
                kind="causal",
                provenance={"adapter": "run_process_request"},
                details={
                    "affected_variables": legacy.affected_variables,
                    "propagation_paths": legacy.propagation_paths,
                },
            )
        )

    hypotheses: list[DiagnosticHypothesis] = []
    if legacy.fault_label is not None:
        hypotheses.append(
            DiagnosticHypothesis(
                label=legacy.fault_label,
                score=float(np.clip(legacy.confidence, 0.0, 1.0)),
                rationale="Produced by the existing deterministic process diagnostic pipeline.",
                evidence_ids=[row.evidence_id for row in evidence if row.evidence_id],
                details={"root_cause": legacy.root_cause},
            )
        )

    abstained = legacy.abstain_reason is not None
    decision = "abstain" if abstained else ("diagnose" if legacy.fault_detected else "monitor")
    result = DiagnosticResult(
        domain="process",
        task=request.task,
        decision=decision,
        detection=DetectionResult(
            abnormal=legacy.fault_detected,
            score=float(np.clip(legacy.confidence, 0.0, 1.0)),
            method="process_diagnostic_pipeline",
        ),
        localization=LocalizationResult(
            components=[legacy.root_cause] if legacy.root_cause else [],
            channels=list(legacy.affected_variables),
            details={"propagation_paths": legacy.propagation_paths},
        ),
        hypotheses=hypotheses,
        evidence=evidence,
        confidence=float(np.clip(legacy.confidence, 0.0, 1.0)),
        uncertainty=float(np.clip(1.0 - legacy.confidence, 0.0, 1.0)),
        abstained=abstained,
        abstain_reason=legacy.abstain_reason,
        tool_trace=[ToolTraceStep(tool=name) for name in legacy.tool_trace],
        metadata={
            "adapter": "process_pipeline_v1",
            "sampling_rate_hz": metadata["sampling_rate_hz"],
            "legacy_artifact_keys": sorted(legacy.artifacts),
        },
    )
    result.validate()
    return result


def _metadata_abstention(request: RunRequest, missing: str) -> DiagnosticResult:
    result = DiagnosticResult(
        domain=request.domain,
        task=request.task,
        decision="abstain",
        detection=DetectionResult(abnormal=None, method="runtime_metadata_validation"),
        confidence=0.0,
        uncertainty=1.0,
        abstained=True,
        abstain_reason=f"Missing required metadata: {missing}",
        metadata={"missing_required_metadata": [missing]},
    )
    result.validate()
    return result
