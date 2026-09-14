from __future__ import annotations

from typing import Any

import numpy as np

from .domains import DOMAIN_PACKS
from .domains.bearing_runner import BearingDiagnosticPipeline
from .domains.process_runner import ProcessDiagnosticPipeline, ProcessDiagnosticResult
from .domains.runners import (
    BatteryDiagnosticPipeline,
    TransformerDiagnosticPipeline,
    TurbofanDiagnosticPipeline,
    WindScadaDiagnosticPipeline,
)
from .models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    ToolTraceStep,
)

PIPELINE_VERSION = "1.0.0"


def _abstain(domain: str, task: str, reason: str) -> DiagnosticResult:
    result = DiagnosticResult(
        domain=domain,
        task=task,
        decision="abstain",
        detection=DetectionResult(abnormal=None, method="pipeline_validation"),
        confidence=0.0,
        uncertainty=1.0,
        abstained=True,
        abstain_reason=reason,
        tool_trace=[ToolTraceStep("input_validation", status="warning", details={"reason": reason})],
        metadata={"pipeline_version": PIPELINE_VERSION},
    )
    result.validate()
    return result


def _process_result(raw: ProcessDiagnosticResult) -> DiagnosticResult:
    evidence: list[Evidence] = []
    hypotheses: list[DiagnosticHypothesis] = []
    if raw.fault_detected:
        ev = Evidence(
            source="process_root_cause",
            statement=f"Detected process deviation; root-cause candidate={raw.root_cause!r}.",
            score=float(np.clip(raw.confidence, 0.0, 1.0)),
            details={"affected_variables": raw.affected_variables, "propagation_paths": raw.propagation_paths},
            evidence_id="process-diagnostic-evidence",
            kind="causal",
        )
        evidence.append(ev)
        if raw.fault_label or raw.root_cause:
            hypotheses.append(DiagnosticHypothesis(
                label=str(raw.fault_label or raw.root_cause),
                score=float(np.clip(raw.confidence, 0.0, 1.0)),
                rationale="Process monitoring, contribution, temporal and causal evidence were combined.",
                evidence_ids=[ev.evidence_id],
            ))
    abstained = bool(raw.fault_detected and raw.abstain_reason)
    decision = "abstain" if abstained else ("diagnose" if raw.fault_detected else "monitor")
    confidence = float(np.clip(raw.confidence, 0.0, 1.0))
    result = DiagnosticResult(
        domain="process", task="root_cause", decision=decision,
        detection=DetectionResult(abnormal=raw.fault_detected, score=confidence, method="pca_process_pipeline"),
        localization=LocalizationResult(
            components=[raw.root_cause] if raw.root_cause else [],
            channels=list(raw.affected_variables),
            scores={raw.root_cause: confidence} if raw.root_cause else {},
            details={"propagation_paths": raw.propagation_paths},
        ),
        hypotheses=hypotheses, evidence=evidence, confidence=confidence,
        uncertainty=1.0-confidence, abstained=abstained, abstain_reason=raw.abstain_reason,
        recommended_actions=["Verify the proposed root cause against process topology and operating history."] if raw.root_cause else [],
        tool_trace=[ToolTraceStep(name, evidence_ids=["process-diagnostic-evidence"] if name == "process_diagnosis" and evidence else []) for name in raw.tool_trace],
        metadata={"pipeline_version": PIPELINE_VERSION, "artifacts": raw.artifacts},
    )
    result.validate()
    return result


def _with_task(result: DiagnosticResult, task: str | None) -> DiagnosticResult:
    if task is not None:
        result.task = task
        result.validate()
    return result


class DiagnosticPipeline:
    """Stable public dispatcher for all supported industrial domain pipelines."""

    version = PIPELINE_VERSION

    def run(self, domain: str, *, task: str | None = None, metadata: dict[str, Any] | None = None, **inputs) -> DiagnosticResult:
        domain = str(domain).strip().lower()
        if domain not in DOMAIN_PACKS:
            raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(DOMAIN_PACKS)}")
        values = dict(metadata or {})
        values.update(inputs)
        default_task = str(task or DOMAIN_PACKS[domain].tasks[0].value)
        supported_tasks = {row.value for row in DOMAIN_PACKS[domain].tasks}
        if task is not None and task not in supported_tasks:
            return _abstain(domain, default_task, f"Unsupported task {task!r}; available tasks: {sorted(supported_tasks)}")
        missing = [key for key in DOMAIN_PACKS[domain].required_metadata if values.get(key) is None]
        if missing:
            return _abstain(domain, default_task, f"Missing required metadata: {', '.join(missing)}")
        try:
            if domain == "bearing":
                signal = values.get("signal", values.get("signal_matrix"))
                if signal is None:
                    return _abstain(domain, default_task, "Missing required input: signal")
                frequencies = dict(values.get("fault_frequencies") or {})
                for key in ("BPFO", "BPFI", "BSF", "FTF"):
                    if values.get(key) is not None:
                        frequencies[key] = float(values[key])
                result = BearingDiagnosticPipeline(
                    minimum_confidence=float(values.get("minimum_confidence", 0.45)),
                    minimum_harmonics=int(values.get("minimum_harmonics", 2)),
                ).run(signal, values["sampling_rate_hz"], fault_frequencies=frequencies,
                      shaft_rate_hz=values.get("shaft_rate_hz"), channel_name=values.get("channel_name", "ch0"),
                      operating_condition=values.get("operating_condition"))
                return _with_task(result, task)
            if domain == "process":
                if values.get("signal_matrix") is None or values.get("normal_reference") is None:
                    return _abstain(domain, default_task, "Process diagnosis requires signal_matrix and normal_reference")
                raw = ProcessDiagnosticPipeline(
                    maxlag=int(values.get("maxlag", 3)), diagnosis_threshold=float(values.get("diagnosis_threshold", 0.35))
                ).run(values["signal_matrix"], values["normal_reference"], values["channel_names"],
                      process_topology=values.get("process_topology"), fault_catalog=values.get("fault_catalog"),
                      timestamps=values.get("timestamps"))
                return _with_task(_process_result(raw), task)
            if domain == "wind_scada":
                return _with_task(WindScadaDiagnosticPipeline().run(**values), task)
            if domain == "battery":
                return _with_task(BatteryDiagnosticPipeline().run(**values), task)
            if domain == "turbofan":
                return _with_task(TurbofanDiagnosticPipeline().run(**values), task)
            return _with_task(TransformerDiagnosticPipeline().run(**values), task)
        except (TypeError, ValueError) as exc:
            return _abstain(domain, default_task, f"Input validation failed: {exc}")
        except Exception as exc:
            return _abstain(domain, default_task, f"Pipeline execution failed in {domain}: {exc}")


def diagnose(domain: str, *, task: str | None = None, metadata: dict[str, Any] | None = None, **inputs) -> DiagnosticResult:
    """Run one domain pipeline and always return the cross-domain result contract."""
    return DiagnosticPipeline().run(domain, task=task, metadata=metadata, **inputs)
