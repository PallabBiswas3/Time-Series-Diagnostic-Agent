from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import DiagnosticRequest
from .domains import DOMAIN_PACKS
from .domains.bearing_runner import BearingDiagnosticPipeline
from .domains.process_runner import ProcessDiagnosticPipeline, ProcessDiagnosticResult
from .domains.runners import (
    BatteryDiagnosticPipeline,
    TransformerDiagnosticPipeline,
    TurbofanDiagnosticPipeline,
    WindScadaDiagnosticPipeline,
)
from .execution import DomainInputSchema, InputField, StepExecutionError, WorkflowExecutor
from .models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    ToolTraceStep,
)
from .registry import domain_registry
from .result_contract import standardize_result

PIPELINE_VERSION = "1.1.0"

DOMAIN_INPUT_SCHEMAS = {
    "bearing": DomainInputSchema("bearing", (InputField("signal"), InputField("sampling_rate_hz"), InputField("fault_frequencies", False))),
    "process": DomainInputSchema("process", (InputField("signal_matrix"), InputField("normal_reference"), InputField("channel_names"), InputField("sampling_rate_hz"))),
    "wind_scada": DomainInputSchema("wind_scada", (InputField("signal_matrix"), InputField("channel_names"), InputField("timestamps"), InputField("normal_reference", False))),
    "battery": DomainInputSchema("battery", (InputField("cell_voltage"), InputField("cell_temperature"), InputField("cell_ids"), InputField("timestamps"))),
    "turbofan": DomainInputSchema("turbofan", (InputField("signal_matrix"), InputField("channel_names"), InputField("cycle_index"))),
    "transformer": DomainInputSchema("transformer", (InputField("signal_matrix"), InputField("sampling_rate_hz"), InputField("sensor_positions"), InputField("trained_image_model", False))),
}


def get_input_schema(domain: str) -> DomainInputSchema:
    try:
        return DOMAIN_INPUT_SCHEMAS[domain]
    except KeyError as exc:
        raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(DOMAIN_INPUT_SCHEMAS)}") from exc


def _abstain(domain: str, task: str, reason: str, trace: list[ToolTraceStep] | None = None) -> DiagnosticResult:
    return standardize_result(DiagnosticResult(
        domain=domain,
        task=task,
        decision="abstain",
        detection=DetectionResult(abnormal=None, method="pipeline_validation"),
        confidence=0.0,
        uncertainty=1.0,
        abstained=True,
        abstain_reason=reason,
        tool_trace=trace or [ToolTraceStep("input_validation", status="warning", details={"reason": reason})],
        metadata={"pipeline_version": PIPELINE_VERSION},
    ), pipeline_version=PIPELINE_VERSION)


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
    return standardize_result(DiagnosticResult(
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
        tool_trace=[ToolTraceStep(name, duration_seconds=(raw.timings or {}).get(name, 0.0), evidence_ids=["process-diagnostic-evidence"] if name == "process_diagnosis" and evidence else []) for name in raw.tool_trace],
        metadata={"pipeline_version": PIPELINE_VERSION, "artifacts": raw.artifacts},
    ), pipeline_version=PIPELINE_VERSION)


def _with_task(result: DiagnosticResult, task: str | None) -> DiagnosticResult:
    if task is not None:
        result.task = task
    return standardize_result(result, pipeline_version=PIPELINE_VERSION)


def _ensure_plugins() -> None:
    if domain_registry.get("wind_scada") is None:
        from .domains.wind_scada_plugin import WindScadaPlugin
        domain_registry.register(WindScadaPlugin())


def _diagnose_request(request: DiagnosticRequest) -> DiagnosticResult:
    _ensure_plugins()
    plugin = domain_registry.resolve(request.domain)
    task = str(request.task or "diagnosis")
    try:
        validated = dict(plugin.validate(request))
        execution, trace = WorkflowExecutor().run(plugin.workflow(request), validated)
        result = plugin.policy(request).decide(execution, trace)
        if request.task is not None:
            result.task = request.task
        result.metadata.setdefault("pipeline_version", PIPELINE_VERSION)
        if request.run_context is not None:
            result.metadata.setdefault("run_context", {
                "run_id": request.run_context.run_id,
                "source": request.run_context.source,
                "metadata": dict(request.run_context.metadata),
            })
        if request.model_refs:
            result.metadata.setdefault("model_refs", dict(request.model_refs))
        if request.policy_ref:
            result.metadata.setdefault("policy_ref", request.policy_ref)
        return standardize_result(result, pipeline_version=PIPELINE_VERSION)
    except StepExecutionError as exc:
        return _abstain(request.domain, task, f"Pipeline step failed: {exc}", exc.trace)
    except (TypeError, ValueError, KeyError) as exc:
        return _abstain(request.domain, task, f"Input validation failed: {exc}")
    except Exception as exc:
        return _abstain(request.domain, task, f"Pipeline execution failed in {request.domain}: {exc}")


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
            schema_values = dict(values)
            if domain == "bearing" and schema_values.get("signal") is None:
                schema_values["signal"] = schema_values.get("signal_matrix")
            get_input_schema(domain).validate(schema_values)
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
                    maxlag=int(values.get("maxlag", 3)),
                    diagnosis_threshold=float(values.get("diagnosis_threshold", 0.35)),
                    minimum_alarm_fraction=float(values.get("minimum_alarm_fraction", 0.05)),
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
        except StepExecutionError as exc:
            return _abstain(domain, default_task, f"Pipeline step failed: {exc}", exc.trace)
        except (TypeError, ValueError) as exc:
            return _abstain(domain, default_task, f"Input validation failed: {exc}")
        except Exception as exc:
            return _abstain(domain, default_task, f"Pipeline execution failed in {domain}: {exc}")


def diagnose(
    request_or_domain: DiagnosticRequest | str,
    *,
    task: str | None = None,
    metadata: dict[str, Any] | None = None,
    **inputs,
) -> DiagnosticResult:
    """Run a structured plugin request or the backwards-compatible legacy API."""
    if isinstance(request_or_domain, DiagnosticRequest):
        if task is not None or metadata is not None or inputs:
            raise TypeError("task/metadata/inputs cannot be combined with DiagnosticRequest")
        return _diagnose_request(request_or_domain)
    return DiagnosticPipeline().run(str(request_or_domain), task=task, metadata=metadata, **inputs)
