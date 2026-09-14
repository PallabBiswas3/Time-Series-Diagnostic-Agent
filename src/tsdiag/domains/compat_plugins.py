from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..contracts import DiagnosticRequest
from ..execution import ExecutionTrace, Step, Workflow
from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    ToolTraceStep,
)
from ..result_contract import standardize_result
from .bearing_runner import BearingDiagnosticPipeline
from .process_runner import ProcessDiagnosticPipeline, ProcessDiagnosticResult
from .runners import BatteryDiagnosticPipeline, TransformerDiagnosticPipeline, TurbofanDiagnosticPipeline


def _model_version(request: DiagnosticRequest) -> str | None:
    values = tuple(str(value) for value in request.model_refs.values() if value)
    return values[0] if len(values) == 1 else (";".join(values) if values else None)


def _workflow_trace_metadata(trace: ExecutionTrace) -> list[dict[str, Any]]:
    return [
        {
            "step": step.tool,
            "status": step.status,
            "duration_seconds": float(step.duration_seconds),
        }
        for step in trace.steps
    ]


def _stamp_result(
    result: DiagnosticResult,
    request: DiagnosticRequest,
    trace: ExecutionTrace,
    *,
    workflow_version: str,
    policy_version: str,
) -> DiagnosticResult:
    """Add new-architecture provenance without changing domain science.

    The wrapped runner's detailed chronological tool trace remains the public
    ``result.tool_trace`` during migration. The outer workflow trace records the
    compatibility boundary separately so no existing audit detail is lost.
    """
    result.metadata["workflow_version"] = workflow_version
    result.metadata["policy_version"] = policy_version
    result.metadata["model_version"] = _model_version(request)
    result.metadata["model_refs"] = dict(request.model_refs)
    result.metadata["compatibility_adapter"] = True
    result.metadata["outer_workflow_trace"] = _workflow_trace_metadata(trace)
    return standardize_result(result)


@dataclass(frozen=True)
class PassthroughDecisionPolicy:
    result_key: str
    workflow_version: str
    version: str = "compat-1.0"

    def decide(self, execution: Mapping[str, Any], trace: ExecutionTrace, request: DiagnosticRequest) -> DiagnosticResult:
        result = execution[self.result_key]
        if not isinstance(result, DiagnosticResult):
            raise TypeError(f"{self.result_key!r} did not produce DiagnosticResult")
        return _stamp_result(
            result,
            request,
            trace,
            workflow_version=self.workflow_version,
            policy_version=self.version,
        )


@dataclass(frozen=True)
class ProcessDecisionPolicy:
    workflow_version: str
    version: str = "compat-1.0"

    def decide(self, execution: Mapping[str, Any], trace: ExecutionTrace, request: DiagnosticRequest) -> DiagnosticResult:
        raw = execution["process_analysis"]
        if not isinstance(raw, ProcessDiagnosticResult):
            raise TypeError("process_analysis did not produce ProcessDiagnosticResult")

        evidence: list[Evidence] = []
        hypotheses: list[DiagnosticHypothesis] = []
        if raw.fault_detected:
            ev = Evidence(
                source="process_root_cause",
                statement=f"Detected process deviation; root-cause candidate={raw.root_cause!r}.",
                score=float(np.clip(raw.confidence, 0.0, 1.0)),
                details={
                    "affected_variables": raw.affected_variables,
                    "propagation_paths": raw.propagation_paths,
                },
                evidence_id="process-diagnostic-evidence",
                kind="causal",
            )
            evidence.append(ev)
            if raw.fault_label or raw.root_cause:
                hypotheses.append(
                    DiagnosticHypothesis(
                        label=str(raw.fault_label or raw.root_cause),
                        score=float(np.clip(raw.confidence, 0.0, 1.0)),
                        rationale="Process monitoring, contribution, temporal and causal evidence were combined.",
                        evidence_ids=[ev.evidence_id],
                    )
                )

        abstained = bool(raw.fault_detected and raw.abstain_reason)
        decision = "abstain" if abstained else ("diagnose" if raw.fault_detected else "monitor")
        confidence = float(np.clip(raw.confidence, 0.0, 1.0))
        inner_trace = ExecutionTrace(
            ToolTraceStep(
                name,
                duration_seconds=(raw.timings or {}).get(name, 0.0),
                evidence_ids=["process-diagnostic-evidence"]
                if name == "process_diagnosis" and evidence
                else [],
            )
            for name in raw.tool_trace
        )
        result = DiagnosticResult(
            domain="process",
            task="root_cause",
            decision=decision,
            detection=DetectionResult(
                abnormal=raw.fault_detected,
                score=confidence,
                method="pca_process_pipeline",
            ),
            localization=LocalizationResult(
                components=[raw.root_cause] if raw.root_cause else [],
                channels=list(raw.affected_variables),
                scores={raw.root_cause: confidence} if raw.root_cause else {},
                details={"propagation_paths": raw.propagation_paths},
            ),
            hypotheses=hypotheses,
            evidence=evidence,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            abstained=abstained,
            abstain_reason=raw.abstain_reason,
            recommended_actions=[
                "Verify the proposed root cause against process topology and operating history."
            ]
            if raw.root_cause
            else [],
            tool_trace=inner_trace,
            metadata={"artifacts": raw.artifacts},
        )
        return _stamp_result(
            result,
            request,
            trace,
            workflow_version=self.workflow_version,
            policy_version=self.version,
        )


class BearingPlugin:
    name = "bearing"
    workflow_version = "compat-1.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        if values.get("signal") is None:
            values["signal"] = values.get("signal_matrix")
        if values.get("signal") is None or values.get("sampling_rate_hz") is None:
            raise ValueError("bearing requires signal and sampling_rate_hz")
        frequencies = dict(values.get("fault_frequencies") or {})
        for key in ("BPFO", "BPFI", "BSF", "FTF"):
            if values.get(key) is not None:
                frequencies[key] = float(values[key])
        values["fault_frequencies"] = frequencies
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        def analysis(state: dict[str, Any]) -> dict[str, Any]:
            result = BearingDiagnosticPipeline(
                minimum_confidence=float(state.get("minimum_confidence", 0.45)),
                minimum_harmonics=int(state.get("minimum_harmonics", 2)),
            ).run(
                state["signal"],
                state["sampling_rate_hz"],
                fault_frequencies=state.get("fault_frequencies"),
                shaft_rate_hz=state.get("shaft_rate_hz"),
                channel_name=state.get("channel_name", "ch0"),
                operating_condition=state.get("operating_condition"),
            )
            return {"bearing_analysis": result}

        return Workflow((Step("bearing_analysis", analysis, version="legacy-runner-v1"),), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> PassthroughDecisionPolicy:
        return PassthroughDecisionPolicy("bearing_analysis", self.workflow_version)


class ProcessPlugin:
    name = "process"
    workflow_version = "compat-1.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("signal_matrix", "normal_reference", "channel_names")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"process requires {', '.join(missing)}")
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        def analysis(state: dict[str, Any]) -> dict[str, Any]:
            result = ProcessDiagnosticPipeline(
                variance_target=float(state.get("variance_target", 0.95)),
                control_alpha=float(state.get("control_alpha", 0.99)),
                maxlag=int(state.get("maxlag", 3)),
                granger_alpha=float(state.get("granger_alpha", 0.05)),
                onset_z_threshold=float(state.get("onset_z_threshold", 3.5)),
                onset_persistence=int(state.get("onset_persistence", 3)),
                diagnosis_threshold=float(state.get("diagnosis_threshold", 0.35)),
                minimum_alarm_fraction=float(state.get("minimum_alarm_fraction", 0.05)),
                use_knowledge_catalog=bool(state.get("use_knowledge_catalog", True)),
            ).run(
                state["signal_matrix"],
                state["normal_reference"],
                state["channel_names"],
                process_topology=state.get("process_topology"),
                fault_catalog=state.get("fault_catalog"),
                timestamps=state.get("timestamps"),
            )
            return {"process_analysis": result}

        return Workflow((Step("process_analysis", analysis, version="legacy-runner-v1"),), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> ProcessDecisionPolicy:
        return ProcessDecisionPolicy(self.workflow_version)


class BatteryPlugin:
    name = "battery"
    workflow_version = "compat-1.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("cell_voltage", "cell_temperature", "cell_ids", "timestamps")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"battery requires {', '.join(missing)}")
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        def analysis(state: dict[str, Any]) -> dict[str, Any]:
            required = {key: state[key] for key in ("cell_voltage", "cell_temperature", "cell_ids", "timestamps")}
            context = {key: value for key, value in state.items() if key not in required}
            return {"battery_analysis": BatteryDiagnosticPipeline().run(**required, **context)}

        return Workflow((Step("battery_analysis", analysis, version="legacy-runner-v1"),), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> PassthroughDecisionPolicy:
        return PassthroughDecisionPolicy("battery_analysis", self.workflow_version)


class TurbofanPlugin:
    name = "turbofan"
    workflow_version = "compat-1.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("signal_matrix", "channel_names", "cycle_index")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"turbofan requires {', '.join(missing)}")
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        def analysis(state: dict[str, Any]) -> dict[str, Any]:
            required = {key: state[key] for key in ("signal_matrix", "channel_names", "cycle_index")}
            context = {key: value for key, value in state.items() if key not in required}
            return {"turbofan_analysis": TurbofanDiagnosticPipeline().run(**required, **context)}

        return Workflow((Step("turbofan_analysis", analysis, version="legacy-runner-v1"),), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> PassthroughDecisionPolicy:
        return PassthroughDecisionPolicy("turbofan_analysis", self.workflow_version)


class TransformerPlugin:
    name = "transformer"
    workflow_version = "compat-1.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("signal_matrix", "sampling_rate_hz", "sensor_positions")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"transformer requires {', '.join(missing)}")
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        def analysis(state: dict[str, Any]) -> dict[str, Any]:
            required = {key: state[key] for key in ("signal_matrix", "sampling_rate_hz", "sensor_positions")}
            context = {key: value for key, value in state.items() if key not in required}
            return {"transformer_analysis": TransformerDiagnosticPipeline().run(**required, **context)}

        return Workflow((Step("transformer_analysis", analysis, version="legacy-runner-v1"),), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> PassthroughDecisionPolicy:
        return PassthroughDecisionPolicy("transformer_analysis", self.workflow_version)
