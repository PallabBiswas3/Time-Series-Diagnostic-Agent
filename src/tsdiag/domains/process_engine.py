from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import ToolRegistry
from ..core import ExecutionEngine, ExecutionPlan, PlanStep, RunContext, RunRequest, ToolOutcome
from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    ToolTraceStep,
)
from ..tools import (
    causal_graph_filter,
    contribution_analysis,
    data_quality_check,
    fault_onset_timing,
    granger_causality,
    knowledge_guided_root_cause_decision,
    pca_monitoring,
    pre_post_shift_evidence,
    process_diagnosis,
    root_cause_rank_enhanced,
    standardize_against_normal,
    stationarity_analysis,
    temporal_fault_type_evidence,
)
from .process import PROCESS_PACK


@dataclass(frozen=True)
class ProcessEngineConfig:
    variance_target: float = 0.95
    control_alpha: float = 0.99
    maxlag: int = 3
    granger_alpha: float = 0.05
    onset_z_threshold: float = 3.5
    onset_persistence: int = 3
    diagnosis_threshold: float = 0.35
    use_knowledge_catalog: bool = True


class ProcessExecutionPipeline:
    """Architecture-native process pipeline executed through ExecutionEngine.

    Numerical tools and thresholds intentionally mirror ProcessDiagnosticPipeline.
    The migration changes orchestration only; it does not alter the benchmarked
    process methodology.
    """

    def __init__(self, config: ProcessEngineConfig | None = None):
        self.config = config or ProcessEngineConfig()
        self.registry = self._build_registry()
        self.engine = ExecutionEngine(self.registry)

    @staticmethod
    def _causal_signal(standardized_signal, stationarity: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
        x = np.asarray(standardized_signal, dtype=float)
        recommendations = stationarity["differencing_recommendations"]
        differenced = [int(row["channel"]) for row in recommendations if row.get("difference")]
        if not differenced:
            return x, []
        transformed = x.copy()
        for channel in differenced:
            transformed[1:, channel] = np.diff(x[:, channel])
            transformed[0, channel] = transformed[1, channel]
        return transformed, differenced

    def _build_registry(self) -> ToolRegistry:
        cfg = self.config
        registry = ToolRegistry()

        registry.register("process_data_quality", data_quality_check)
        registry.register("standardize_against_normal", standardize_against_normal)

        def pca_step(standardized_signal, standardized_reference):
            return pca_monitoring(
                standardized_signal,
                standardized_reference,
                variance_target=cfg.variance_target,
                alpha=cfg.control_alpha,
            )

        def contribution_step(standardized_signal, pca_state, alarm_mask):
            return contribution_analysis(standardized_signal, pca_state, alarm_mask)

        def shift_step(standardized_signal, channel_names, alarm_mask):
            return pre_post_shift_evidence(
                standardized_signal,
                channel_names,
                alarm_mask=alarm_mask,
            )

        def type_step(standardized_signal, channel_names, alarm_mask):
            return temporal_fault_type_evidence(
                standardized_signal,
                channel_names,
                alarm_mask=alarm_mask,
            )

        def stationarity_step(standardized_signal):
            result = stationarity_analysis(standardized_signal)
            causal_signal, differenced = self._causal_signal(standardized_signal, result)
            return {
                **result,
                "causal_signal": causal_signal,
                "differenced_channels": differenced,
            }

        def granger_step(causal_signal, channel_names):
            return granger_causality(
                causal_signal,
                channel_names,
                maxlag=cfg.maxlag,
                alpha=cfg.granger_alpha,
            )

        def filter_step(directed_edges, process_topology=None):
            return causal_graph_filter(
                directed_edges,
                process_topology=process_topology,
                p_value_threshold=cfg.granger_alpha,
            )

        def onset_step(standardized_signal, alarm_mask, channel_names, timestamps=None):
            return fault_onset_timing(
                standardized_signal,
                alarm_mask,
                channel_names,
                timestamps=timestamps,
                z_threshold=cfg.onset_z_threshold,
                persistence=cfg.onset_persistence,
            )

        def rank_step(
            suspect_variables,
            filtered_causal_graph,
            onset_order,
            variable_contributions,
            shift_scores,
            channel_names,
        ):
            return root_cause_rank_enhanced(
                suspect_variables,
                filtered_causal_graph,
                onset_order,
                variable_contributions=variable_contributions,
                shift_scores=shift_scores,
                channel_names=channel_names,
            )

        def diagnosis_step(
            root_cause_ranking,
            propagation_paths,
            fault_catalog=None,
            shift_scores=None,
            variable_contributions=None,
            onset_order=None,
            fault_type_scores=None,
            channel_names=None,
        ):
            names = list(channel_names or [])
            contributions = np.asarray(variable_contributions if variable_contributions is not None else [], dtype=float)
            contribution_scores = {
                names[i]: float(value)
                for i, value in enumerate(contributions)
                if i < len(names)
            }

            if (
                cfg.use_knowledge_catalog
                and isinstance(fault_catalog, dict)
                and "faults" in fault_catalog
            ):
                kg = knowledge_guided_root_cause_decision(
                    root_cause_ranking,
                    fault_catalog,
                    shift_scores=shift_scores,
                    contribution_scores=contribution_scores,
                    onset_order=onset_order,
                    fault_type_scores=fault_type_scores,
                )
                abstain_reason = kg.get("abstain_reason")
                accepted = (
                    kg.get("root_cause") is not None
                    and float(kg.get("confidence", 0.0)) >= cfg.diagnosis_threshold
                    and abstain_reason is None
                )
                root = kg.get("root_cause")
                affected = sorted(
                    {
                        node
                        for path in propagation_paths
                        for node in path
                        if node != root
                    }
                )
                return {
                    "fault_label": kg.get("fault_label") if accepted else None,
                    "root_cause": root if accepted else None,
                    "confidence": float(kg.get("confidence", 0.0)),
                    "abstain_reason": None if accepted else (
                        abstain_reason or "knowledge_guided_evidence_below_threshold"
                    ),
                    "affected_variables": affected,
                    "propagation_paths": list(propagation_paths),
                    "knowledge_decision": kg,
                }

            return process_diagnosis(
                root_cause_ranking,
                propagation_paths,
                fault_catalog=fault_catalog,
                confidence_threshold=cfg.diagnosis_threshold,
            )

        for name, fn in {
            "pca_monitoring": pca_step,
            "contribution_analysis": contribution_step,
            "pre_post_shift_evidence": shift_step,
            "temporal_fault_type_evidence": type_step,
            "stationarity_analysis": stationarity_step,
            "granger_causality": granger_step,
            "causal_graph_filter": filter_step,
            "fault_onset_timing": onset_step,
            "root_cause_rank": rank_step,
            "process_diagnosis": diagnosis_step,
        }.items():
            registry.register(name, fn)
        return registry

    @staticmethod
    def _plan(task: str, tools: tuple[str, ...]) -> ExecutionPlan:
        return ExecutionPlan(
            domain="process",
            task=task,
            steps=tuple(PlanStep(tool=name) for name in tools),
            policy="deterministic-process-engine-v1",
        )

    @staticmethod
    def _combine_trace(*parts: list[ToolTraceStep]) -> list[ToolTraceStep]:
        return [step for part in parts for step in part]

    def run_request(self, request: RunRequest) -> DiagnosticResult:
        request.validate()
        if request.domain != "process":
            raise ValueError("ProcessExecutionPipeline requires domain='process'")

        missing = [
            name
            for name in PROCESS_PACK.required_metadata
            if request.metadata.get(name) is None
        ]
        if missing:
            return self._abstain_missing(request, missing)

        context = RunContext.from_request(request)
        context.publish({"signal_matrix": request.observation})

        phase1 = self.engine.run(
            PROCESS_PACK,
            self._plan(
                request.task,
                (
                    "process_data_quality",
                    "standardize_against_normal",
                    "pca_monitoring",
                ),
            ),
            context,
        )
        if phase1.abstained:
            return self._abstain_execution(request, phase1)

        alarm_mask = np.asarray(context.get("alarm_mask"), dtype=bool)
        fault_detected = bool(np.any(alarm_mask))
        if not fault_detected:
            result = DiagnosticResult(
                domain="process",
                task=request.task,
                decision="monitor",
                detection=DetectionResult(
                    abnormal=False,
                    score=0.0,
                    method="pca_monitoring",
                    details={"alarm_count": 0},
                ),
                evidence=phase1.evidence,
                confidence=0.8,
                uncertainty=0.2,
                tool_trace=phase1.trace,
                metadata={
                    "runtime": "execution_engine",
                    "policy": "deterministic-process-engine-v1",
                    "quality_flags": context.get("quality_flags", []),
                    "control_limits": context.get("control_limits"),
                },
            )
            result.validate()
            return result

        phase2 = self.engine.run(
            PROCESS_PACK,
            self._plan(
                request.task,
                (
                    "contribution_analysis",
                    "pre_post_shift_evidence",
                    "temporal_fault_type_evidence",
                    "stationarity_analysis",
                    "granger_causality",
                    "causal_graph_filter",
                    "fault_onset_timing",
                    "root_cause_rank",
                    "process_diagnosis",
                ),
            ),
            context,
        )
        if phase2.abstained:
            return self._abstain_execution(
                request,
                phase2,
                prior_trace=phase1.trace,
                prior_evidence=phase1.evidence,
            )

        diagnosis_confidence = float(np.clip(context.get("confidence", 0.0), 0.0, 1.0))
        root = context.get("root_cause")
        label = context.get("fault_label")
        abstain_reason = context.get("abstain_reason")
        propagation_paths = list(context.get("propagation_paths", []))
        affected_variables = list(context.get("affected_variables", []))

        evidence = list(phase1.evidence) + list(phase2.evidence)
        if root is not None:
            root_evidence = Evidence(
                source="process_diagnosis",
                statement=f"Root-cause candidate: {root}",
                score=diagnosis_confidence,
                evidence_id="process-root-cause",
                kind="causal",
                provenance={
                    "runtime": "execution_engine",
                    "policy": "deterministic-process-engine-v1",
                },
                details={
                    "affected_variables": affected_variables,
                    "propagation_paths": propagation_paths,
                },
            )
            evidence.append(root_evidence)

        hypotheses: list[DiagnosticHypothesis] = []
        if label is not None:
            hypotheses.append(
                DiagnosticHypothesis(
                    label=str(label),
                    score=diagnosis_confidence,
                    rationale="Produced by the deterministic process tool chain executed through ExecutionEngine.",
                    evidence_ids=["process-root-cause"] if root is not None else [],
                    details={"root_cause": root},
                )
            )

        abstained = abstain_reason is not None
        decision = "abstain" if abstained else "diagnose"
        result = DiagnosticResult(
            domain="process",
            task=request.task,
            decision=decision,
            detection=DetectionResult(
                abnormal=True,
                score=diagnosis_confidence,
                onset=(context.get("onset_order") or [None])[0],
                method="pca_monitoring",
                details={
                    "alarm_count": int(np.sum(alarm_mask)),
                    "control_limits": context.get("control_limits"),
                },
            ),
            localization=LocalizationResult(
                components=[str(root)] if root is not None else [],
                channels=affected_variables,
                details={"propagation_paths": propagation_paths},
            ),
            hypotheses=hypotheses,
            evidence=evidence,
            confidence=diagnosis_confidence,
            uncertainty=float(np.clip(1.0 - diagnosis_confidence, 0.0, 1.0)),
            abstained=abstained,
            abstain_reason=abstain_reason,
            tool_trace=self._combine_trace(phase1.trace, phase2.trace),
            metadata={
                "runtime": "execution_engine",
                "policy": "deterministic-process-engine-v1",
                "quality_flags": context.get("quality_flags", []),
                "differenced_channels": context.get("differenced_channels", []),
                "root_cause_ranking": context.get("root_cause_ranking", []),
                "fault_type_scores": context.get("fault_type_scores", {}),
            },
        )
        result.validate()
        return result

    @staticmethod
    def _abstain_missing(request: RunRequest, missing: list[str]) -> DiagnosticResult:
        result = DiagnosticResult(
            domain="process",
            task=request.task,
            decision="abstain",
            detection=DetectionResult(abnormal=None, method="runtime_metadata_validation"),
            confidence=0.0,
            uncertainty=1.0,
            abstained=True,
            abstain_reason="Missing required metadata: " + ", ".join(missing),
            metadata={"missing_required_metadata": missing},
        )
        result.validate()
        return result

    @staticmethod
    def _abstain_execution(
        request: RunRequest,
        execution,
        *,
        prior_trace=None,
        prior_evidence=None,
    ) -> DiagnosticResult:
        result = DiagnosticResult(
            domain="process",
            task=request.task,
            decision="abstain",
            detection=DetectionResult(abnormal=None, method="execution_engine"),
            evidence=list(prior_evidence or []) + list(execution.evidence),
            confidence=0.0,
            uncertainty=1.0,
            abstained=True,
            abstain_reason=execution.abstain_reason or "Process execution aborted.",
            tool_trace=list(prior_trace or []) + list(execution.trace),
            metadata={"runtime": "execution_engine"},
        )
        result.validate()
        return result
