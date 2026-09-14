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
    UncertaintyEstimate,
)
from ..result_contract import standardize_result
from ..tools import (
    causal_graph_filter,
    contribution_analysis,
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


def _inactive(state: Mapping[str, Any]) -> bool:
    return state.get("fault_detected") is False


def _causal_input(x: np.ndarray, stationarity: Mapping[str, Any]) -> tuple[np.ndarray, list[int]]:
    recommendations = stationarity["differencing_recommendations"]
    difference_channels = [int(row["channel"]) for row in recommendations if row.get("difference")]
    if not difference_channels:
        return x, []
    transformed = x.copy()
    for channel in difference_channels:
        transformed[1:, channel] = np.diff(x[:, channel])
        transformed[0, channel] = transformed[1, channel]
    return transformed, difference_channels


@dataclass(frozen=True)
class ProcessDecisionPolicy:
    request: DiagnosticRequest
    version: str = "process-policy-v2"

    def decide(self, execution: Mapping[str, Any], trace: ExecutionTrace) -> DiagnosticResult:
        state = dict(execution)
        detected = bool(state.get("fault_detected", False))
        alarm_fraction = float(state.get("alarm_fraction", 0.0))
        if not detected:
            confidence = 0.8
            return standardize_result(DiagnosticResult(
                domain="process",
                task="root_cause",
                decision="monitor",
                detection=DetectionResult(False, score=alarm_fraction, method="pca_process_monitoring", details={"alarm_fraction": alarm_fraction}),
                confidence=confidence,
                uncertainty=None,
                uncertainty_estimate=UncertaintyEstimate(None, "not_calibrated_process_baseline"),
                tool_trace=trace,
                metadata={
                    "workflow_version": ProcessPlugin.workflow_version,
                    "policy_version": self.version,
                    "alarm_fraction": alarm_fraction,
                    "allow_confidence_complement_uncertainty": False,
                },
            ))

        diagnosis = dict(state.get("diagnosis") or {})
        confidence = float(np.clip(diagnosis.get("confidence", state.get("root_confidence", 0.0)), 0.0, 1.0))
        root = diagnosis.get("root_cause")
        fault_label = diagnosis.get("fault_label")
        abstain_reason = diagnosis.get("abstain_reason")
        decision = "abstain" if abstain_reason else "diagnose"
        propagation_paths = list(diagnosis.get("propagation_paths") or state.get("propagation_paths") or [])
        affected = list(diagnosis.get("affected_variables") or [])

        evidence: list[Evidence] = []
        if root or state.get("root_cause_ranking"):
            evidence = [Evidence(
                source="process_root_cause",
                statement=f"Detected process deviation; root-cause candidate={root!r}.",
                score=confidence,
                details={
                    "root_cause_ranking": state.get("root_cause_ranking", []),
                    "affected_variables": affected,
                    "propagation_paths": propagation_paths,
                    "alarm_fraction": alarm_fraction,
                },
                evidence_id="process-diagnostic-evidence",
                kind="causal",
            )]
            target_step = "knowledge_guided_root_cause_decision" if state.get("knowledge_guided_used") else "process_diagnosis"
            trace.require(target_step).evidence_ids.append("process-diagnostic-evidence")

        hypotheses: list[DiagnosticHypothesis] = []
        if fault_label or root:
            hypotheses.append(DiagnosticHypothesis(
                label=str(fault_label or root),
                score=confidence,
                rationale="PCA deviation, contribution, temporal, causal, onset, and topology evidence were combined.",
                evidence_ids=["process-diagnostic-evidence"] if evidence else [],
                details={"root_cause": root},
            ))

        return standardize_result(DiagnosticResult(
            domain="process",
            task="root_cause",
            decision=decision,
            detection=DetectionResult(True, score=float(np.clip(max(alarm_fraction, confidence), 0.0, 1.0)), method="pca_process_monitoring", details={"alarm_fraction": alarm_fraction}),
            localization=LocalizationResult(
                components=[str(root)] if root else [],
                channels=affected,
                scores={str(root): confidence} if root else {},
                details={"propagation_paths": propagation_paths},
            ),
            hypotheses=hypotheses,
            evidence=evidence,
            confidence=confidence,
            uncertainty=None,
            uncertainty_estimate=UncertaintyEstimate(None, "not_calibrated_process_baseline"),
            abstained=bool(abstain_reason),
            abstain_reason=abstain_reason,
            recommended_actions=["Verify the proposed root cause against process topology and operating history."] if root else [],
            tool_trace=trace,
            metadata={
                "workflow_version": ProcessPlugin.workflow_version,
                "policy_version": self.version,
                "alarm_fraction": alarm_fraction,
                "differenced_channels": state.get("differenced_channels", []),
                "fault_type_scores": state.get("fault_type_scores", {}),
                "allow_confidence_complement_uncertainty": False,
            },
        ))


class ProcessPlugin:
    name = "process"
    workflow_version = "2.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("signal_matrix", "normal_reference", "channel_names")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"process requires {', '.join(missing)}")
        x = np.asarray(values["signal_matrix"], dtype=float)
        ref = np.asarray(values["normal_reference"], dtype=float)
        if x.ndim == 1: x = x[:, None]
        if ref.ndim == 1: ref = ref[:, None]
        if x.ndim != 2 or ref.ndim != 2 or x.shape[1] != ref.shape[1]:
            raise ValueError("current and normal-reference matrices must be 2-D with matching channels")
        channel_names = [str(row) for row in values["channel_names"]]
        if len(channel_names) != x.shape[1]:
            raise ValueError("channel_names length mismatch")
        values["signal_matrix"] = x
        values["normal_reference"] = ref
        values["channel_names"] = channel_names
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        cfg = dict(request.inputs)

        def standardize(state):
            return standardize_against_normal(state["signal_matrix"], state["normal_reference"])

        def pca(state):
            out = pca_monitoring(
                state["standardized_signal"], state["standardized_reference"],
                variance_target=float(cfg.get("variance_target", 0.95)),
                alpha=float(cfg.get("control_alpha", 0.99)),
            )
            alarm_fraction = float(np.mean(np.asarray(out["alarm_mask"], dtype=bool)))
            return {**out, "alarm_fraction": alarm_fraction, "fault_detected": alarm_fraction >= float(cfg.get("minimum_alarm_fraction", 0.05))}

        def contributions(state):
            if _inactive(state): return {}
            return contribution_analysis(state["standardized_signal"], state["pca_state"], state["alarm_mask"])

        def shift(state):
            if _inactive(state): return {}
            return pre_post_shift_evidence(state["standardized_signal"], state["channel_names"], alarm_mask=state["alarm_mask"])

        def type_evidence(state):
            if _inactive(state): return {}
            return temporal_fault_type_evidence(state["standardized_signal"], state["channel_names"], alarm_mask=state["alarm_mask"])

        def stationarity(state):
            if _inactive(state): return {}
            out = stationarity_analysis(state["standardized_signal"])
            causal, differenced = _causal_input(state["standardized_signal"], out)
            return {**out, "causal_input": causal, "differenced_channels": differenced}

        def granger(state):
            if _inactive(state): return {}
            return granger_causality(
                state["causal_input"], state["channel_names"],
                maxlag=int(cfg.get("maxlag", 3)), alpha=float(cfg.get("granger_alpha", 0.05)),
            )

        def causal_filter(state):
            if _inactive(state): return {}
            return causal_graph_filter(
                state["directed_edges"],
                process_topology=state.get("process_topology"),
                p_value_threshold=float(cfg.get("granger_alpha", 0.05)),
            )

        def onset(state):
            if _inactive(state): return {}
            return fault_onset_timing(
                state["standardized_signal"], state["alarm_mask"], state["channel_names"],
                timestamps=state.get("timestamps"),
                z_threshold=float(cfg.get("onset_z_threshold", 3.5)),
                persistence=int(cfg.get("onset_persistence", 3)),
            )

        def rank(state):
            if _inactive(state): return {}
            return root_cause_rank_enhanced(
                state["suspect_variables"], state["filtered_causal_graph"], state["onset_order"],
                variable_contributions=state["variable_contributions"],
                shift_scores=state["shift_scores"], channel_names=state["channel_names"],
            )

        def knowledge(state):
            if _inactive(state): return {"knowledge_guided_used": False}
            catalog = state.get("fault_catalog")
            if not (bool(cfg.get("use_knowledge_catalog", True)) and isinstance(catalog, dict) and "faults" in catalog):
                return {"knowledge_guided_used": False}
            scores = {state["channel_names"][i]: float(value) for i, value in enumerate(state["variable_contributions"])}
            out = knowledge_guided_root_cause_decision(
                state["root_cause_ranking"], catalog,
                shift_scores=state["shift_scores"], contribution_scores=scores,
                onset_order=state["onset_order"], fault_type_scores=state["fault_type_scores"],
            )
            accepted = out.get("root_cause") is not None and float(out["confidence"]) >= float(cfg.get("diagnosis_threshold", 0.35)) and out.get("abstain_reason") is None
            diagnosis = {
                "fault_label": out["fault_label"] if accepted else None,
                "root_cause": out["root_cause"] if accepted else None,
                "confidence": float(out["confidence"]),
                "abstain_reason": None if accepted else (out.get("abstain_reason") or "knowledge_guided_evidence_below_threshold"),
                "affected_variables": sorted({v for path in state["propagation_paths"] for v in path if v != out.get("root_cause")}),
                "propagation_paths": state["propagation_paths"],
            }
            return {"knowledge_guided_used": True, "knowledge_guided_result": out, "diagnosis": diagnosis}

        def diagnose_step(state):
            if _inactive(state): return {"diagnosis": {}}
            if state.get("knowledge_guided_used"):
                return {}
            out = process_diagnosis(
                state["root_cause_ranking"], state["propagation_paths"],
                fault_catalog=state.get("fault_catalog"),
                confidence_threshold=float(cfg.get("diagnosis_threshold", 0.35)),
            )
            return {"diagnosis": out}

        steps = (
            Step("standardize_against_normal", standardize, version="baseline-v1"),
            Step("pca_monitoring", pca, depends_on=("standardize_against_normal",), version="baseline-v1"),
            Step("contribution_analysis", contributions, depends_on=("pca_monitoring",), version="baseline-v1"),
            Step("pre_post_shift_evidence", shift, depends_on=("contribution_analysis",), version="baseline-v1"),
            Step("temporal_fault_type_evidence", type_evidence, depends_on=("pre_post_shift_evidence",), version="baseline-v1"),
            Step("stationarity_analysis", stationarity, depends_on=("temporal_fault_type_evidence",), version="baseline-v1"),
            Step("granger_causality", granger, depends_on=("stationarity_analysis",), version="baseline-v1"),
            Step("causal_graph_filter", causal_filter, depends_on=("granger_causality",), version="baseline-v1"),
            Step("fault_onset_timing", onset, depends_on=("causal_graph_filter",), version="baseline-v1"),
            Step("root_cause_rank_enhanced", rank, depends_on=("fault_onset_timing",), version="baseline-v1"),
            Step("knowledge_guided_root_cause_decision", knowledge, depends_on=("root_cause_rank_enhanced",), version="baseline-v1"),
            Step("process_diagnosis", diagnose_step, depends_on=("knowledge_guided_root_cause_decision",), version="baseline-v1"),
        )
        return Workflow(steps, version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> ProcessDecisionPolicy:
        return ProcessDecisionPolicy(request)
