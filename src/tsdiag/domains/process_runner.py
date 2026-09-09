from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..tools import (
    causal_graph_filter,
    contribution_analysis,
    fault_onset_timing,
    granger_causality,
    pca_monitoring,
    pre_post_shift_evidence,
    process_diagnosis,
    root_cause_rank_enhanced,
    standardize_against_normal,
    stationarity_analysis,
)


@dataclass
class ProcessDiagnosticResult:
    fault_detected: bool
    fault_label: str | None
    root_cause: str | None
    affected_variables: list[str]
    propagation_paths: list[list[str]]
    confidence: float
    abstain_reason: str | None
    tool_trace: list[str]
    artifacts: dict[str, Any]


class ProcessDiagnosticPipeline:
    """Deterministic end-to-end baseline for multivariate process diagnosis.

    This runner wires the process-domain tool contracts together. It deliberately
    keeps every intermediate artifact so an adaptive/LLM policy can later be
    compared with the same numerical tools and evidence.
    """

    def __init__(
        self,
        *,
        variance_target: float = 0.95,
        control_alpha: float = 0.99,
        maxlag: int = 3,
        granger_alpha: float = 0.05,
        onset_z_threshold: float = 3.5,
        onset_persistence: int = 3,
        diagnosis_threshold: float = 0.35,
    ):
        self.variance_target = variance_target
        self.control_alpha = control_alpha
        self.maxlag = maxlag
        self.granger_alpha = granger_alpha
        self.onset_z_threshold = onset_z_threshold
        self.onset_persistence = onset_persistence
        self.diagnosis_threshold = diagnosis_threshold

    @staticmethod
    def _causal_input(x: np.ndarray, stationarity: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
        recommendations = stationarity["differencing_recommendations"]
        difference_channels = [int(r["channel"]) for r in recommendations if r.get("difference")]
        if not difference_channels:
            return x, []

        transformed = x.copy()
        for j in difference_channels:
            transformed[1:, j] = np.diff(x[:, j])
            transformed[0, j] = transformed[1, j]
        return transformed, difference_channels

    def run(
        self,
        signal_matrix,
        normal_reference,
        channel_names,
        *,
        process_topology=None,
        fault_catalog=None,
        timestamps=None,
    ) -> ProcessDiagnosticResult:
        x = np.asarray(signal_matrix, dtype=float)
        ref = np.asarray(normal_reference, dtype=float)
        names = list(channel_names)
        trace: list[str] = []
        artifacts: dict[str, Any] = {}

        if x.ndim == 1:
            x = x[:, None]
        if ref.ndim == 1:
            ref = ref[:, None]
        if x.ndim != 2 or ref.ndim != 2 or x.shape[1] != ref.shape[1]:
            raise ValueError("current and normal-reference matrices must be 2-D with matching channels")
        if len(names) != x.shape[1]:
            raise ValueError("channel_names length mismatch")

        standardized = standardize_against_normal(x, ref)
        artifacts["standardization"] = standardized
        trace.append("standardize_against_normal")

        pca = pca_monitoring(
            standardized["standardized_signal"],
            standardized["standardized_reference"],
            variance_target=self.variance_target,
            alpha=self.control_alpha,
        )
        artifacts["pca_monitoring"] = pca
        trace.append("pca_monitoring")

        alarm_mask = np.asarray(pca["alarm_mask"], dtype=bool)
        fault_detected = bool(np.any(alarm_mask))
        if not fault_detected:
            return ProcessDiagnosticResult(
                fault_detected=False,
                fault_label=None,
                root_cause=None,
                affected_variables=[],
                propagation_paths=[],
                confidence=0.8,
                abstain_reason=None,
                tool_trace=trace,
                artifacts=artifacts,
            )

        contributions = contribution_analysis(
            standardized["standardized_signal"],
            pca["pca_state"],
            alarm_mask,
        )
        artifacts["contribution_analysis"] = contributions
        trace.append("contribution_analysis")

        shift = pre_post_shift_evidence(
            standardized["standardized_signal"],
            names,
            alarm_mask=alarm_mask,
        )
        artifacts["pre_post_shift_evidence"] = shift
        trace.append("pre_post_shift_evidence")

        stationarity = stationarity_analysis(standardized["standardized_signal"])
        artifacts["stationarity_analysis"] = stationarity
        trace.append("stationarity_analysis")

        causal_input, differenced_channels = self._causal_input(
            standardized["standardized_signal"], stationarity
        )
        artifacts["causal_preprocessing"] = {"differenced_channels": differenced_channels}

        granger = granger_causality(
            causal_input,
            names,
            maxlag=self.maxlag,
            alpha=self.granger_alpha,
        )
        artifacts["granger_causality"] = granger
        trace.append("granger_causality")

        filtered = causal_graph_filter(
            granger["directed_edges"],
            process_topology=process_topology,
            p_value_threshold=self.granger_alpha,
        )
        artifacts["causal_graph_filter"] = filtered
        trace.append("causal_graph_filter")

        onset = fault_onset_timing(
            standardized["standardized_signal"],
            alarm_mask,
            names,
            timestamps=timestamps,
            z_threshold=self.onset_z_threshold,
            persistence=self.onset_persistence,
        )
        artifacts["fault_onset_timing"] = onset
        trace.append("fault_onset_timing")

        ranking = root_cause_rank_enhanced(
            contributions["suspect_variables"],
            filtered["filtered_causal_graph"],
            onset["onset_order"],
            variable_contributions=contributions["variable_contributions"],
            shift_scores=shift["shift_scores"],
            channel_names=names,
        )
        artifacts["root_cause_rank"] = ranking
        trace.append("root_cause_rank_enhanced")

        diagnosis = process_diagnosis(
            ranking["root_cause_ranking"],
            ranking["propagation_paths"],
            fault_catalog=fault_catalog,
            confidence_threshold=self.diagnosis_threshold,
        )
        artifacts["process_diagnosis"] = diagnosis
        trace.append("process_diagnosis")

        return ProcessDiagnosticResult(
            fault_detected=True,
            fault_label=diagnosis["fault_label"],
            root_cause=diagnosis["root_cause"],
            affected_variables=list(diagnosis["affected_variables"]),
            propagation_paths=list(diagnosis["propagation_paths"]),
            confidence=float(diagnosis["confidence"]),
            abstain_reason=diagnosis["abstain_reason"],
            tool_trace=trace,
            artifacts=artifacts,
        )
