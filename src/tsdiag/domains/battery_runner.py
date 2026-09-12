from __future__ import annotations

import numpy as np

from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    PrognosisResult,
    ToolTraceStep,
    VerificationResult,
)
from ..tools.battery import capacity_health_features, estimate_capacity_rul


class BatteryPrognosticPipeline:
    """Transparent capacity-trajectory battery health and RUL baseline."""

    def __init__(
        self,
        *,
        nominal_capacity_ah: float = 2.0,
        eol_capacity_ah: float = 1.4,
        minimum_observations: int = 20,
        slope_window: int = 20,
    ):
        self.nominal_capacity_ah = float(nominal_capacity_ah)
        self.eol_capacity_ah = float(eol_capacity_ah)
        self.minimum_observations = int(minimum_observations)
        self.slope_window = int(slope_window)

    def run(self, cycle_index, capacity_ah, *, battery_id: str = "cell") -> DiagnosticResult:
        x = np.asarray(cycle_index, dtype=float).ravel()
        y = np.asarray(capacity_ah, dtype=float).ravel()
        if x.size != y.size:
            raise ValueError("cycle_index and capacity_ah must have equal length")
        if x.size == 0:
            result = DiagnosticResult(
                domain="battery",
                task="prognosis",
                decision="abstain",
                detection=DetectionResult(abnormal=None, method="capacity_trajectory"),
                confidence=0.0,
                uncertainty=1.0,
                abstained=True,
                abstain_reason="No capacity history is available.",
                recommended_actions=["Collect discharge-capacity history before prognosis."],
            )
            result.validate()
            return result

        features = capacity_health_features(
            x,
            y,
            nominal_capacity_ah=self.nominal_capacity_ah,
            slope_window=self.slope_window,
        )
        prognosis = estimate_capacity_rul(
            x,
            y,
            eol_capacity_ah=self.eol_capacity_ah,
            nominal_capacity_ah=self.nominal_capacity_ah,
            minimum_observations=self.minimum_observations,
            slope_window=self.slope_window,
        )

        evidence = [
            Evidence(
                source="capacity_health_features",
                statement=(
                    f"SOH={features['soh']:.3f}, local capacity slope="
                    f"{features['local_slope_ah_per_cycle'] if features['local_slope_ah_per_cycle'] is not None else float('nan'):.6f} Ah/cycle."
                ),
                score=float(np.clip(1.0 - abs(float(features["soh"] or 0.0) - 0.7), 0.0, 1.0)),
                evidence_id="battery-capacity-trajectory",
                kind="prognostic",
                provenance={"battery_id": battery_id, "observed_cycles": int(x.size)},
                details=features,
            )
        ]
        trace = [ToolTraceStep(tool="capacity_health_features", evidence_ids=["battery-capacity-trajectory"])]

        if features.get("knee_cycle") is not None:
            evidence.append(
                Evidence(
                    source="capacity_knee_detection",
                    statement=f"A degradation knee candidate is detected near cycle {features['knee_cycle']}.",
                    score=float(np.clip(float(features.get("knee_score") or 0.0) / 2.0, 0.0, 1.0)),
                    evidence_id="battery-knee",
                    kind="prognostic",
                    provenance={"battery_id": battery_id},
                    details={"knee_cycle": features["knee_cycle"], "knee_score": features["knee_score"]},
                )
            )
            trace.append(ToolTraceStep(tool="capacity_knee_detection", evidence_ids=["battery-knee"]))

        if prognosis["abstained"]:
            result = DiagnosticResult(
                domain="battery",
                task="prognosis",
                decision="abstain",
                detection=DetectionResult(
                    abnormal=(None if features["soh"] is None else bool(features["soh"] <= 0.8)),
                    score=(None if features["soh"] is None else float(np.clip(1.0 - features["soh"], 0.0, 1.0))),
                    method="capacity_soh",
                ),
                evidence=evidence,
                prognosis=PrognosisResult(
                    risk=None,
                    horizon="cycles_to_1.4Ah_EOL",
                    remaining_useful_life=None,
                    uncertainty=None,
                    details=prognosis,
                ),
                confidence=0.2,
                uncertainty=0.8,
                abstained=True,
                abstain_reason=str(prognosis["abstain_reason"]),
                recommended_actions=["Collect more discharge cycles or wait for a stable degradation trend before RUL prediction."],
                tool_trace=trace + [ToolTraceStep(tool="estimate_capacity_rul", status="warning")],
                metadata={"battery_id": battery_id, "features": features, "prognosis": prognosis},
            )
            result.validate()
            return result

        uncertainty_cycles = float(prognosis["uncertainty_cycles"])
        rul = float(prognosis["remaining_useful_life_cycles"])
        relative_uncertainty = uncertainty_cycles / max(rul + uncertainty_cycles, 1.0)
        confidence = float(np.clip(1.0 - relative_uncertainty, 0.0, 1.0))
        evidence_ids = [row.evidence_id for row in evidence if row.evidence_id]
        hypothesis = DiagnosticHypothesis(
            label="capacity_fade",
            score=confidence,
            rationale="Capacity trajectory supports a degradation trend toward the NASA EOL threshold.",
            evidence_ids=evidence_ids,
            details={"soh": features["soh"], "rul_cycles": rul},
        )
        verification = VerificationResult(
            hypothesis="capacity_fade",
            status="SUPPORTED",
            reason="Recent capacity slope is negative and supports bounded EOL extrapolation.",
            evidence_ids=evidence_ids,
            verifier="capacity_trajectory_physics",
        )

        result = DiagnosticResult(
            domain="battery",
            task="prognosis",
            decision="monitor",
            detection=DetectionResult(
                abnormal=bool(float(features["soh"]) <= 0.8),
                score=float(np.clip(1.0 - float(features["soh"]), 0.0, 1.0)),
                method="capacity_soh",
                details={"soh": features["soh"]},
            ),
            hypotheses=[hypothesis],
            evidence=evidence,
            verification=[verification],
            prognosis=PrognosisResult(
                risk=float(prognosis["risk"]),
                horizon="cycles_to_1.4Ah_EOL",
                remaining_useful_life=rul,
                uncertainty=float(np.clip(relative_uncertainty, 0.0, 1.0)),
                details={
                    "predicted_eol_cycle": prognosis["predicted_eol_cycle"],
                    "uncertainty_cycles": uncertainty_cycles,
                    "method": prognosis["method"],
                },
            ),
            confidence=confidence,
            uncertainty=float(np.clip(1.0 - confidence, 0.0, 1.0)),
            abstained=False,
            recommended_actions=["Continue capacity monitoring and recompute RUL as additional discharge cycles arrive."],
            tool_trace=trace + [ToolTraceStep(tool="estimate_capacity_rul", evidence_ids=evidence_ids)],
            metadata={"battery_id": battery_id, "features": features, "prognosis": prognosis},
        )
        result.validate()
        return result
