from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.context import RunContext
from ..models import DiagnosticHypothesis, VerificationResult


@dataclass(frozen=True)
class DataQualityVerifier:
    name: str = "data_quality"
    fatal_flags: frozenset[str] = field(
        default_factory=lambda: frozenset({"nan_or_inf", "too_short", "invalid_sampling_rate", "no_finite_samples"})
    )

    def verify(self, hypothesis: DiagnosticHypothesis, context: RunContext) -> VerificationResult:
        flags = set(context.get("quality_flags", []) or [])
        fatal = sorted(flags.intersection(self.fatal_flags))
        if fatal:
            return VerificationResult(
                hypothesis=hypothesis.label,
                status="CONTRADICTED",
                reason=f"Input quality is not sufficient for a reliable diagnosis: {fatal}",
                verifier=self.name,
                details={"fatal_quality_flags": fatal},
            )
        if not flags:
            return VerificationResult(
                hypothesis=hypothesis.label,
                status="INSUFFICIENT",
                reason="No explicit data-quality evidence was supplied.",
                verifier=self.name,
            )
        return VerificationResult(
            hypothesis=hypothesis.label,
            status="SUPPORTED",
            reason="No fatal data-quality flags are present.",
            verifier=self.name,
            details={"quality_flags": sorted(flags)},
        )


@dataclass(frozen=True)
class OODVerifier:
    threshold: float = 0.8
    score_key: str = "ood_score"
    name: str = "ood"

    def verify(self, hypothesis: DiagnosticHypothesis, context: RunContext) -> VerificationResult:
        raw = context.get(self.score_key)
        if raw is None:
            return VerificationResult(
                hypothesis=hypothesis.label,
                status="INSUFFICIENT",
                reason="No out-of-distribution score is available.",
                verifier=self.name,
            )
        score = float(raw)
        if not 0.0 <= score <= 1.0:
            return VerificationResult(
                hypothesis=hypothesis.label,
                status="CONTRADICTED",
                reason="OOD score is outside the required [0, 1] range.",
                verifier=self.name,
                details={"ood_score": score},
            )
        if score >= self.threshold:
            return VerificationResult(
                hypothesis=hypothesis.label,
                status="CONTRADICTED",
                reason=f"Observation is too far outside the supported data regime (OOD={score:.3f}).",
                verifier=self.name,
                details={"ood_score": score, "threshold": self.threshold},
            )
        return VerificationResult(
            hypothesis=hypothesis.label,
            status="SUPPORTED",
            reason="Observation is within the configured OOD acceptance region.",
            verifier=self.name,
            details={"ood_score": score, "threshold": self.threshold},
        )


@dataclass(frozen=True)
class PhysicalBoundsVerifier:
    """Check named runtime values against externally supplied physical bounds.

    Bounds belong in metadata/state and are never learned from the evaluation
    case. Example: {"temperature_c": (-40, 180)}.
    """

    bounds_key: str = "physical_bounds"
    name: str = "physical_bounds"

    def verify(self, hypothesis: DiagnosticHypothesis, context: RunContext) -> VerificationResult:
        bounds = context.get(self.bounds_key)
        if not bounds:
            return VerificationResult(
                hypothesis=hypothesis.label,
                status="INSUFFICIENT",
                reason="No physical bounds were supplied for verification.",
                verifier=self.name,
            )

        violations: dict[str, Any] = {}
        checked = 0
        for key, limits in dict(bounds).items():
            value = context.get(key)
            if value is None:
                continue
            low, high = limits
            checked += 1
            if float(value) < float(low) or float(value) > float(high):
                violations[key] = {"value": float(value), "low": float(low), "high": float(high)}

        if violations:
            return VerificationResult(
                hypothesis=hypothesis.label,
                status="CONTRADICTED",
                reason="One or more runtime values violate supplied physical bounds.",
                verifier=self.name,
                details={"violations": violations},
            )
        if checked == 0:
            return VerificationResult(
                hypothesis=hypothesis.label,
                status="INSUFFICIENT",
                reason="Physical bounds were supplied but none of the bounded variables are available.",
                verifier=self.name,
            )
        return VerificationResult(
            hypothesis=hypothesis.label,
            status="SUPPORTED",
            reason="Available bounded variables satisfy supplied physical limits.",
            verifier=self.name,
            details={"checked_variables": checked},
        )
