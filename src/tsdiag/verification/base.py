from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.context import RunContext
from ..models import DiagnosticHypothesis, VerificationResult


class Verifier(Protocol):
    name: str

    def verify(self, hypothesis: DiagnosticHypothesis, context: RunContext) -> VerificationResult:
        ...


@dataclass
class VerificationSuite:
    verifiers: tuple[Verifier, ...] = ()

    def run(self, hypotheses: list[DiagnosticHypothesis], context: RunContext) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        for hypothesis in hypotheses:
            for verifier in self.verifiers:
                results.append(verifier.verify(hypothesis, context))
        return results

    @staticmethod
    def contradicted(results: list[VerificationResult]) -> bool:
        return any(row.status == "CONTRADICTED" for row in results)
