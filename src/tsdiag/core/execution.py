from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from ..contracts import DomainPack, ToolRegistry
from ..models import Evidence, ToolTraceStep
from .context import RunContext
from .evidence import EvidenceStore
from .planning import ExecutionPlan


@dataclass
class ToolOutcome:
    status: Literal["ok", "warning", "error", "skipped", "abstain"] = "ok"
    outputs: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    context: RunContext
    evidence: list[Evidence]
    trace: list[ToolTraceStep]
    abstained: bool = False
    abstain_reason: str | None = None


class ExecutionEngine:
    """Policy- and domain-independent executor for contracted tools."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, pack: DomainPack, plan: ExecutionPlan, context: RunContext) -> ExecutionResult:
        if not plan.ready:
            return ExecutionResult(
                context=context,
                evidence=[],
                trace=[],
                abstained=True,
                abstain_reason=(
                    "Missing required metadata: " + ", ".join(plan.missing_required_metadata)
                ),
            )

        contracts = {tool.name: tool for tool in pack.tools}
        evidence_store = EvidenceStore()
        trace: list[ToolTraceStep] = []

        for step in plan.steps:
            contract = contracts.get(step.tool)
            if contract is None:
                raise KeyError(f"Plan references tool not declared by domain pack: {step.tool}")

            missing_inputs = tuple(name for name in contract.required_inputs if not context.has(name))
            if missing_inputs:
                trace.append(
                    ToolTraceStep(
                        tool=step.tool,
                        status="skipped",
                        details={"reason": "missing_inputs", "missing": list(missing_inputs)},
                    )
                )
                if step.required:
                    return ExecutionResult(
                        context=context,
                        evidence=evidence_store.all(),
                        trace=trace,
                        abstained=True,
                        abstain_reason=(
                            f"Required tool {step.tool!r} missing inputs: {', '.join(missing_inputs)}"
                        ),
                    )
                continue

            fn = self.registry.get(step.tool)
            if fn is None:
                trace.append(
                    ToolTraceStep(
                        tool=step.tool,
                        status="skipped",
                        details={"reason": "implementation_unavailable"},
                    )
                )
                if step.required:
                    return ExecutionResult(
                        context=context,
                        evidence=evidence_store.all(),
                        trace=trace,
                        abstained=True,
                        abstain_reason=f"Required tool {step.tool!r} has no registered implementation.",
                    )
                continue

            inputs = {name: context.get(name) for name in contract.required_inputs}
            inputs.update(
                {name: context.get(name) for name in contract.optional_inputs if context.has(name)}
            )

            started = perf_counter()
            try:
                raw = fn(**inputs)
                outcome = self._normalize_outcome(raw, contract.outputs)
            except Exception as exc:
                duration = perf_counter() - started
                trace.append(
                    ToolTraceStep(
                        tool=step.tool,
                        status="error",
                        inputs_summary={"keys": sorted(inputs)},
                        duration_seconds=duration,
                        details={"exception": type(exc).__name__, "message": str(exc)},
                    )
                )
                return ExecutionResult(
                    context=context,
                    evidence=evidence_store.all(),
                    trace=trace,
                    abstained=True,
                    abstain_reason=f"Tool {step.tool!r} failed: {exc}",
                )

            duration = perf_counter() - started
            context.publish(outcome.outputs)
            stored_evidence = evidence_store.extend(outcome.evidence, tool=step.tool)
            trace_status = "warning" if outcome.status in {"warning", "abstain"} else outcome.status
            trace.append(
                ToolTraceStep(
                    tool=step.tool,
                    status=trace_status,
                    inputs_summary={"keys": sorted(inputs)},
                    outputs_summary={"keys": sorted(outcome.outputs)},
                    evidence_ids=[row.evidence_id for row in stored_evidence if row.evidence_id],
                    duration_seconds=duration,
                    details={**outcome.details, **({"message": outcome.message} if outcome.message else {})},
                )
            )

            if outcome.status == "abstain":
                return ExecutionResult(
                    context=context,
                    evidence=evidence_store.all(),
                    trace=trace,
                    abstained=True,
                    abstain_reason=outcome.message or f"Tool {step.tool!r} abstained.",
                )

        return ExecutionResult(context=context, evidence=evidence_store.all(), trace=trace)

    @staticmethod
    def _normalize_outcome(raw: Any, declared_outputs: tuple[str, ...]) -> ToolOutcome:
        if isinstance(raw, ToolOutcome):
            return raw
        if isinstance(raw, dict):
            return ToolOutcome(outputs=raw)
        if len(declared_outputs) == 1:
            return ToolOutcome(outputs={declared_outputs[0]: raw})
        if raw is None and not declared_outputs:
            return ToolOutcome()
        raise TypeError(
            "Tool must return ToolOutcome, dict, or a scalar when exactly one output is declared"
        )
