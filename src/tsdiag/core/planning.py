from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..contracts import DomainPack
from .context import RunContext


@dataclass(frozen=True)
class PlanStep:
    tool: str
    reason: str = "contracted domain step"
    required: bool = True


@dataclass(frozen=True)
class ExecutionPlan:
    domain: str
    task: str
    steps: tuple[PlanStep, ...]
    missing_required_metadata: tuple[str, ...] = ()
    policy: str = "deterministic"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return not self.missing_required_metadata

    @property
    def tool_sequence(self) -> tuple[str, ...]:
        return tuple(step.tool for step in self.steps)


class RouterPolicy(Protocol):
    name: str

    def plan(self, pack: DomainPack, context: RunContext) -> ExecutionPlan:
        ...


class DeterministicRouter:
    """Ordered DomainPack baseline used as the scientific control router."""

    name = "deterministic"

    def plan(self, pack: DomainPack, context: RunContext) -> ExecutionPlan:
        metadata = context.metadata
        missing = tuple(name for name in pack.required_metadata if metadata.get(name) is None)
        return ExecutionPlan(
            domain=pack.key,
            task=context.request.task,
            steps=tuple(PlanStep(tool=name) for name in pack.tool_names()),
            missing_required_metadata=missing,
            policy=self.name,
        )
