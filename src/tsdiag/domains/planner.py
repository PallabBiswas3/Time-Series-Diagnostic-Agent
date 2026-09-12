from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import DomainPack
from ..core.context import RunContext, RunRequest
from ..core.planning import DeterministicRouter
from . import DOMAIN_PACKS


@dataclass(frozen=True)
class DomainExecutionPlan:
    domain: str
    tool_sequence: tuple[str, ...]
    missing_required_metadata: tuple[str, ...]
    available_optional_metadata: tuple[str, ...]
    ready: bool


def build_execution_plan(domain: str, metadata: dict[str, Any] | None = None) -> DomainExecutionPlan:
    metadata = metadata or {}
    if domain not in DOMAIN_PACKS:
        raise KeyError(f"Unknown domain {domain!r}. Available: {sorted(DOMAIN_PACKS)}")

    pack = DOMAIN_PACKS[domain]
    request = RunRequest(domain=domain, task="diagnosis", observation=object(), metadata=metadata)
    context = RunContext.from_request(request)
    plan = DeterministicRouter().plan(pack, context)
    optional = tuple(name for name in pack.optional_metadata if metadata.get(name) is not None)

    return DomainExecutionPlan(
        domain=plan.domain,
        tool_sequence=plan.tool_sequence,
        missing_required_metadata=plan.missing_required_metadata,
        available_optional_metadata=optional,
        ready=plan.ready,
    )


def describe_contract(pack: DomainPack) -> list[dict[str, object]]:
    return [
        {
            "name": tool.name,
            "purpose": tool.purpose,
            "required_inputs": list(tool.required_inputs),
            "optional_inputs": list(tool.optional_inputs),
            "outputs": list(tool.outputs),
            "preconditions": list(tool.preconditions),
            "evidence_fields": list(tool.evidence_fields),
            "failure_modes": list(tool.failure_modes),
            "implementation": tool.implementation,
        }
        for tool in pack.tools
    ]
