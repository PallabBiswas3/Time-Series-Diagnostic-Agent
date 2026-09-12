from __future__ import annotations

from dataclasses import dataclass

from ..contracts import ToolRegistry
from ..domains import get_domain_pack
from .context import RunContext, RunRequest
from .execution import ExecutionEngine, ExecutionResult
from .planning import DeterministicRouter, RouterPolicy


@dataclass
class DiagnosticRuntime:
    """Single entry point for policy-independent cross-domain execution."""

    registry: ToolRegistry
    router: RouterPolicy | None = None

    def __post_init__(self) -> None:
        if self.router is None:
            self.router = DeterministicRouter()
        self.engine = ExecutionEngine(self.registry)

    def plan(self, request: RunRequest):
        context = RunContext.from_request(request)
        pack = get_domain_pack(request.domain)
        return self.router.plan(pack, context)

    def run(self, request: RunRequest) -> ExecutionResult:
        context = RunContext.from_request(request)
        pack = get_domain_pack(request.domain)
        plan = self.router.plan(pack, context)
        return self.engine.run(pack, plan, context)
