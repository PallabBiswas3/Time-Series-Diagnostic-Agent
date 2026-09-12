from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .core.context import RunRequest
from .domains.runtime_adapters import run_bearing_request, run_process_request
from .models import DiagnosticResult

DomainAdapter = Callable[[RunRequest], DiagnosticResult]


@dataclass
class DiagnosticService:
    """Public request-level facade over domain runtime adapters.

    This is intentionally thin. Domain science remains in domain pipelines/tools,
    while callers get one stable `run(RunRequest) -> DiagnosticResult` API.
    """

    adapters: dict[str, DomainAdapter] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "DiagnosticService":
        return cls(
            adapters={
                "bearing": run_bearing_request,
                "process": run_process_request,
            }
        )

    def register(self, domain: str, adapter: DomainAdapter) -> None:
        if not domain.strip():
            raise ValueError("domain must be non-empty")
        if not callable(adapter):
            raise TypeError("adapter must be callable")
        self.adapters[domain] = adapter

    def run(self, request: RunRequest) -> DiagnosticResult:
        request.validate()
        adapter = self.adapters.get(request.domain)
        if adapter is None:
            available = ", ".join(sorted(self.adapters)) or "none"
            raise KeyError(
                f"No runtime adapter is registered for domain {request.domain!r}. "
                f"Available: {available}"
            )
        result = adapter(request)
        result.validate()
        return result
