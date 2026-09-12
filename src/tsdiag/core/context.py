from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunRequest:
    """Canonical cross-domain runtime request.

    `observation` is intentionally opaque to the core runtime. Domain adapters and
    tools interpret it through the selected DomainPack contracts.
    """

    domain: str
    task: str
    observation: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    cutpoint: int | float | str | None = None
    horizon: int | float | str | None = None
    run_id: str | None = None

    def validate(self) -> None:
        if not self.domain.strip():
            raise ValueError("domain must be non-empty")
        if not self.task.strip():
            raise ValueError("task must be non-empty")
        if self.observation is None:
            raise ValueError("observation must not be None")


@dataclass
class RunContext:
    request: RunRequest
    state: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(cls, request: RunRequest) -> "RunContext":
        request.validate()
        return cls(
            request=request,
            state={"observation": request.observation},
            metadata=dict(request.metadata),
        )

    def get(self, name: str, default: Any = None) -> Any:
        if name in self.state:
            return self.state[name]
        if name in self.metadata:
            return self.metadata[name]
        return default

    def has(self, name: str) -> bool:
        return name in self.state or name in self.metadata

    def publish(self, values: dict[str, Any]) -> None:
        self.state.update(values)
