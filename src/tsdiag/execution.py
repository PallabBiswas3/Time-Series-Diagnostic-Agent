from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from .models import ToolTraceStep, json_safe


class StepExecutionError(RuntimeError):
    def __init__(self, step: str, message: str, trace: list[ToolTraceStep]):
        super().__init__(f"{step}: {message}")
        self.step = step
        self.trace = trace


@dataclass(frozen=True)
class InputField:
    name: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class DomainInputSchema:
    domain: str
    fields: tuple[InputField, ...]

    def validate(self, values: dict[str, Any]) -> None:
        missing = [row.name for row in self.fields if row.required and values.get(row.name) is None]
        if missing:
            raise ValueError(f"Missing required inputs: {', '.join(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "fields": [json_safe(row.__dict__) for row in self.fields]}


@dataclass
class DomainToolRegistry:
    implementations: dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]] = field(default_factory=dict)

    def register(self, domain: str, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        if not callable(fn):
            raise TypeError(f"Implementation for {domain}.{name} must be callable")
        self.implementations[(domain, name)] = fn

    def get(self, domain: str, name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        try:
            return self.implementations[(domain, name)]
        except KeyError as exc:
            raise KeyError(f"No registered implementation for {domain}.{name}") from exc

    def names(self, domain: str) -> tuple[str, ...]:
        return tuple(name for pack, name in self.implementations if pack == domain)


class StepExecutor:
    """Execute registered tools with isolated errors and auditable timing."""

    def __init__(self, domain: str, registry: DomainToolRegistry):
        self.domain = domain
        self.registry = registry
        self.trace: list[ToolTraceStep] = []

    def run(self, name: str, state: dict[str, Any], *, optional: bool = False) -> dict[str, Any]:
        started = perf_counter()
        input_keys = sorted(state.keys())
        try:
            output = self.registry.get(self.domain, name)(state)
            if not isinstance(output, dict):
                raise TypeError("tool must return a dictionary")
            duration = perf_counter() - started
            state.update(output)
            self.trace.append(ToolTraceStep(
                tool=name, status="ok", duration_seconds=duration,
                inputs_summary={"available_keys": input_keys},
                outputs_summary={"output_keys": sorted(output.keys())},
            ))
            return output
        except Exception as exc:
            trace = ToolTraceStep(
                tool=name, status="skipped" if optional else "error",
                duration_seconds=perf_counter() - started,
                details={"error_type": type(exc).__name__, "message": str(exc)},
            )
            self.trace.append(trace)
            if optional:
                return {}
            raise StepExecutionError(name, str(exc), list(self.trace)) from exc
