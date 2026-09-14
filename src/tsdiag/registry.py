from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import DomainPlugin


@dataclass
class DomainRegistry:
    _plugins: dict[str, DomainPlugin] = field(default_factory=dict)

    def register(self, plugin: DomainPlugin) -> None:
        name = str(plugin.name).strip().lower()
        if not name:
            raise ValueError("domain plugin name must be non-empty")
        self._plugins[name] = plugin

    def resolve(self, domain: str) -> DomainPlugin:
        key = str(domain).strip().lower()
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise KeyError(f"No domain plugin registered for {key!r}; available: {sorted(self._plugins)}") from exc

    def get(self, domain: str) -> DomainPlugin | None:
        return self._plugins.get(str(domain).strip().lower())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))


domain_registry = DomainRegistry()
