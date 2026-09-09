from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np


@dataclass
class SignalRecord:
    signal: np.ndarray
    fs: float
    timestamps: np.ndarray | None = None
    channels: list[str] | None = None
    shaft_rate: float | None = None
    load: float | None = None
    fault_frequencies: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.signal = np.asarray(self.signal, dtype=float)
        if self.signal.ndim == 1:
            self.signal = self.signal[:, None]
        if self.signal.ndim != 2:
            raise ValueError("signal must be [samples] or [samples, channels]")
        if self.signal.shape[0] < 16:
            raise ValueError("signal must contain at least 16 samples")
        if not np.all(np.isfinite(self.signal)):
            raise ValueError("signal contains NaN or infinite values")
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("fs must be a positive finite number")


@dataclass
class Evidence:
    source: str
    statement: str
    score: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent: str
    status: Literal["ok", "warning", "abstain", "unavailable"]
    summary: str
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    decision: Literal["diagnose", "monitor", "abstain"]
    label: str | None
    confidence: float
    agent_results: list[AgentResult]
    evidence: list[Evidence]
    trace: list[str]
