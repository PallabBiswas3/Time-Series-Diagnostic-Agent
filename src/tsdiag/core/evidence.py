from __future__ import annotations

from dataclasses import replace

from ..models import Evidence


class EvidenceStore:
    """Run-local auditable evidence registry with stable identifiers."""

    def __init__(self) -> None:
        self._rows: list[Evidence] = []
        self._ids: set[str] = set()
        self._counter = 0

    def add(self, evidence: Evidence, *, tool: str | None = None) -> Evidence:
        evidence_id = evidence.evidence_id
        if evidence_id is None:
            self._counter += 1
            evidence_id = f"evidence-{self._counter:04d}"
        if evidence_id in self._ids:
            raise ValueError(f"duplicate evidence_id: {evidence_id}")

        provenance = dict(evidence.provenance)
        if tool is not None:
            provenance.setdefault("tool", tool)
        stored = replace(evidence, evidence_id=evidence_id, provenance=provenance)
        self._rows.append(stored)
        self._ids.add(evidence_id)
        return stored

    def extend(self, rows: list[Evidence], *, tool: str | None = None) -> list[Evidence]:
        return [self.add(row, tool=tool) for row in rows]

    def all(self) -> list[Evidence]:
        return list(self._rows)

    def ids(self) -> tuple[str, ...]:
        return tuple(row.evidence_id for row in self._rows if row.evidence_id)
