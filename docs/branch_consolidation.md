# Branch consolidation

This repository uses `main` as the canonical development line after the September 2026 branch cleanup.

## Consolidated history

The feature branches behind merged PRs #1-#10 and #12-#14 are historical only; their accepted work is already represented in `main` and must not be merged again.

The Wind-SCADA branch family was reconciled by taking the latest CARE/real-benchmark implementation from `feat/wind-scada-real-benchmark` and layering it onto the newer six-domain public pipeline already on `main`. The public generic `WindScadaDiagnosticPipeline` remains the stable cross-domain API. The CARE-specific real-data implementation is exposed separately as `WindScadaBenchmarkPipeline` so the two runtimes do not shadow one another.

PR #17 (`arch/platform-v1-clean`) is superseded by the newer public pipeline/execution architecture on `main`. It is intentionally not merged wholesale because its last CI run failed on a circular import and merging it would create a second competing runtime abstraction.

PR #15 (`feat/battery-nasa-prognostics`) is retained as a frozen scientific baseline and is not merged. The NASA protocol is technically valid, but the local-linear prognostic method has a documented B0007 right-censoring consistency failure; its thresholds and method parameters must not be post-hoc tuned on those four evaluation cells.

## Development rule

After this consolidation, new work should be committed directly to `main`. GitHub Actions `Unit Tests` runs the full test suite on every push to `main`. Dataset-specific real-data workflows remain additional scientific gates.
