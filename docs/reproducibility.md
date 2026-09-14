# Reproducibility Guide

## Environment

The supported reference environment is Python 3.11.

```bash
git clone https://github.com/PallabBiswas3/Time-Series-Diagnostic-Agent.git
cd Time-Series-Diagnostic-Agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

## Public API smoke test

```bash
python examples/final_demo.py
```

Every public run returns the same `DiagnosticResult` contract with:

- `decision`: `diagnose`, `monitor`, or `abstain`
- detection and localization fields
- hypotheses and evidence
- verification when available
- prognosis when available
- confidence and uncertainty
- explicit abstention reason
- auditable tool trace
- standardized execution/evidence/result summaries in metadata

## Real-data benchmark workflows

### Tennessee Eastman Process

The workflow `.github/workflows/tep-benchmark.yml` downloads the fixed Braatz TEP data, runs PCA/DPCA calibration, root-cause evaluation and regression gates, and uploads the result artifact.

It runs automatically for relevant process/TEP changes on `main` and can also be dispatched manually.

### CWRU Bearing

The workflow `.github/workflows/bearing-cwru-benchmark.yml` uses the fixed 0.007-inch drive-end CWRU manifest across four loads, validates the files, runs the fixed window protocol, and uploads JSON/CSV artifacts.

It runs automatically for relevant bearing changes on `main`.

### CARE Wind SCADA

The workflow `.github/workflows/wind-scada-benchmark.yml` runs the CARE-to-Compare v6 real-data benchmark and wind-specific unit checks.

## Scientific-change rule

Do not tune a method using the same frozen benchmark that will be used to claim final performance. Changes to thresholds, features, scoring, censoring rules or model selection should be declared before benchmark execution and developed on independent development data whenever possible.

Architecture-only changes should preserve the frozen protocol and be checked as regressions rather than treated as new performance experiments.

## Artifact expectations

A benchmark run should retain:

1. protocol/configuration,
2. per-record or per-fault outputs,
3. aggregate metrics,
4. failures/abstentions,
5. runtime information,
6. the exact commit SHA.

This makes benchmark claims auditable and reproducible rather than dependent on screenshots or manually copied numbers.
