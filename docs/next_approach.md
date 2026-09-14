# Pipeline-first implementation record

Pipeline version `1.0.0` now provides a usable end-to-end diagnostic path for each of the six domain packs. Benchmark optimization follows this API freeze. Existing benchmark commands remain regression checks; thresholds and scoring weights were not tuned as part of the pipeline implementation.

## Completed state

- `diagnose` and `DiagnosticPipeline` provide the shared public entry point and return `DiagnosticResult` for all domains.
- Bearing and process run through their existing implementations; the process result is adapted without removing its artifacts.
- Wind SCADA, battery, turbofan, and transformer have deterministic end-to-end runners with explicit optional model hooks.
- `IndustrialDiagnosticOrchestrator` remains available for compatibility with the older agent API.

## 1. Executable contract

- Define a public entry point that accepts a domain, task, signal data, metadata, and optional references/models. It should validate inputs, select the domain pack, and return one `DiagnosticResult`.
- Extend planning to distinguish implemented, optional, and unavailable steps. The executor should pass named outputs between tools, record evidence and timing in `tool_trace`, and give a specific abstain reason when required inputs or implementations are missing.
- Keep detection, localization, diagnosis, verification, and prognosis as separate result fields. Warnings about domain shift or future risk must not become fault labels by themselves.
- Preserve the legacy API through an adapter while migrating callers; document which entry point new users should call.

**Done when:** one documented call can execute a selected domain, return a validated `DiagnosticResult`, and explain every skipped or failed step.

## 2. Bearing and process integration

- Register `BearingDiagnosticPipeline` and `ProcessDiagnosticPipeline` behind the new entry point. Convert the process result to `DiagnosticResult` without losing its root-cause ranking, affected variables, abstention reason, or intermediate artifacts.
- Make the top-level bearing path use the pipeline's evidence thresholds and abstention behavior. A ranked harmonic family alone is not enough for a diagnosis.
- Add end-to-end tests for a valid signal, weak evidence, missing metadata, and contradicted verification. Check the returned decision, evidence IDs, and trace, not just whether execution succeeds.

**Done when:** bearing and process use the same public result shape and their normal, fault, and abstain paths work through the public entry point.

## 3. Remaining domain paths

- Implement each contracted path in this order: wind SCADA, battery, turbofan, transformer. Start with deterministic data-quality, feature, detection, and decision steps that can run without a trained model.
- Treat learned models, physics models, and external evidence as optional adapters with explicit input/output contracts. When a required capability is absent, return a meaningful abstention instead of a fabricated diagnosis or prognosis.
- Add a small end-to-end fixture for each domain, covering successful execution and a missing-input/unsupported-capability case.

**Done when:** all six domain packs can execute from input validation to a structured decision or an explicit abstention, with evidence and trace. Model-dependent claims are made only when the necessary model is supplied.

## 4. Integration and documentation

- Add one example per domain and a capabilities table to `README.md`, distinguishing runnable steps from optional model-backed steps.
- Keep unit and integration tests green, and run existing CWRU/TEP workflows to catch regressions. Record results without changing benchmark thresholds during this phase.
- Review output schema stability and error handling before treating the pipeline as the default project API.

**Done when:** a user can identify the required inputs, run each supported path, and understand its decision or abstention from the returned evidence.

## 5. Improve benchmarks afterward

- Freeze the completed pipeline version and record CWRU/TEP baselines. Expand evaluation to the other domains as real datasets become available.
- Improve calibration, false alarms, coverage, localization, and runtime using separate calibration and held-out evaluation units. Preserve fault-level and record-level separation to avoid leakage.
- Compare changes against the frozen baseline; do not further tune on the same TEP fault cases used to select the current root-ranking weights.
