# Diagnostic Architecture Migration Status

This document records which benchmark paths currently exercise the same diagnostic boundary exposed to callers. It is intentionally conservative: a benchmark is marked aligned only when its actual inference call crosses `DiagnosticRequest -> diagnose() -> DomainPlugin -> Workflow -> DecisionPolicy` without moving labels or future information into the request.

## Stable public boundary

The repository now provides:

- ordered `ExecutionTrace` with chronological `steps`, named `get()` and `require()` lookup;
- versioned `Step` and `Workflow` execution;
- `DiagnosticRequest` and `RunContext` contracts;
- `DomainPlugin` and `DomainRegistry` abstractions;
- versioned `PolicyRegistry` selection through `policy_ref`;
- `ModelRegistry` records for explicit model/version provenance;
- separate `workflow_version`, `policy_version`, and `model_version` provenance.

The legacy string API remains available during migration. Bearing, Process, Battery, Turbofan, and Transformer legacy calls are internally routed through the plugin boundary. The original simple Wind legacy call remains a compatibility path because it does not contain the separate healthy-reference and prediction matrices required by the richer Wind workflow.

## Benchmark alignment

| Domain | Dataset / protocol | Public diagnose boundary | Status / caveat |
| --- | --- | --- | --- |
| Wind SCADA | CARE to Compare v6, 95 events | Yes | Real-data parity confirmed. Current event policy has very high normal-event FAR and must not be tuned on CARE labels. |
| Bearing | CWRU fixed 0.007-in drive-end set | Yes | Locked benchmark metrics reproduced through `diagnose()`. |
| Battery prognosis | NASA B0005/B0006/B0007/B0018 capacity-history protocol | Yes | Exact point, interval, and censoring metrics reproduced through task-specific `prognosis` plugin path. Evaluated baseline, not research-ready. |
| Turbofan RUL | NASA C-MAPSS FD001-FD004 | Yes | Train-only learned RUL model remains fit before test labels are loaded; 707-engine locked metrics reproduced through `diagnose()`. |
| Process | TEP calibration/root-cause ablation | Partly | Public process diagnosis is plugin-routed, but the canonical TEP workflow is a research calibration + ablation protocol and intentionally remains separate rather than being mislabeled as the public system. |
| Transformer | SGAH | Plugin boundary only | Do not rerun or tune against the repeatedly inspected frozen SGAH test. Hybrid architecture requires independent richer transformer/protection data for genuine validation. |

## Confirmed parity checkpoints

### CARE / Wind

- 95 successful events, 0 benchmark failures.
- Event recall: `0.911111`.
- Event precision: `0.455556`.
- Event F1: `0.607407`.
- Normal-event false-alarm rate: `0.98`.
- Abstention rate: `0.031579`.

The high false-alarm rate is a scientific limitation, not an architecture-migration defect. Future policy work must use healthy-only calibration or a fresh validation protocol rather than CARE-label tuning.

### CWRU / Bearing

- all-record accuracy: `0.3125`;
- macro F1: `0.35`;
- coverage: `0.3125`;
- abstention: `0.6875`;
- covered accuracy: `1.0`;
- normal FPR: `0`;
- fault detection rate: `0.416667`.

### NASA Battery

- exact point coverage: `0.9`;
- MAE: `14.891853` cycles;
- RMSE: `17.917658` cycles;
- exact interval coverage: `0.777778`;
- censored coverage: `1.0`;
- censored interval compatibility: `0.5`;
- mean one-sided violation: `6.686877` cycles;
- max one-sided violation: `18.831462` cycles.

### C-MAPSS / Turbofan

- 707 cases, 707 covered;
- coverage: `1.0`;
- MAE: `24.084955` cycles;
- RMSE: `33.761282` cycles;
- signed error: `10.873314` cycles;
- FD001 RMSE: `26.862615`;
- FD002 RMSE: `28.819175`;
- FD003 RMSE: `36.471670`;
- FD004 RMSE: `39.429718`.

## Intentionally deferred cleanup

Legacy runners and adapters are **not** removed yet. They remain useful as the scientifically frozen implementations wrapped by plugins and as compatibility surfaces for existing callers. Removing them before TEP representation, independent transformer validation, and Wind legacy-input migration are resolved would create churn without improving scientific validity.

The next architectural cleanup should happen only after:

1. the remaining compatibility workflow steps are decomposed into stable named steps without changing frozen outputs;
2. TEP clearly separates research calibration/ablation from the deployable process diagnosis path;
3. Transformer is validated on independent protection data;
4. calibrated uncertainty/abstention is implemented where scientifically justified;
5. a unified cross-domain report records dataset/version/protocol/workflow/policy/model versions and commit SHA.

Only after that boundary is stable should an LLM/RL planner choose workflows or tools above `diagnose()`. Mandatory verification and safety policies must remain outside planner control.
