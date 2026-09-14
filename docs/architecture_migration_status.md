# Diagnostic Architecture Migration Status

This document records which benchmark paths currently exercise the same diagnostic boundary exposed to callers. A benchmark is marked aligned only when its actual inference call crosses `DiagnosticRequest -> diagnose() -> DomainPlugin -> Workflow -> DecisionPolicy` without moving labels or future information into the request.

## Stable public boundary

The repository now provides:

- ordered hierarchical `ExecutionTrace` with chronological `steps`, recursive `get()` / `require()` lookup, and child traces;
- versioned `Step` and `Workflow` execution;
- `DiagnosticRequest` and `RunContext` contracts;
- first-class `UncertaintyEstimate`, `Abstention`, and `RunProvenance` result contracts;
- `DomainPlugin` and `DomainRegistry` abstractions;
- authoritative versioned `PolicyRegistry` selection through `policy_ref`;
- validated `ModelRegistry` records with artifact injection, version metadata, and optional checksums;
- automatic run provenance including timestamp, deterministic input hash, dataset/protocol IDs when supplied, artifact checksums, Git SHA, workflow version, policy version, and model versions.

The legacy string API remains available during migration. Bearing, Process, Battery pack diagnosis, Battery capacity prognosis, Turbofan, and Transformer calls are routed through the public diagnostic boundary. The original simple Wind legacy call remains a compatibility path because it does not contain the separate healthy-reference and prediction matrices required by the richer Wind benchmark workflow.

## Workflow migration state

| Domain | Current execution structure | Status / caveat |
| --- | --- | --- |
| Bearing | Fully decomposed named workflow | Shared executor owns signal integrity, features, PSD, spectral kurtosis, filtering, envelope analysis, frequency matching, and fusion. |
| Process | Decomposed deployable workflow | Existing PCA/contribution/temporal/causal/root-cause science is expressed as named shared-executor steps. The research TEP calibration/ablation protocol remains intentionally separate. |
| Battery pack | Decomposed named workflow | Uses the canonical Battery `DomainPack` tool names and `battery-pack-policy-v2`. |
| Battery capacity prognosis | Single task-specific prognostic step | Intentionally atomic because capacity-history prognosis is itself the task. Workflow version is `prognosis-1.0`; policy version is `capacity-prognosis-policy-v1`. |
| Transformer | Decomposed deployable workflow | Shared executor owns synchronization through decision. SGAH is not used for further architecture/method selection. |
| Turbofan | Compatibility wrapper remains | The next domain to migrate fully into named shared-executor steps. |
| Wind SCADA | Rich plugin workflow plus simple legacy compatibility path | CARE benchmark path is plugin aligned; simple legacy inputs retain a compatibility runner. |

## Benchmark alignment

| Domain | Dataset / protocol | Public diagnose boundary | Status / caveat |
| --- | --- | --- | --- |
| Wind SCADA | CARE to Compare v6, 95 events | Yes | Real-data parity confirmed. Current event policy has very high normal-event FAR and must not be tuned on CARE labels. |
| Bearing | CWRU fixed 0.007-in drive-end set | Yes | Locked benchmark metrics reproduced through the decomposed `diagnose()` path. Exact parity gates protect against silent coverage regressions. |
| Battery prognosis | NASA B0005/B0006/B0007/B0018 capacity-history protocol | Yes | Exact point, interval, and censoring metrics reproduced through task-specific `prognosis` path. Evaluated baseline, not research-ready. |
| Turbofan RUL | NASA C-MAPSS FD001-FD004 | Yes | Train-only learned RUL model remains fit before test labels are loaded; 707-engine locked metrics reproduced through `diagnose()`. |
| Process | TEP calibration/root-cause ablation | Partly | Deployable process diagnosis is decomposed and plugin-routed, while the canonical TEP calibration + ablation protocol intentionally remains a separate research evaluator. |
| Transformer | SGAH | Plugin boundary only | Frozen SGAH real-data evaluation is manual-only. Do not rerun or tune against the repeatedly inspected test set; future methodology requires independent richer transformer/protection data. |

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

## Remaining structural cleanup

The architecture backbone is now the primary execution model, but migration-era code still creates ownership ambiguity. In particular, `domains/` contains active plugins alongside `compat_plugins.py` and legacy runners, and the repository still has overlapping `domain/` / `domains/` organization that should eventually be consolidated.

Next cleanup order:

1. fully decompose Turbofan into named shared-executor steps while preserving the locked C-MAPSS method and label isolation;
2. migrate or isolate the remaining Wind simple-input compatibility path;
3. remove compatibility classes and legacy dispatch paths only after no active caller or benchmark depends on them;
4. consolidate `domain/` versus `domains/` ownership and imports;
5. add calibrated uncertainty/abstention only where a scientifically valid calibration protocol exists;
6. produce one cross-domain report recording dataset/version/protocol/workflow/policy/model versions, artifact checksums, and commit SHA.

Only after that boundary is stable should an LLM/RL planner choose workflows or tools above `diagnose()`. Mandatory verification and safety policies must remain outside planner control.
