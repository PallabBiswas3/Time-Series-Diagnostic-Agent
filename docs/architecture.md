# Time-Series Diagnostic Agent — Platform Architecture v1

This document freezes the **software and scientific architecture before model tuning**. Benchmark-specific thresholds, cutpoints, labels, and model hyperparameters live outside the architecture layer and must not be changed merely to improve a benchmark result.

## 1. Design goals

The platform should support:

- multiple industrial time-series domains without domain logic leaking into the core runtime;
- deterministic, classical ML, deep-learning, LLM and RL-based routing behind the same interfaces;
- diagnosis, root-cause analysis, anomaly localization, prognosis and RUL;
- auditable evidence and provenance for every conclusion;
- explicit abstention when evidence is insufficient or contradictory;
- frozen, reproducible benchmark protocols that remain separate from model development;
- incremental replacement of simple methods by stronger methods without rewriting datasets, reports or evaluation code.

## 2. Top-level architecture

```text
                    ┌──────────────────────────────┐
                    │  Raw industrial data/source │
                    └──────────────┬───────────────┘
                                   │
                         dataset/domain adapter
                                   │
                    ┌──────────────▼───────────────┐
                    │ Canonical Observation Layer │
                    │ signal/series + metadata    │
                    └──────────────┬───────────────┘
                                   │
                      validation + preprocessing
                                   │
                    ┌──────────────▼───────────────┐
                    │      Domain Pack Contract    │
                    │ tasks, tools, metadata      │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │       Planning / Router      │
                    │ deterministic / ML / RL     │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │       Execution Engine       │
                    │ tool registry + state       │
                    └───────┬───────────┬──────────┘
                            │           │
                  signal/stat tools   learned models
                            │           │
                            └─────┬─────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ Evidence + Hypothesis Layer│
                    │ provenance / uncertainty   │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ Verification / Guardrails  │
                    │ physics, consistency, OOD  │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ DiagnosticResult v1        │
                    │ diagnose/monitor/abstain   │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ Benchmark / API / UI       │
                    └────────────────────────────┘
```

## 3. Layer responsibilities

### A. Dataset adapters

`tsdiag.datasets`

Responsible only for acquiring and parsing external datasets into validated domain observations. Adapters must not contain benchmark-specific model tuning or evaluation logic.

Examples: NASA battery, Tennessee Eastman, CWRU bearing.

### B. Canonical data/observation layer

A domain-independent runtime request carries:

- domain;
- task;
- observation payload;
- metadata;
- requested cutpoint/horizon when relevant;
- reproducibility identifiers.

Domain-specific arrays may differ, but downstream components should consume them through explicit contracts rather than implicit dictionary conventions.

### C. Domain packs

`tsdiag.contracts.DomainPack`

A Domain Pack declares **what should be done**, not how the controller chooses it. It defines required metadata, supported tasks, ordered baseline tools, evidence fields, failure modes and benchmark targets.

The deterministic ordered pack is the scientific control condition for future LLM/RL routing.

### D. Tool registry

The registry maps stable tool names to implementations. A tool may be:

- signal processing;
- statistical monitoring;
- physics-derived feature extraction;
- causal/root-cause analysis;
- classical ML;
- deep learning;
- retrieval/knowledge;
- prognosis;
- verification.

Tools must emit structured outputs and evidence rather than free-form conclusions only.

### E. Planner / router

The router receives the domain pack, task, available metadata and current evidence state and returns an execution plan.

Controller implementations should be interchangeable:

1. `DeterministicRouter` — fixed scientific baseline;
2. `PolicyRouter` — learned supervised/heuristic policy;
3. `LLMRouter` — tool-selection policy with bounded actions;
4. `RLRouter` — optimized for diagnostic utility/cost under the same action space.

No controller may bypass contracts, validation, verification or provenance.

### F. Execution engine

The engine owns execution state, not domain science. It:

- validates preconditions;
- executes tools;
- records timings/status/errors;
- stores intermediate state;
- emits `ToolTraceStep` records;
- supports skip/retry/abstain semantics;
- prevents tools from silently consuming unavailable future information.

### G. Evidence layer

Every material claim should be backed by evidence with:

- stable `evidence_id`;
- source/tool;
- evidence kind;
- score;
- provenance;
- timestamp/cutpoint;
- supporting details.

Later, this can become an evidence graph for GraphRAG/causal reasoning, but the primitive unit remains the existing `Evidence` contract.

### H. Hypothesis layer

Detection, localization, diagnosis and prognosis must stay separated.

A high anomaly score is **not** automatically a root-cause diagnosis. Candidate hypotheses reference evidence IDs and can be supported, contradicted or left insufficient by verification tools.

### I. Verification and abstention

Before a high-impact conclusion is returned, verification may check:

- physical feasibility;
- temporal consistency;
- cross-sensor consistency;
- counter-evidence;
- OOD/data-quality warnings;
- causal plausibility;
- prognosis censoring consistency.

If requirements are not satisfied, the correct output is an explicit abstention or monitoring recommendation rather than manufactured confidence.

### J. Result contract

`DiagnosticResult` is the public cross-domain result schema. It separates:

- detection;
- localization;
- hypotheses;
- evidence;
- verification;
- prognosis;
- confidence/uncertainty;
- abstention;
- recommended actions;
- execution trace.

Legacy `DiagnosticReport` should remain only as a compatibility adapter while domains migrate.

### K. Benchmark layer

`tsdiag.benchmarks` must remain scientifically isolated from training/tuning.

Each benchmark defines a frozen protocol containing:

- exact dataset/cells/runs;
- cutpoints/windows;
- labels/endpoints;
- leakage rules;
- censoring policy;
- metrics;
- failure policy.

Benchmark artifacts must preserve protocol metadata, parser diagnostics, per-case predictions and aggregate metrics.

Real-data benchmark results should never cause silent threshold changes. Any later tuning requires a separate development/validation split or an independent dataset.

## 4. Proposed package structure

```text
src/tsdiag/
├── core/                    # domain-independent runtime
│   ├── context.py           # RunRequest / RunContext
│   ├── planning.py          # ExecutionPlan / routing interfaces
│   ├── execution.py         # execution engine
│   ├── evidence.py          # evidence utilities / store
│   └── registry.py          # tool/model registry facade
├── contracts.py             # DomainPack + ToolContract
├── models.py                # DiagnosticResult and public schemas
├── datasets/                # external-data adapters only
├── domains/                 # domain packs + domain-specific pipelines
├── tools/                   # reusable deterministic analysis tools
├── agents/                  # specialist wrappers and adaptive agents
├── verification/            # physics/consistency/OOD verifiers
├── benchmarks/              # frozen benchmark protocols
├── policies/                # deterministic/LLM/RL routing policies
└── knowledge/               # optional domain knowledge / retrieval
```

Not every directory needs an implementation immediately. The purpose is to enforce dependency direction.

## 5. Dependency rule

Preferred dependency flow:

```text
datasets ─┐
          ▼
        models/contracts
          ▲
          │
tools ─ domains ─ core runtime ─ policies
          │            │
          ▼            ▼
    verification   DiagnosticResult
                         ▲
                         │
                    benchmarks
```

Critical rules:

- benchmarks may call production code but production code must never import benchmark labels/results;
- dataset adapters must not tune algorithms;
- policies may choose tools but may not alter frozen benchmark definitions;
- verification is independent from the router that generated the hypothesis;
- all public domain outputs eventually converge to `DiagnosticResult`.

## 6. Runtime lifecycle

```text
1. Ingest + parse data
2. Validate observation and metadata
3. Resolve domain pack + task
4. Build execution plan
5. Execute contracted tools
6. Accumulate evidence
7. Generate/localize hypotheses or prognosis
8. Verify physical/statistical consistency
9. Calibrate confidence or abstain
10. Emit DiagnosticResult + full trace
11. Benchmark/report externally
```

## 7. Scientific development lifecycle

Architecture and model improvement are intentionally separate:

```text
ARCHITECTURE
    ↓
PARSER + DATA VALIDATION
    ↓
DETERMINISTIC BASELINE
    ↓
FROZEN REAL-DATA BENCHMARK
    ↓
ERROR ANALYSIS
    ↓
INDEPENDENT DEV/VALIDATION DATA
    ↓
METHOD IMPROVEMENT / TUNING
    ↓
LOCK METHOD
    ↓
FINAL HELD-OUT EVALUATION
```

The NASA four-cell result therefore remains a frozen baseline. Its weaknesses guide the design of the next method, but the four cells are not used as an unconstrained tuning set.

## 8. Near-term implementation phases

### Phase A — architecture foundation

- add runtime request/context contracts;
- add execution-plan contract;
- isolate routing policy from execution;
- standardize tool outcomes and errors;
- add evidence/provenance store;
- keep all current domain code working.

### Phase B — domain migration

Migrate existing process, bearing and battery pipelines onto the common runtime one at a time while preserving their benchmark results.

### Phase C — verification layer

Add explicit data-quality, OOD, physics and censoring consistency verifiers.

### Phase D — stronger models

Only after the architecture is stable:

- better battery degradation models;
- richer health indicators;
- multivariate prognostics;
- learned anomaly/root-cause models;
- calibrated uncertainty.

### Phase E — agentic routing

Evaluate LLM/RL routing against the deterministic router on fixed tasks using utility metrics that include accuracy, cost, latency, abstention and verification success.

## 9. Definition of architecture-complete v1

Architecture v1 is complete when:

- one runtime request can represent all current domains;
- every domain resolves through a `DomainPack`;
- routing is behind a policy interface;
- execution is handled by one domain-independent engine;
- tools emit structured evidence and traces;
- verification can veto/abstain independently;
- all new domain pipelines return `DiagnosticResult`;
- benchmark protocols are immutable inputs to evaluation and cannot be modified by routing/model code;
- current TEP, CWRU and NASA benchmarks remain reproducible.
