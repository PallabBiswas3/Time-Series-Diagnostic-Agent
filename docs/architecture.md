# Platform Architecture v1

This architecture is frozen before benchmark-driven model tuning. Benchmark thresholds, cutpoints, labels and model hyperparameters are outside the architecture layer.

## Runtime flow

```text
raw industrial data
      ↓
dataset/domain adapter
      ↓
canonical RunRequest / RunContext
      ↓
DomainPack contract
      ↓
RouterPolicy
      ↓
ExecutionPlan
      ↓
ExecutionEngine + ToolRegistry
      ↓
structured outputs + EvidenceStore
      ↓
hypotheses / prognosis
      ↓
independent VerificationSuite
      ↓
DiagnosticResult
      ↓
benchmark / API / UI
```

## Core rules

1. Dataset adapters parse data; they do not tune algorithms.
2. Domain packs declare required metadata, supported tasks, ordered tools, outputs, evidence and failure modes.
3. Routing and execution are separate. The deterministic ordered domain pack is the scientific control router for later ML/LLM/RL policies.
4. The execution engine is domain-independent and records tool status, outputs, evidence IDs, timing and failures.
5. Evidence is structured and provenance-aware. Diagnostic hypotheses should reference evidence instead of relying on free-form prose.
6. Verification is independent of the router/model and may contradict a hypothesis or force abstention.
7. Detection, localization, diagnosis and prognosis remain separate concepts.
8. Benchmarks call production code, but production code must never import benchmark labels/results or silently modify frozen protocol definitions.
9. Real-data results must not directly trigger threshold tuning. Method development should use independent development/validation data before reevaluating held-out benchmarks.

## Package responsibilities

```text
src/tsdiag/
├── core/          # RunRequest, RunContext, routing, plans, execution, evidence
├── contracts.py   # DomainPack and ToolContract
├── models.py      # DiagnosticResult and public schemas
├── datasets/      # external-data adapters
├── domains/       # domain contracts and domain-specific runners
├── tools/         # reusable deterministic/model tools
├── agents/        # specialist/adaptive agents
├── verification/  # independent physics/consistency/OOD checks
├── benchmarks/    # frozen evaluation protocols
├── policies/      # future deterministic/ML/LLM/RL routing policies
└── knowledge/     # optional retrieval/domain knowledge
```

## Implemented in Architecture v1 foundation

- `RunRequest` and `RunContext`
- `ExecutionPlan`, `PlanStep`, and `RouterPolicy`
- deterministic domain-pack router
- domain-independent `ExecutionEngine`
- normalized `ToolOutcome`
- run-local `EvidenceStore` with stable evidence IDs and tool provenance
- independent `Verifier` / `VerificationSuite` interface
- `DiagnosticRuntime` facade
- backward-compatible legacy `build_execution_plan`
- focused architecture CI contract tests

## Next migration sequence

1. Wrap existing process/TEP runner behind the shared runtime.
2. Wrap existing bearing/CWRU runner behind the shared runtime.
3. Wrap battery/NASA runner after the frozen benchmark branch is integrated or explicitly stacked.
4. Add data-quality/OOD/physics verifiers.
5. Standardize all new domain outputs on `DiagnosticResult`.
6. Only then improve domain models and later introduce learned/LLM/RL routing.

The NASA four-cell result remains a frozen baseline and is not used for unconstrained tuning during this architecture work.
