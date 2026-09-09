# TEP knowledge-guided root-cause diagnosis

This module upgrades the Tennessee Eastman Process root-cause stage from a generic contribution/Granger scorer to a knowledge-guided diagnostic layer.

## Motivation

Contribution plots can localize variables that became abnormal, but they can confuse root causes with downstream consequences. Granger-style causal screening adds directionality, but data-driven links are still predictive evidence rather than physical causality proof. Recent TEP root-cause literature therefore combines data evidence with process knowledge, topology, and known fault mechanisms.

## Added knowledge sources

`src/tsdiag/datasets/tep_knowledge.py` adds:

- 41 XMEAS variable descriptions
- 12 XMV manipulated-variable descriptions
- IDV(1)-IDV(21) fault catalog
- expected root-variable priors for known physical faults
- expected affected-variable neighbourhoods
- sparse process-topology prior

The expected-root and affected-variable lists are engineering priors. They are used to regularize ranking, not as hard truth for every trajectory.

## New reasoning flow

```text
DPCA/PCA alarm
      ↓
pre/post shift evidence
      ↓
PCA contribution evidence
      ↓
fault onset ordering
      ↓
Granger screening
      ↓
TEP topology filtering
      ↓
generic root-cause ranking
      ↓
TEP fault-catalog matching
      ↓
knowledge-guided root + fault label
```

## Leakage control

During benchmark prediction, the true fault ID is not passed to the diagnosis function. The diagnosis stage sees the full TEP catalog and must choose the best matching mechanism from observed evidence. The true IDV label is used only after prediction for scoring.

## Current limitation

The topology is intentionally sparse and qualitative. This is better than pure generic ranking, but not yet a full first-principles simulator graph. IDV(16)-IDV(20) remain benchmark-unknown faults and do not provide expected root priors.
