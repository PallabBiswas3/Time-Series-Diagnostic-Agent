# Time-Series Diagnostic Agent

Adaptive, evidence-backed industrial time-series diagnostics.

This project is designed as a **specialist-agent platform**, not a single bearing classifier. The long-term goal is to combine deterministic signal analysis, statistical monitoring, causal/root-cause reasoning, learned models, multimodal evidence, physics/manual verification, prognostics, and later sovereign industrial orchestration.

## Current specialist agents

### 1. SignalProcessingAgent
Generic first-pass signal diagnostics:
- RMS / peak / crest factor
- Pearson kurtosis
- Welch PSD
- STFT-based temporal spectral variability
- generic anomaly/impulsiveness score

### 2. BearingDiagnosticAgent
Rotating-bearing diagnostics using:
- band-pass resonance isolation
- Hilbert envelope
- envelope spectrum
- BPFO / BPFI / BSF / FTF harmonic matching
- diagnosis confidence and explicit evidence

### 3. StatisticalMonitoringAgent
Reference-based process monitoring:
- PCA
- reconstruction/SPE monitoring
- Hotelling-style T2 monitoring
- Local Outlier Factor
- 99% reference limits

### 4. ProbabilisticDiagnosticAgent
Uncertainty-aware reference comparison:
- multivariate Gaussian baseline
- Mahalanobis-distance scoring
- reference-derived anomaly threshold

### 5. CausalRootCauseAgent
For multivariate industrial processes:
- pairwise Granger-causality screening
- directed dependency graph
- candidate upstream/root-cause channel ranking

> Granger relationships are treated as predictive evidence, not proof of physical causality.

### 6. TransferRobustnessAgent
Checks whether deployment data differs strongly from the source/training domain:
- normalized mean shift
- scale shift
- operating-domain shift score

This is intended to flag when domain adaptation or recalibration may be needed.

### 7. LearnedModelAgent
Adapter for trained models supplied by the application:
- CNN
- LSTM / GRU / BiLSTM
- Transformer
- autoencoder
- ensemble classifier
- custom anomaly/fault model

Heavy learned models are **not falsely bundled as pretrained models**. A trained predictor must be registered explicitly.

### 8. MultimodalFusionAgent
Fuses evidence from combinations of:
- raw/time-series analysis
- learned models
- text/manual/KG evidence
- images/time-frequency artifacts
- external diagnostic tools

### 9. EvidenceVerificationAgent
Verification layer for candidate diagnostic claims:
- physics checks
- domain constraints
- maintenance/manual evidence
- custom verifier hooks
- SUPPORTED / CONTRADICTED / INSUFFICIENT-style aggregation

### 10. PrognosticsAgent
Initial prognostics baseline:
- health-index trend analysis
- threshold-crossing estimate
- simple RUL estimate

This is intentionally a baseline; validated industrial RUL models should replace it for real maintenance use.

## Orchestration

`IndustrialDiagnosticOrchestrator` routes a record to the specialist agents that are applicable to the available data/context.

```text
SignalRecord
   |
   +--> SignalProcessingAgent
   |
   +--> BearingDiagnosticAgent            if BPFO/BPFI/BSF/FTF available
   |
   +--> StatisticalMonitoringAgent        if normal-reference data available
   +--> ProbabilisticDiagnosticAgent
   +--> TransferRobustnessAgent
   |
   +--> CausalRootCauseAgent              if multichannel data available
   |
   +--> LearnedModelAgent                 if a trained predictor is registered
   |
   +--> MultimodalFusionAgent             if external/modal evidence is supplied
   |
   +--> EvidenceVerificationAgent         if candidate claims/verifiers are supplied
   |
   +--> PrognosticsAgent                  if health-history data is supplied
   |
   v
DiagnosticReport
```

The current router is deliberately deterministic and inspectable. Later we can benchmark an LLM/learned/RL router against this baseline instead of assuming agentic routing is automatically better.

## Install

```bash
pip install -e ".[dev]"
pytest
```

## Example

```python
import numpy as np
from tsdiag import SignalRecord, IndustrialDiagnosticOrchestrator

fs = 12000.0
x = np.random.randn(24000)

record = SignalRecord(
    signal=x,
    fs=fs,
    fault_frequencies={
        "BPFO": 85.0,
        "BPFI": 120.0,
        "BSF": 50.0,
        "FTF": 11.0,
    },
)

report = IndustrialDiagnosticOrchestrator().run(record)
print(report.decision, report.label, report.confidence)
```

## Research-method coverage

The repository now has concrete homes/interfaces for the major method families we identified from the literature:

- signal processing
- statistical process monitoring
- probabilistic diagnosis
- deep/learned time-series models
- graph/causal root-cause analysis
- transfer/domain-shift robustness
- multimodal diagnosis
- agentic/tool-based diagnosis
- evidence and physics verification
- prognostics / predictive maintenance

Not every research method is implemented as a full production model yet. For example, domain-specific CNN/Transformer models, digital twins, full Bayesian networks, advanced wavelet/framelet models, and validated RUL networks require separate datasets/model artifacts and will be added as specialized implementations rather than placeholder claims.

## Planned next phases

1. **Real bearing benchmark** — MFPT/CWRU-style validation and better resonance-band selection.
2. **Generic anomaly toolkit** — change-point, autocorrelation, wavelet, spectral-kurtosis/kurtogram, multivariate anomaly taxonomy.
3. **Learned-model packs** — autoencoder, Transformer and domain-specific fault classifiers.
4. **Industrial process pack** — causal/root-cause diagnostics for multivariate process systems.
5. **Evidence integration** — connect the Adaptive Evidence-Grounded Graph Agent for manuals/specifications/history.
6. **Sovereign integration** — expose this specialist suite to the Sovereign Industrial Operations Agent.
