# Benchmark Status

This table distinguishes validated real-data evidence from implemented-but-not-yet-validated domain paths.

| Domain | Public method | Real-data status | Current evidence | Next scientific step |
| --- | --- | --- | --- | --- |
| Process / TEP | PCA/DPCA monitoring, contribution, temporal evidence, Granger/topology root-cause ranking | Validated regression benchmark | Fixed Braatz TEP workflow with root-cause regression gates | Continue improving verification without changing frozen gates post hoc |
| Bearing / CWRU | Spectral kurtosis, resonance band, Hilbert envelope, BPFO/BPFI/BSF/FTF family matching | Validated real-data baseline | 16 fixed CWRU records; 31.25% coverage, 68.75% abstention, 100% covered accuracy, 0% normal false alarms | Improve coverage using a predeclared methodology on independent development data |
| Wind SCADA / CARE v6 | Regime-aware residual diagnostics, persistence/change-point evidence, physics checks | Validated real-data workflow | CARE v6 benchmark and wind-specific tests are green | Strengthen subsystem localization and prospective validation |
| Battery | Cell-to-pack voltage/temperature deviation plus preliminary prognosis | Pipeline implemented; NASA prognosis baseline intentionally not merged | Existing NASA baseline has censoring-safety weaknesses | Redesign censor-aware prognosis/evaluation before merge |
| Turbofan | Sensor trend screening, health index, linear/trained RUL adapter | Executable baseline only | Synthetic/public pipeline tests | Add frozen real-data turbofan benchmark and RUL metrics |
| Transformer | Wavelet denoising, correlation weighting, multisensor fusion, optional classifier | Executable baseline only | Synthetic/public pipeline tests | Add frozen real-data transformer fault benchmark and classifier validation |

## CWRU locked post-refactor baseline

The current architecture-only bearing refactor preserves the following fixed CWRU result:

- records: 16
- all-record accuracy: 0.3125
- macro F1 including abstention: 0.35
- coverage: 0.3125
- abstention rate: 0.6875
- covered accuracy: 1.0
- normal false-alarm rate: 0.0
- fault detection rate: 0.4167

These numbers should be treated as a baseline, not a final performance claim. The low coverage is a known scientific weakness and must not be improved by tuning directly against the frozen evaluation set.

## Validation levels

- **Unit validated**: deterministic functions/contracts behave as intended.
- **Integration validated**: the public pipeline executes end to end with the standardized result contract.
- **Real-data validated**: a frozen real-data protocol runs successfully and produces retained artifacts.
- **Research-ready**: methodology, leakage controls, uncertainty/abstention, independent verification, and benchmark reporting are strong enough for comparative scientific claims.
