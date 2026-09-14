# Benchmark Status

This table distinguishes validated real-data evidence from implemented-but-not-yet-validated domain paths.

| Domain | Public method | Real-data status | Current evidence | Next scientific step |
| --- | --- | --- | --- | --- |
| Process / TEP | PCA/DPCA monitoring, contribution, temporal evidence, Granger/topology root-cause ranking | Validated regression benchmark | Fixed Braatz TEP workflow with root-cause regression gates | Continue improving verification without changing frozen gates post hoc |
| Bearing / CWRU | Spectral kurtosis, resonance band, Hilbert envelope, BPFO/BPFI/BSF/FTF family matching | Validated real-data baseline | 16 fixed CWRU records; 31.25% coverage, 68.75% abstention, 100% covered accuracy, 0% normal false alarms | Improve coverage using a predeclared methodology on independent development data |
| Wind SCADA / CARE v6 | Regime-aware residual diagnostics, persistence/change-point evidence, physics checks | Validated real-data workflow | CARE v6 benchmark and wind-specific tests are green | Strengthen subsystem localization and prospective validation |
| Battery | Cell-to-pack diagnostics plus causal capacity-trajectory prognosis | Real-data evaluated baseline; not research-ready | NASA B0005/B0006/B0007/B0018 at cutpoints 50/70/90/110: exact-event coverage 90%, MAE 14.89 cycles, RMSE 17.92 cycles; interval coverage 77.8%; B0007 right-censor interval compatibility 50% after multiscale uncertainty calibration | Do not tune further on the same four-cell frozen set; validate a materially different prognosis model on independent development/holdout data before stronger claims |
| Turbofan / C-MAPSS | Sensor trend screening plus fixed train-only HistGradientBoosting RUL adapter | Validated real-data baseline | Current NASA archive: 707 test engines across FD001-FD004; 100% coverage, MAE 24.08 cycles, RMSE 33.76 cycles; deterministic train-only holdout RMSE 24.77-34.56 cycles | Add calibrated RUL uncertainty/abstention and stronger regime-aware physics verification before research-ready claims |
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

## NASA battery censor-aware baseline

The battery prognosis path now has a dedicated real-data workflow over B0005, B0006, B0007 and B0018. Predictions use only history available at each cutpoint. Exact-EOL cells are scored with point RUL error; cells whose records end above the 1.4 Ah EOL threshold are treated as right-censored and never assigned a fabricated point target.

Locked post-calibration results:

- evaluated cases: 14
- exact-EOL cases: 10
- exact-EOL prediction coverage: 0.90
- exact-EOL MAE: 14.89 cycles
- exact-EOL RMSE: 17.92 cycles
- exact-event interval coverage: 0.778
- right-censored prediction coverage: 1.00
- B0007 one-sided interval compatibility: 0.50
- mean B0007 one-sided violation: 6.69 cycles
- maximum B0007 one-sided violation: 18.83 cycles

The uncertainty method was changed once, from a narrow residual/slope approximation to regression-parameter uncertainty plus fixed-window (20, 40, full-history) model disagreement. Because the same frozen four-cell set exposed the remaining weakness, it should now be treated as evaluation data rather than tuned repeatedly. Battery prognosis is therefore **real-data evaluated, but not research-ready**.

## NASA C-MAPSS turbofan baseline

The turbofan path now has a dedicated NASA C-MAPSS workflow over FD001-FD004. The current official archive contains 707 test engines in total. The downloadable FD004 files contain 249 training engines, 248 test engines and 248 RUL targets; this differs from the current Open Data metadata page, so the workflow locks to the internally consistent downloadable archive bytes.

The original generic linear health-index extrapolator was evaluated first and failed the frozen RMSE gate badly (RMSE 1858.56 cycles). It was not rescued by loosening the gate. A materially different train-only model was then predeclared: fixed current/recent sensor-state and trend features with fixed HistGradientBoosting hyperparameters. Training targets are derived only from run-to-failure training trajectories; published test RUL labels are loaded only after fitting.

Locked train-only model results:

- test engines: 707
- coverage: 1.00
- MAE: 24.08 cycles
- RMSE: 33.76 cycles
- median absolute error: 17.46 cycles
- mean signed error: +10.87 cycles
- FD001 RMSE: 26.86 cycles
- FD002 RMSE: 28.82 cycles
- FD003 RMSE: 36.47 cycles
- FD004 RMSE: 39.43 cycles

Deterministic development holdout results, using only training trajectories and predicting at 70% of each held-out engine lifetime:

- FD001 holdout RMSE: 24.77 cycles
- FD002 holdout RMSE: 29.55 cycles
- FD003 holdout RMSE: 34.56 cycles
- FD004 holdout RMSE: 34.49 cycles

The frozen CI gate is coverage >= 0.80 and RMSE <= 75 cycles. The current baseline passes comfortably. It is therefore **real-data validated**, while uncertainty calibration, abstention and stronger operating-regime verification remain before research-ready claims.

## Validation levels

- **Unit validated**: deterministic functions/contracts behave as intended.
- **Integration validated**: the public pipeline executes end to end with the standardized result contract.
- **Real-data validated**: a frozen real-data protocol runs successfully and produces retained artifacts.
- **Research-ready**: methodology, leakage controls, uncertainty/abstention, independent verification, and benchmark reporting are strong enough for comparative scientific claims.
