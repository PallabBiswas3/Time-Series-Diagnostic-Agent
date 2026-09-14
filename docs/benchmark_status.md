# Benchmark Status

This table distinguishes validated real-data evidence from implemented-but-not-yet-validated domain paths.

| Domain | Public method | Real-data status | Current evidence | Next scientific step |
| --- | --- | --- | --- | --- |
| Process / TEP | PCA/DPCA monitoring, contribution, temporal evidence, Granger/topology root-cause ranking | Validated regression benchmark | Fixed Braatz TEP workflow with root-cause regression gates | Continue improving verification without changing frozen gates post hoc |
| Bearing / CWRU | Spectral kurtosis, resonance band, Hilbert envelope, BPFO/BPFI/BSF/FTF family matching | Validated real-data baseline | 16 fixed CWRU records; 31.25% coverage, 68.75% abstention, 100% covered accuracy, 0% normal false alarms | Improve coverage using a predeclared methodology on independent development data |
| Wind SCADA / CARE v6 | Regime-aware residual diagnostics, persistence/change-point evidence, physics checks | Validated real-data workflow | CARE v6 benchmark and wind-specific tests are green | Strengthen subsystem localization and prospective validation |
| Battery | Cell-to-pack diagnostics plus causal capacity-trajectory prognosis | Real-data evaluated baseline; not research-ready | NASA B0005/B0006/B0007/B0018 at cutpoints 50/70/90/110: exact-event coverage 90%, MAE 14.89 cycles, RMSE 17.92 cycles; interval coverage 77.8%; B0007 right-censor interval compatibility 50% after multiscale uncertainty calibration | Do not tune further on the same four-cell frozen set; validate a materially different prognosis model on independent development/holdout data before stronger claims |
| Turbofan / C-MAPSS | Sensor trend screening plus fixed train-only HistGradientBoosting RUL adapter | Validated real-data baseline | Current NASA archive: 707 test engines across FD001-FD004; 100% coverage, MAE 24.08 cycles, RMSE 33.76 cycles; deterministic train-only holdout RMSE 24.77-34.56 cycles | Add calibrated RUL uncertainty/abstention and stronger regime-aware physics verification before research-ready claims |
| Transformer / SGAH | Wavelet denoising, correlation weighting, multisensor fusion, spectral representation, optional train-only classifier | Real-data evaluated baseline; not validated | Frozen 348-event SGAH test: first linear baseline recall 7.81%, balanced accuracy 51.09%; one predeclared nonlinear revision improved recall to 20.31%, balanced accuracy 58.57%, normal FPR 8.0%, competing-fault FPR 0.54%, but still failed fixed gates | Do not tune again on the frozen SGAH test. Develop a materially different electrical-feature method using train/dev data or an independent transformer dataset before reevaluation |

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

## SGAH transformer electrical-fault baseline

The transformer path now has a dedicated real-data workflow pinned to `smartlab-hfut/SGAH-datasets` commit `bbe1020e3fade83f7861657bb3eaea41c25ec0c9`. Each event is kept as a whole 100-sample, six-channel waveform; no row-level split is allowed. The pinned class-3 CSV contains 99 trailing rows that do not form a complete event, so they are explicitly discarded rather than padded or synthesized.

The frozen per-class split is contiguous 60% train / 20% development / 20% test. The final frozen test contains 348 events: 64 main-transformer-fault positives and 284 negatives drawn from normal operation plus three competing grid-fault classes. The predeclared gates are transformer-fault recall >= 0.60, balanced accuracy >= 0.70, normal false-positive rate <= 0.15, and competing-fault false-positive rate <= 0.30.

The first train-only linear spectral classifier failed the frozen gate:

- raw development classifier accuracy: 0.617
- transformer-fault recall: 0.078
- specificity: 0.944
- precision: 0.238
- balanced accuracy: 0.511
- normal false-positive rate: 0.090
- competing-fault false-positive rate: 0.038

One methodology revision was then predeclared without changing the frozen gates: a fixed train-only RandomForest classifier on the same spectral representation, with the public pipeline run in classifier-led electrical mode so the legacy impulsiveness heuristic is retained as evidence rather than allowed to veto an electrical fault label. That revision improved discrimination but still failed the frozen recall and balanced-accuracy gates:

- raw development classifier accuracy: 0.820
- transformer-fault recall: 0.203
- specificity: 0.968
- precision: 0.591
- F1: 0.302
- balanced accuracy: 0.586
- normal false-positive rate: 0.080
- competing-fault false-positive rate: 0.0054
- public-result coverage: 0.063
- abstention rate: 0.937

The fixed false-positive gates are comfortably satisfied, but sensitivity is inadequate. The SGAH set has therefore already served as frozen evaluation data and must not be used for another round of threshold or hyperparameter tuning. Transformer diagnosis is **real-data evaluated, but not validated/research-ready**. The next method should be designed using training/development data only, preferably with electrical phase-sequence, symmetrical-component, voltage/current imbalance and transient features, or evaluated on an independent transformer dataset before returning to the frozen SGAH test.

## Validation levels

- **Unit validated**: deterministic functions/contracts behave as intended.
- **Integration validated**: the public pipeline executes end to end with the standardized result contract.
- **Real-data validated**: a frozen real-data protocol runs successfully and produces retained artifacts.
- **Research-ready**: methodology, leakage controls, uncertainty/abstention, independent verification, and benchmark reporting are strong enough for comparative scientific claims.
