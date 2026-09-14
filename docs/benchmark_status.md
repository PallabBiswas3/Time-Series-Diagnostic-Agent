# Benchmark Status

This table distinguishes validated real-data evidence from implemented-but-not-yet-validated domain paths.

| Domain | Public method | Real-data status | Current evidence | Next scientific step |
| --- | --- | --- | --- | --- |
| Process / TEP | PCA/DPCA monitoring, contribution, temporal evidence, Granger/topology root-cause ranking | Validated regression benchmark | Fixed Braatz TEP workflow with root-cause regression gates | Continue improving verification without changing frozen gates post hoc |
| Bearing / CWRU | Spectral kurtosis, resonance band, Hilbert envelope, BPFO/BPFI/BSF/FTF family matching | Validated real-data baseline | 16 fixed CWRU records; 31.25% coverage, 68.75% abstention, 100% covered accuracy, 0% normal false alarms | Improve coverage using a predeclared methodology on independent development data |
| Wind SCADA / CARE v6 | Regime-aware residual diagnostics, persistence/change-point evidence, physics checks | Validated real-data workflow | CARE v6 benchmark and wind-specific tests are green | Strengthen subsystem localization and prospective validation |
| Battery | Cell-to-pack diagnostics plus causal capacity-trajectory prognosis | Real-data evaluated baseline; not research-ready | NASA B0005/B0006/B0007/B0018 at cutpoints 50/70/90/110: exact-event coverage 90%, MAE 14.89 cycles, RMSE 17.92 cycles; interval coverage 77.8%; B0007 right-censor interval compatibility 50% after multiscale uncertainty calibration | Do not tune further on the same four-cell frozen set; validate a materially different prognosis model on independent development/holdout data before stronger claims |
| Turbofan / C-MAPSS | Sensor trend screening plus fixed train-only HistGradientBoosting RUL adapter | Validated real-data baseline | Current NASA archive: 707 test engines across FD001-FD004; 100% coverage, MAE 24.08 cycles, RMSE 33.76 cycles; deterministic train-only holdout RMSE 24.77-34.56 cycles | Add calibrated RUL uncertainty/abstention and stronger regime-aware physics verification before research-ready claims |
| Transformer / SGAH | Wavelet/multisensor spectral representation with fixed RandomForest classifier; electrical protection-rule features retained as experimental verification support | Real-data evaluated baseline; not validated | Canonical frozen result: 13/64 transformer faults detected, 20.31% recall, 96.83% specificity, 59.09% precision, 30.23% F1, 58.57% balanced accuracy, 8.0% normal FPR, 0.54% competing-fault FPR; dev classifier accuracy 82.03% | Do not tune further on the repeatedly exposed SGAH test. Develop genuinely electrical features on train/dev or an independent transformer dataset before a new evaluation |

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

## SGAH transformer baseline

The transformer workflow is pinned to `smartlab-hfut/SGAH-datasets` commit `bbe1020e3fade83f7861657bb3eaea41c25ec0c9`. Each event is a whole 100-sample six-channel `Ua, Ub, Uc, Ia, Ib, Ic` waveform. The class-3 CSV has 99 incomplete trailing rows; they are discarded rather than padded or synthesized. The per-class protocol is contiguous 60% train / 20% development / 20% frozen test, giving 348 test events: 64 main-transformer-fault positives and 284 normal/competing-fault negatives.

The fixed benchmark gates remain recall >= 0.60, balanced accuracy >= 0.70, normal false-positive rate <= 0.15, and competing-fault false-positive rate <= 0.30.

The canonical retained transformer baseline uses the wavelet/multisensor spectral representation with a fixed `RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight=balanced_subsample, random_state=0)` and a fixed 0.5 decision threshold. Training uses only the training partition; frozen test labels are used only for scoring.

Canonical frozen result:

- frozen test events: 348
- transformer-fault cases: 64
- true positives / false negatives: 13 / 51
- transformer-fault recall: 0.2031
- specificity: 0.9683
- precision: 0.5909
- F1: 0.3023
- balanced accuracy: 0.5857
- normal false-positive rate: 0.080
- competing-grid-fault false-positive rate: 0.0054
- development classifier accuracy: 0.8203

This baseline is **real-data evaluated, but not validated/research-ready**. Specificity is already strong; the main weakness is insufficient sensitivity to the main-transformer-fault class.

Later exploratory experiments with development-threshold changes and deterministic electrical protection-inspired rules were informative but are **not** the canonical benchmark. They showed that threshold lowering could raise recall substantially only by causing unacceptable normal false positives, while hard sequence/asymmetry rejection could suppress transformer-fault recall entirely. These experiments are retained as evidence that threshold tuning or rigid rule vetoes are not the right solution for SGAH.

The next scientific step is therefore a genuinely electrical feature pipeline developed only on train/development data or, preferably, on an independent transformer dataset. Candidate features include positive/negative/zero-sequence components, phase/current imbalance, transient energy, voltage sag/current surge relationships, apparent/sequence impedance-style quantities, voltage-current relationships, and physically justified spectral features. The protection-rule code remains useful as an interpretable verification/support layer, but it must not replace the retained RF baseline or veto it on SGAH.

Because the SGAH frozen test partition has already been inspected across multiple method revisions, it should no longer be used for further method selection. A stronger transformer claim should come from a fresh holdout or an independent dataset, ideally with primary/secondary currents and transformer-specific protection measurements.

## Validation levels

- **Unit validated**: deterministic functions/contracts behave as intended.
- **Integration validated**: the public pipeline executes end to end with the standardized result contract.
- **Real-data validated**: a frozen real-data protocol runs successfully and produces retained artifacts.
- **Research-ready**: methodology, leakage controls, uncertainty/abstention, independent verification, and benchmark reporting are strong enough for comparative scientific claims.
