# Wind-SCADA Real-Data Benchmark (PR #11)

## Dataset

PR #11 uses **CARE to Compare: Wind Turbine SCADA Data for Early Fault Detection**, Zenodo version 6:

- DOI: `10.5281/zenodo.15846963`
- published dataset version: v6
- 95 event datasets from 36 wind turbines across three wind farms
- 10-minute SCADA time series
- 45 anomaly/fault events and 50 normal events in v6
- wind-farm-specific feature counts (A: 86, B: 257, C: 957)
- each event includes a normal training portion and a prediction portion

Version 6 is used because earlier releases contained label and metadata issues that were subsequently corrected.

The official archive is large (about 5.5 GB compressed), so it is not vendored into this repository and is not downloaded by unit tests. Unit tests use small synthetic fixtures; real benchmark execution uses a separately cached/extracted dataset.

## Data-quality policy

The v6 dataset documentation warns that many per-timestamp Min/Max/Std summary statistics contain implausible values, with particularly severe contamination for portions of Wind Farms B and C. Therefore the first benchmark uses **average (`Avg`) channels only** unless a feature is explicitly validated.

Training data are filtered using operator/status labels so the normal-behavior model is fitted only on samples considered normal. Prediction labels are never used to fit the model.

## Baseline pipeline

```text
SCADA event
    ↓
data-quality checks
    ↓
healthy training filter
    ↓
operating-regime discovery
    ↓
regime-aware normal-behavior model
    ↓
observed − expected residuals
    ↓
robust residual normalization
    ↓
persistent anomaly detection
    ↓
rolling/change-point evidence (next PR #11 stage)
    ↓
physics consistency + subsystem risk (next PR #11 stage)
```

The initial normal-behavior model uses gradient-boosted regressors and is conditioned on healthy operating regimes. Prediction samples are assigned to the nearest healthy regime rather than re-clustering possibly faulty data.

## Why regimes matter

Wind-turbine telemetry is strongly dependent on operating conditions. Startup, partial-load, rated-power and other conditions have different normal relationships among wind speed, power, rotational speed and temperatures. A single unconditional residual model would therefore create avoidable false alarms during normal operating changes.

## Benchmark metrics

PR #11 will report at least:

- event recall / event-level detection rate
- false-alarm rate on normal prediction events
- warning lead time / earliness before labeled fault onset when timestamps are available
- per-event alarm coverage
- subsystem/localization accuracy when a defensible subsystem label exists
- calibration of risk/confidence
- tool-call count and runtime, recorded now so PR #12 can compare fixed vs adaptive routing

The CARE score defined with the dataset will be added once the event-level evaluation adapter is complete.

## Scientific guardrails

1. No prediction-event labels are used for model fitting.
2. No random time-window splitting across the same event.
3. Normal-behavior models are trained only on the provided healthy training region.
4. Hyperparameters are selected on training/validation information, not individual held-out fault events.
5. TEP-specific root-cause logic remains frozen; Wind-SCADA receives its own domain adapter instead of modifying TEP heuristics.
