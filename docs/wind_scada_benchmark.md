# Wind-SCADA Real-Data Benchmark (PR #11)

## Dataset

PR #11 uses **CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data**, Zenodo version 6:

- DOI: `10.5281/zenodo.15846963`
- 95 event datasets from 36 wind turbines across three wind farms
- 10-minute SCADA time series
- 45 anomaly events and 50 normal events in v6
- wind-farm-specific feature spaces
- each event contains a training portion and a prediction portion

The archive is not vendored. The benchmark can read either an extracted CARE directory or the official ZIP directly, avoiding the large extracted footprint.

## Label and data-quality policy

`status_type_id` is used to filter healthy training rows for the normal-behavior model. `event_label` is used only after inference for event-level evaluation. For Wind Farm A, prediction-time status is not used as an evaluation mask; for B/C, already-abnormal operator states are excluded from CARE-style pointwise evaluation.

The first benchmark uses average channels by default. This avoids relying on problematic Min/Max/Std fields documented in CARE data-quality guidance.

## Diagnostic pipeline

```text
SCADA event
    ↓
data-quality / timestamp checks
    ↓
healthy training filter
    ↓
operating-regime discovery
    ↓
regime-aware normal-behavior model
    ↓
observed − expected residuals
    ├───────────────┐
    ↓               ↓
persistent       sequential
residual alarm   CUSUM drift
    └──────┬────────┘
           ↓
    fused event stream
           ↓
physics / subsystem verification
           ↓
event decision + evidence trace
```

### Regime-aware normal behavior

Healthy samples are clustered into comparable operating regimes. Prediction samples are assigned to the nearest healthy-regime center rather than being re-clustered with potentially faulty data. Gradient-boosted normal-behavior models are trained per regime with a global fallback. Target and predictor counts and per-regime training samples are bounded so the complete CARE benchmark remains tractable.

### Residual change-point detection

`src/tsdiag/detectors/residual_changepoint.py` implements a two-sided sequential CUSUM on normalized NBM residuals. It reports:

- aggregate change-point sample indices
- per-channel change points
- corresponding timestamps when available
- CUSUM strength traces
- a short post-change hold mask fused into the main event alarm stream

This detector is intended to capture low-amplitude persistent degradation that may not cross the instantaneous residual threshold.

### Physics and subsystem consistency

`src/tsdiag/domain/wind_physics.py` provides conservative verification checks rather than a second classifier.

**Aerodynamic/electrical consistency** checks active power against wind speed, pitch and generator/rotor state. When rotor area is supplied, the physical upper bound uses

```text
P_available = 0.5 * rho * A * Cp_max * v^3
```

with `Cp_max <= 0.59`. If exact turbine geometry is unavailable because of anonymization, that bound is disabled rather than fabricated. Reference-derived rated-power behavior can still support a conservative possible-curtailment/electrical-loss finding.

**Thermal-mechanical consistency** fits healthy steady-state temperature relationships against available ambient temperature, torque, rotor speed and power. Persistent excess temperature beyond a robust healthy dissipation curve can flag bearing, gearbox or generator thermal inconsistency. Findings are grouped into subsystem evidence such as `bearing`, `gearbox`, `generator`, and `electrical`.

## Evaluation

The full benchmark is run with:

```bash
python scripts/run_wind_scada_benchmark.py \
  --data-dir CARE_To_Compare.zip \
  --output-dir outputs/wind_benchmark
```

It produces:

- `outputs/wind_benchmark/wind_scada_benchmark.json`
- `outputs/wind_benchmark/wind_scada_event_table.csv`
- `outputs/wind_benchmark/care_v6_summary.md`

The summary reports:

- Event Recall
- Event Precision
- Event F1
- normal-event false-alarm rate
- anomaly-window recall
- mean and median early-warning lead time
- CARE-style operational criticality statistics
- tool invocations per event
- per-event and total runtime

The project reports **CARE-style operational criticality**, not the complete official composite CARE scalar. This distinction is kept explicit.

## CARE-style criticality

Criticality is a bounded sequential evidence counter:

- +1 when an anomaly is detected while the turbine is in an operator-normal state
- −1 for a non-alarm point while the turbine is in an operator-normal state
- unchanged when the turbine is already in an operator-abnormal state

The default event threshold is 72. The first threshold crossing is also used to calculate early-warning lead time relative to the event endpoint.

## Scientific guardrails

1. Prediction-event labels are never used to fit the normal-behavior model.
2. There is no random time-window split across a single event.
3. Normal-behavior fitting uses only the provided healthy training region.
4. Prediction regimes are assigned from healthy reference structure.
5. Missing turbine specifications disable a physics bound rather than introducing guessed constants.
6. TEP-specific root-cause logic remains frozen after PR #10.
7. Runtime and tool-call counts are retained for the later fixed-vs-adaptive routing study in PR #12.

## Measured CARE v6 results

The canonical full-dataset GitHub Actions run populates `care_v6_summary.md`. Final measured values are copied here only after that real-data run completes successfully; synthetic unit-test values are never reported as benchmark results.
