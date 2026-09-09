# Calibrated Tennessee Eastman benchmark

This benchmark extends the first TEP integration from a single PCA detector into a more research-useful comparison harness.

## What changed

1. **False-alarm reduction**
   - Control limits are tuned on held-out normal-operation data.
   - A persistence filter suppresses isolated single-sample alarms.

2. **PCA/control-limit tuning**
   - `alpha` and `min_consecutive` are selected from grids.
   - Tuning targets normal validation false-alarm rate only, so no fault labels are used during calibration.

3. **Improved root-cause ranking**
   - Ranking now combines PCA contribution, onset order, Granger outgoing influence, upstreamness, and robust pre/post shift evidence.
   - Granger edges are still treated only as predictive evidence, not as physical causal proof.

4. **DPCA baseline**
   - Dynamic PCA builds lagged features `[x_t, x_{t-1}, ..., x_{t-p}]`.
   - This is useful for autocorrelated process variables where static PCA may over-alarm.

5. **Tables and plots**
   - The benchmark writes JSON, CSV, Markdown summary, and PNG plots.

## Command

```bash
python -m tsdiag.benchmarks.tep \
  --download \
  --faults 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
  --methods pca dpca \
  --target-far 0.05 \
  --diagnostic-method dpca \
  --diagnostic-faults 4 5 6 14 15 21 \
  --output outputs/tep_benchmark.json \
  --artifact-dir outputs/tep_benchmark
```

## Output artifacts

- `outputs/tep_benchmark.json`
- `outputs/tep_benchmark/tep_fault_table.csv`
- `outputs/tep_benchmark/tep_summary.md`
- `outputs/tep_benchmark/plots/tep_detection_false_alarm_by_fault.png`
- `outputs/tep_benchmark/plots/tep_detection_delay_by_fault.png`

## Interpretation rule

The benchmark now reports detection and root-cause/localization separately.

- Detection metrics are measured across all selected TEP faults.
- Localization metrics are only proxy metrics for the conservative subset with engineering proxy variables.
- A strong detector with poor localization is not yet a good diagnostic agent.
