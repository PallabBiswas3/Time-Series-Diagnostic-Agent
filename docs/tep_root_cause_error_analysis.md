# TEP Root-Cause Error Analysis and Calibrated Ranker

## Purpose

PR #10 replaces manual root-score weighting with a reproducible calibration procedure. The benchmark constructs six leakage-free candidate-root features, applies a fixed label-blind candidate screen, then optimizes scoring weights using fault-level cross-validation. True IDV labels are used only as calibration/evaluation targets; they are never used to construct a held-out fault's candidate features.

## Error taxonomy

Every held-out top-1 mistake is assigned to one primary category:

- **DOWNSTREAM_OVER_WEIGHTING** — propagated symptoms receive stronger evidence than the upstream cause.
- **EARLY_NOISE_ONSET** — a downstream/noisy channel receives a much earlier onset score than the true root.
- **CATALOG_AMBIGUITY** — multiple catalog mechanisms have nearly indistinguishable catalog/type support.
- **WEAK_SIGNAL** — the expected root has weak observed evidence and is difficult to localize reliably.

The benchmark writes both `tep_error_analysis_report.json` and `tep_error_analysis_report.md` with true-root rank, top wrong variable, component scores, support gap, and whether the true root survived candidate screening.

## Candidate screening

Candidate completeness and prediction are intentionally separated. The post-mortem record retains all global TEP catalog roots so an omitted true root can still be audited, but the calibrated ranker receives at most 24 variables selected without using the fault label. Screening is driven by observed pre/post shift, contribution, and onset evidence.

```text
all measured variables + global catalog roots for audit
                    |
       label-blind evidence screen
                    |
          <= 24 ranker candidates
                    |
       calibrated six-term ranker
```

In the final PR #10 benchmark, **candidate omission count was 0/16 known-root cases**. Thus the final localization errors are ranking/evidence errors rather than failures to make the true root available to the ranker.

## Calibrated scoring model

For candidate variable `v`:

```text
score(v) =
    w1 * contribution_score
  + w2 * pre_post_shift_score
  + w3 * onset_earliness_score
  + w4 * topology_upstreamness_score
  + w5 * fault_type_agreement_score
  + w6 * catalog_root_prior_score
```

All six inputs are normalized before scoring. The topology feature is derived from the sparse TEP process graph. Fault-type agreement is a blind temporal signature computed from observed data. The catalog prior is computed against the complete catalog rather than the true fault record.

Final deployment weights selected after the screened fault-level calibration are:

| Feature | Weight |
| --- | ---: |
| Contribution | 0.00 |
| Pre/post shift | 0.00 |
| Onset earliness | 0.10 |
| Topology upstreamness | 0.60 |
| Fault-type agreement | 0.10 |
| Catalog root prior | 0.20 |

These weights are frozen after PR #10. They should not be tuned further on the same TEP benchmark faults.

## Cross-validation protocol

The 16 known-root TEP faults are partitioned by **fault ID**, not by time windows. Four-fold fault-level cross-validation is used so a held-out fault mechanism is not present in that fold's calibration set.

For each fold:

1. Construct leakage-free evidence features.
2. Apply the same label-blind 24-candidate screen.
3. Search simplex weight combinations and an abstention margin on training fault IDs.
4. Select a configuration subject to bounded abstention and false-confident diagnosis rates.
5. Evaluate exactly once on held-out fault IDs.

Only after the out-of-fold result is computed is a final deployment configuration fitted on all known-root faults. The reported generalization result is therefore the held-out aggregate, not the in-sample final fit.

## Final benchmark comparison

Canonical Braatz TEP known-root cases:

| Root-cause method | Top-1 root accuracy | Raw Top-1 | Top-3 root accuracy | Abstention rate | False-confident rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR #9 topology + catalog | 0.3750 | — | 0.5625 | 0.2857 | 0.6250 |
| PR #10 screened calibrated CV | **0.4375** | **0.5000** | **0.8125** | **0.0625** | **0.5000** |

PR #10 therefore improves accepted Top-1 localization by **6.25 percentage points** and Top-3 localization by **25.0 percentage points** over PR #9, while lowering abstention and reducing the false-confident rate. The distinction between raw Top-1 (0.5000) and accepted Top-1 (0.4375) comes from one held-out case being abstained by the calibrated safety rule.

## Post-mortem results

The final held-out error analysis reports 8 raw top-1 ranking errors and no candidate omissions:

| Failure category | Count |
| --- | ---: |
| WEAK_SIGNAL | 7 |
| DOWNSTREAM_OVER_WEIGHTING | 1 |
| EARLY_NOISE_ONSET | 0 |
| CATALOG_AMBIGUITY | 0 |

The dominant remaining limitation is therefore weak root evidence rather than missing candidates or catalog ambiguity. This is a useful stopping point scientifically: further TEP-specific tuning would risk benchmark overfitting, so these errors are retained as documented limitations for cross-domain work.

## Regression gates

The GitHub Actions TEP benchmark fails if a future change violates:

```text
Top-1 root accuracy       > 0.375
Top-3 root accuracy       > 0.5625
Abstention rate           <= 0.35
False-confident rate      <= 0.625
```

The final PR #10 run passed all four gates with `0.4375 / 0.8125 / 0.0625 / 0.5000`, respectively.

## Generated artifacts

A full `--run-ablation` benchmark produces:

- `outputs/tep_benchmark/tep_root_cause_ablation.csv`
- `outputs/tep_benchmark/tep_calibrated_weights.json`
- `outputs/tep_benchmark/tep_error_analysis_report.json`
- `outputs/tep_benchmark/tep_error_analysis_report.md`
- `outputs/tep_benchmark/tep_summary.md`
- root-cause ablation and confusion-matrix plots
