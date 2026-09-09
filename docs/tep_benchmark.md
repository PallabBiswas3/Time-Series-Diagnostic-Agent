# Tennessee Eastman Process benchmark

This project uses the public Braatz-group Tennessee Eastman Process archive as the first real-data benchmark for the Chemical / Process domain pack.

## Upstream data

Source repository:

- `https://github.com/camaramm/tennessee-eastman-profBraatz`

The upstream archive documents:

- 41 measured variables (`XMEAS(1..41)`)
- 11 manipulated variables (`XMV(1..11)`)
- 52 variables in each benchmark observation
- fault-free training file `d00.dat`
- normal test file `d00_te.dat`
- fault files `d01_te.dat` through `d21_te.dat`
- 960 test observations per test run
- 3-minute sampling interval

The standard test protocol uses the first 160 observations as normal operation and introduces the designated fault from the following observation onward. In zero-based Python indexing the benchmark therefore uses `fault_start_index = 160`.

Dataset files are downloaded at benchmark time rather than committed into this repository.

## Detection benchmark

For each selected test fault:

1. fit healthy-reference normalization from `d00.dat`
2. calibrate PCA/DPCA monitoring on held-out normal data
3. calculate T2 and SPE/Q-style statistics
4. apply calibrated quantile limits and alarm persistence
5. calculate false alarm rate on samples 0..159
6. calculate post-fault detection rate on samples 160..end
7. record first post-fault alarm and detection delay

This is deliberately a transparent deterministic baseline.

## Root-cause / localization benchmark

The original public archive provides fault scenario IDs and descriptions, but it does **not** provide a canonical measured-variable root-cause label for every fault. We therefore avoid reporting a misleading universal root-cause ground truth.

Instead, `tsdiag.datasets.tep_knowledge` contains a conservative TEP knowledge base:

- variable descriptions for `XMEAS` and `XMV`
- IDV(1)-IDV(21) fault catalog entries
- expected root-variable priors for faults where the mechanism is known
- affected-variable neighbourhoods
- a sparse directed process-topology prior

These expected roots are engineering evaluation proxies, not simulator-perfect causal labels.

## Root-cause ablation protocol

The benchmark can compare four root-cause variants:

| Variant | Evidence used | Purpose |
|---|---|---|
| `generic` | PCA contribution + onset timing + Granger + pre/post shift | Tests pure data-driven RCA. |
| `topology_only` | Generic RCA + sparse TEP topology filter | Tests whether process structure helps. |
| `catalog_only` | Generic RCA + TEP fault catalog | Tests whether known fault mechanisms help. |
| `topology_catalog` | Generic RCA + topology + catalog | Tests the full knowledge-guided RCA path. |

The true IDV label is used only after prediction for scoring. During prediction the method sees the full TEP catalog, not the true fault ID.

Reported ablation metrics:

- root top-1 accuracy against proxy roots
- root top-3 accuracy
- catalog fault-ID accuracy
- abstention rate
- false-confident diagnosis rate
- mean confidence

The benchmark also writes a fault-ID confusion matrix for the full `topology_catalog` variant.

## Run locally

```bash
pip install -e ".[dev]"

python -m tsdiag.benchmarks.tep \
  --download \
  --faults 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
  --methods pca dpca \
  --target-far 0.05 \
  --diagnostic-method dpca \
  --diagnostic-faults 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
  --diagnostic-top-k 8 \
  --maxlag 1 \
  --run-ablation \
  --output outputs/tep_benchmark.json \
  --artifact-dir outputs/tep_benchmark
```

Outputs include:

- `outputs/tep_benchmark.json`
- `outputs/tep_benchmark/tep_fault_table.csv`
- `outputs/tep_benchmark/tep_root_cause_ablation.csv`
- `outputs/tep_benchmark/tep_summary.md`
- detection, delay, ablation and confusion-matrix plots under `outputs/tep_benchmark/plots/`

## CI

`.github/workflows/tep-benchmark.yml` downloads the public benchmark data, runs the project tests, evaluates all 21 TEP faults for detection, runs the root-cause ablation, and uploads the benchmark artifacts.

## Interpretation

A high post-fault detection rate is not enough by itself. We care about the trade-off among:

- detection rate
- false alarm rate
- detection delay
- fault-to-fault variability
- localization quality
- abstention / uncertainty
- computational cost

The root-cause ablation is the first check of the actual research claim: whether topology and fault-catalog knowledge improve diagnosis beyond generic contribution/Granger scoring. If the full knowledge-guided variant does not improve over generic RCA, the next step is not to add an LLM; it is to improve the TEP topology, fault catalog and scoring formulation.
