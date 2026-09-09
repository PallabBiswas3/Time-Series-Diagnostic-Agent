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
2. build the PCA normal subspace from the same healthy reference
3. calculate T2 and SPE/Q-style statistics
4. derive 99% limits from the healthy reference
5. calculate false alarm rate on samples 0..159
6. calculate post-fault detection rate on samples 160..end
7. record first post-fault alarm and detection delay

This is deliberately a transparent deterministic baseline.

## Root-cause / localization benchmark

The original public archive provides fault scenario IDs and descriptions, but it does **not** provide a canonical measured-variable root-cause label for every fault. We therefore do not report a misleading universal `root_cause_accuracy`.

Instead, a conservative subset of faults has explicit engineering **proxy localization targets**. Examples include cooling-water manipulated variables for valve-sticking faults and the A-feed variables for A-feed loss. These proxies are clearly labeled as benchmark proxies, not canonical ground truth.

For those cases:

1. rank all 52 variables using PCA residual contribution
2. retain the top-k candidate channels
3. run stationarity analysis
4. difference flagged channels before Granger screening
5. estimate directed predictive links
6. estimate per-channel anomaly onset
7. rank candidate root causes from contribution + onset + causal influence
8. compare the top root-cause candidate against the declared engineering proxy set

The benchmark reports this separately as `proxy_localization_accuracy`.

## Run locally

```bash
pip install -e ".[dev]"

python -m tsdiag.benchmarks.tep \
  --download \
  --faults 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 \
  --diagnostic-faults 4 5 6 14 15 21 \
  --diagnostic-top-k 6 \
  --maxlag 1 \
  --output outputs/tep_benchmark.json
```

## CI

`.github/workflows/tep-benchmark.yml` downloads the public benchmark data, runs the project tests, evaluates all 21 TEP faults for detection, runs the bounded root-cause stage on the proxy-labeled subset, and uploads `tep_benchmark.json` as a workflow artifact.

## Interpretation

A high post-fault detection rate is not enough by itself. We care about the trade-off among:

- detection rate
- false alarm rate
- detection delay
- fault-to-fault variability
- localization quality
- abstention / uncertainty
- computational cost

The real-data benchmark is intended to expose weaknesses in the current deterministic process pipeline before an adaptive or LLM-based controller is introduced.
