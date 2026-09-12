#!/usr/bin/env python3
"""Run the complete CARE-to-Compare v6 Wind-SCADA benchmark.

This is the canonical PR #11 real-data entrypoint and accepts either the extracted
CARE directory or the official ZIP archive directly.

Usage:
    python scripts/run_wind_scada_benchmark.py --data-dir CARE_To_Compare.zip
"""

from __future__ import annotations

import argparse
import json

from tsdiag.benchmarks.wind_scada import run_care_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full CARE v6 Wind-SCADA evaluation")
    parser.add_argument("--data-dir", required=True, help="CARE v6 directory or CARE_To_Compare.zip")
    parser.add_argument("--output-dir", default="outputs/wind_benchmark")
    parser.add_argument("--n-regimes", type=int, default=4)
    parser.add_argument("--residual-threshold", type=float, default=3.5)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--cusum-drift", type=float, default=0.25)
    parser.add_argument("--cusum-threshold", type=float, default=8.0)
    parser.add_argument("--cusum-hold-samples", type=int, default=6)
    parser.add_argument("--criticality-threshold", type=int, default=72)
    args = parser.parse_args()

    result = run_care_benchmark(
        args.data_dir,
        n_regimes=args.n_regimes,
        residual_threshold=args.residual_threshold,
        persistence=args.persistence,
        cusum_drift=args.cusum_drift,
        cusum_threshold=args.cusum_threshold,
        cusum_hold_samples=args.cusum_hold_samples,
        criticality_threshold=args.criticality_threshold,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2))
    if result["failures"]:
        print(json.dumps({"failures": result["failures"]}, indent=2))


if __name__ == "__main__":
    main()
