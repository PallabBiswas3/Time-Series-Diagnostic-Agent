from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsdiag.benchmarks.battery_nasa import run_nasa_battery_benchmark
from tsdiag.datasets.battery_nasa import download_nasa_battery_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed NASA Li-ion battery prognosis benchmark")
    parser.add_argument("--data-dir", default="data/nasa_battery")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", default="outputs/battery_nasa")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.download:
        paths = download_nasa_battery_subset(data_dir)
        print(f"NASA battery files ready: {len(paths)}")

    result = run_nasa_battery_benchmark(data_dir, output_dir=args.output_dir)
    print(json.dumps(result["summary"], indent=2))
    if result["failures"]:
        print("Failures:")
        print(json.dumps(result["failures"], indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
