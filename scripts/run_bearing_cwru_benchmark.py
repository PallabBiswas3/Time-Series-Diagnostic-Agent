from __future__ import annotations

import argparse
import json
from pathlib import Path

from tsdiag import diagnose
from tsdiag.benchmarks import bearing_cwru as benchmark_module
from tsdiag.contracts import DiagnosticRequest
from tsdiag.datasets.bearing_cwru import download_cwru_007_drive_end


class _PublicBearingPipeline:
    """Benchmark compatibility adapter that exercises the public diagnose boundary."""

    def __init__(self, *, minimum_confidence: float = 0.45, minimum_harmonics: int = 2):
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_harmonics = int(minimum_harmonics)

    def run(
        self,
        signal,
        sampling_rate_hz: float,
        *,
        fault_frequencies=None,
        shaft_rate_hz=None,
        channel_name="ch0",
        operating_condition=None,
    ):
        return diagnose(DiagnosticRequest(
            domain="bearing",
            task="fault_diagnosis",
            policy_ref="compat-1.0",
            inputs={
                "signal": signal,
                "sampling_rate_hz": sampling_rate_hz,
                "fault_frequencies": dict(fault_frequencies or {}),
                "shaft_rate_hz": shaft_rate_hz,
                "channel_name": channel_name,
                "operating_condition": dict(operating_condition or {}),
                "minimum_confidence": self.minimum_confidence,
                "minimum_harmonics": self.minimum_harmonics,
            },
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled CWRU 0.007-in drive-end bearing benchmark")
    parser.add_argument("--data-dir", default="data/cwru_007_de")
    parser.add_argument("--download", action="store_true", help="download the fixed CWRU manifest before evaluation")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--minimum-confidence", type=float, default=0.45)
    parser.add_argument("--minimum-harmonics", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/bearing_cwru")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.download:
        paths = download_cwru_007_drive_end(data_dir)
        print(f"CWRU files ready: {len(paths)}")

    # Keep the frozen benchmark/evaluation protocol intact while replacing only
    # its diagnostic execution boundary. Labels and aggregation remain inside the
    # benchmark module and are never supplied to diagnose().
    benchmark_module.BearingDiagnosticPipeline = _PublicBearingPipeline
    result = benchmark_module.run_cwru_benchmark(
        data_dir,
        window_seconds=args.window_seconds,
        max_windows=args.max_windows,
        minimum_confidence=args.minimum_confidence,
        minimum_harmonics=args.minimum_harmonics,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2))
    if result["failures"]:
        print("Failures:")
        print(json.dumps(result["failures"], indent=2))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
