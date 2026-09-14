"""Small examples for the stable cross-domain diagnostic entry point."""

import numpy as np

from tsdiag import diagnose

rng = np.random.default_rng(42)

examples = {
    "bearing": dict(signal=rng.normal(size=4096), sampling_rate_hz=4000.0),
    "process": dict(signal_matrix=rng.normal(size=(80, 3)), normal_reference=rng.normal(size=(160, 3)),
                    channel_names=["pressure", "flow", "level"], sampling_rate_hz=1.0, maxlag=1),
    "wind_scada": dict(signal_matrix=rng.normal(size=(80, 3)), normal_reference=rng.normal(size=(120, 3)),
                       channel_names=["power", "temperature", "wind"], timestamps=np.arange(80)),
    "battery": dict(cell_voltage=3.7 + rng.normal(scale=.005, size=(80, 4)),
                    cell_temperature=30 + rng.normal(scale=.1, size=(80, 4)),
                    cell_ids=["c1", "c2", "c3", "c4"], timestamps=np.arange(80)),
    "turbofan": dict(signal_matrix=rng.normal(size=(100, 3)), channel_names=["s1", "s2", "s3"],
                     cycle_index=np.arange(100)),
    "transformer": dict(signal_matrix=rng.normal(size=(2048, 3)), sampling_rate_hz=4000.0,
                        sensor_positions=["left", "center", "right"]),
}

for domain, inputs in examples.items():
    result = diagnose(domain, **inputs)
    print(domain, result.decision, result.confidence, result.abstain_reason)
