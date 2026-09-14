# Diagnostic pipeline 1.0.0

Version `1.0.0` freezes the public `diagnose(domain, **inputs) -> DiagnosticResult` contract before benchmark optimization begins.

The contract guarantees:

- all six declared domains are executable;
- input failures and unsupported labels produce structured abstentions;
- detection, localization, hypotheses, verification and prognosis remain separate;
- evidence identifiers referenced by hypotheses and trace entries are validated;
- every result includes the pipeline version and an inspectable tool trace.
- every domain contract step resolves to a registered callable;
- step failures are isolated with the failed tool name, error type and elapsed time;
- `DiagnosticResult.to_dict()` and `to_json()` convert NumPy values to JSON-native values;
- `get_input_schema(domain)` exposes required and optional public inputs.

Bearing and process retain their existing domain algorithms behind the shared dispatcher. Wind SCADA, battery, turbofan and transformer provide deterministic baselines. Optional trained predictors may replace or extend their model-dependent steps through callable inputs.

Changes after this freeze must preserve the `DiagnosticResult` schema or introduce an explicit new schema/pipeline version. Benchmark calibration must use held-out units and must not silently change the meaning of the public result fields.

Run `python examples/run_full_pipeline.py` after installation to execute normal and fault scenarios for bearing, process, wind SCADA, battery, turbofan and transformer.

Individual contracted tools are exposed through `default_domain_tool_registry()`. Each callable accepts a mutable pipeline-state dictionary and returns the named outputs it produced. `StepExecutor` merges those outputs into state and records status, input/output keys, elapsed time, and any exception details.

```python
from tsdiag.domains import default_domain_tool_registry

registry = default_domain_tool_registry()
quality = registry.get("wind_scada", "scada_quality_check")({
    "signal_matrix": values,
    "channel_names": channel_names,
    "timestamps": timestamps,
})
```
