# Diagnostic pipeline 1.0.0

Version `1.0.0` freezes the public `diagnose(domain, **inputs) -> DiagnosticResult` contract before benchmark optimization begins.

The contract guarantees:

- all six declared domains are executable;
- input failures and unsupported labels produce structured abstentions;
- detection, localization, hypotheses, verification and prognosis remain separate;
- evidence identifiers referenced by hypotheses and trace entries are validated;
- every result includes the pipeline version and an inspectable tool trace.

Bearing and process retain their existing domain algorithms behind the shared dispatcher. Wind SCADA, battery, turbofan and transformer provide deterministic baselines. Optional trained predictors may replace or extend their model-dependent steps through callable inputs.

Changes after this freeze must preserve the `DiagnosticResult` schema or introduce an explicit new schema/pipeline version. Benchmark calibration must use held-out units and must not silently change the meaning of the public result fields.
