# Six-domain diagnostic architecture

The repository uses one shared diagnostic-agent core plus six domain packs. A domain pack is a declarative, ordered analysis contract: each step states its required inputs, optional inputs, outputs, evidence fields, preconditions/failure modes, and (where already available) its concrete implementation.

## v1 domain packs

| Domain | Primary task | Distinctive analysis |
|---|---|---|
| `bearing` | mechanical fault localization | impulsiveness, PSD/STFT, spectral kurtosis, resonance filtering, Hilbert envelope, BPFO/BPFI/BSF/FTF matching |
| `process` | multivariate fault + root cause | PCA/SPE/T2, contribution analysis, stationarity, Granger graph, onset/propagation reasoning |
| `wind_scada` | condition monitoring / early warning | operating regimes, normal-behavior residuals, rolling statistics, correlation drift, change points, physics checks |
| `battery` | cell anomaly + prognosis | operating-state segmentation, cell-to-pack deviations, spatial consistency, temporal trends, multihorizon spatio-temporal reasoning |
| `turbofan` | degradation / RUL | sensor screening, regime normalization, trend extraction, sequence windows, health index, RUL + uncertainty |
| `transformer` | multisensor mechanical fault diagnosis | wavelet denoising, correlation weighting, multisensor fusion, Fast Spectral Correlation, transfer-learning classifier |

## Shared contract

`ToolContract` fields:

- `name`: stable tool identifier for deterministic/LLM/RL routers
- `purpose`: why the step exists
- `required_inputs`: values that must be present
- `optional_inputs`: context that can improve the step
- `outputs`: named values produced by the tool
- `preconditions`: conditions that must hold before execution
- `evidence_fields`: outputs that can support a diagnostic claim
- `failure_modes`: explicit reasons a tool can fail/abstain
- `implementation`: current concrete agent implementation when one already exists

`DomainPack` additionally declares required/optional metadata, ordered tools, final outputs and benchmark targets.

## Planning

```python
from tsdiag import build_execution_plan, get_domain_pack, describe_contract

plan = build_execution_plan(
    "bearing",
    {"sampling_rate_hz": 12000.0, "shaft_rate_hz": 30.0},
)

assert plan.ready
print(plan.tool_sequence)

for tool in describe_contract(get_domain_pack("process")):
    print(tool["name"], tool["required_inputs"], "->", tool["outputs"])
```

The planner intentionally does **not** pretend every algorithm is already implemented. It separates architecture/contract completeness from implementation completeness, so missing tools can be added and benchmarked one-by-one without changing the domain specification.

## Intended adaptive routing

The long-term controller receives:

```text
industrial data + metadata + diagnostic objective
                ↓
         identify domain pack
                ↓
      inspect available evidence
                ↓
 select/skip/repeat contracted tools
                ↓
       verify physical consistency
                ↓
 diagnose / prognose / abstain
```

The ordered pack is the deterministic baseline. Later adaptive, LLM, or RL policies should be evaluated against it rather than assumed to be superior.
