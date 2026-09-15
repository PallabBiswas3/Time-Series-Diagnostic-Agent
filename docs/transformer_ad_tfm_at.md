# Transformer diagnosis with AD-TFM-AT

This experimental transformer path implements the method in Q. Li et al., **“Incipient Fault Detection in Power Distribution System: A Time-Frequency Embedded Deep Learning Based Approach”** (arXiv:2302.09332).

## What is transferred from the paper

The learned model operates directly on a 100 x 6 SGAH event in channel order `Ua, Ub, Uc, Ia, Ib, Ic`.

1. **Joint state/time/frequency forgetting** — each recurrent step computes separate state (`D`), time (`K`) and frequency (`J`) forget gates and combines them by an outer-product style broadcast.
2. **Adaptive Morlet time-frequency memory** — the input gate creates a state vector from which the wavelet scale/frequency and translation parameters are learned at every time step. Real and imaginary time-frequency memories are updated and converted to amplitude before the next hidden state is produced.
3. **Global hidden-state attention** — every recurrent hidden state participates in a trainable context-vector attention pool instead of classifying only from the last step.
4. **Phase switching augmentation** — training events are expanded using the original phase order plus A<->B and A<->C swaps. Voltage and current from the same phase are always moved together.

The paper reports `D=32`, `K=4`, `J=4`, `omega0=16`, Adam, learning rate `0.001`, and batch size `256`; these are the benchmark defaults here too.

## Paper-style ablation

The paper explicitly compares four variants:

| Variant | Adaptive wavelet | Attention | Sequence readout |
| --- | --- | --- | --- |
| `TFM` | No | No | last hidden state |
| `AD-TFM` | Yes | No | last hidden state |
| `TFM-AT` | No | Yes | context-vector attention over all hidden states |
| `AD-TFM-AT` | Yes | Yes | context-vector attention over all hidden states |

All four variants use the same train/dev/test split, normalization procedure, phase-switch augmentation, `D/K/J/omega0`, optimizer, learning rate, batch size, epochs and seed. The only intended architectural changes are adaptive wavelet parameters and attention.

The paper says TFM uses fixed wavelet scale and translation parameters but does not provide their numeric fixed values in the paper text. The benchmark therefore makes the controlled choice `scale=1.0`, `translation=0.0` by default and records those values in the JSON output. They are **not** claimed as hidden paper hyperparameters.

Run the full four-model comparison with:

```bash
python scripts/run_transformer_sgah_ablation.py --download --epochs 30
```

The ablation report writes `outputs/transformer_sgah_ablation/transformer_sgah_tfm_ablation.json` and includes, for every model:

- multiclass accuracy, balanced accuracy and macro-F1;
- one-vs-rest macro AUC and per-class AUC;
- per-class precision/recall/F1/support;
- dedicated main-transformer-fault precision/recall/F1/AUC/specificity;
- trainable parameter count;
- training time and frozen-test inference time;
- epoch-wise train/development history.

The ranking is primarily by frozen-test macro-F1 and then accuracy. Headline conclusions should be made only after the complete ablation has run; architecture alone is not treated as evidence of improvement.

## Deliberate difference: temporal sliding

The paper also augments a longer fault recording by moving a fixed sampling window to different starting positions. The project's pinned SGAH loader, however, exposes the upstream data as already segmented 100-sample events. Applying temporal sliding inside one of these events would either shorten the event or require invented/padded samples. Therefore this benchmark **does not claim to reproduce temporal sliding**. If a future dataset preserves a longer continuous trace around each fault, temporal sliding should be added at the event-construction stage before the train/dev/test split is materialized.

## Relationship to the existing diagnostic pipeline

The AD-TFM-AT model does not replace the project's evidence layer. `ADTFMTransformerDiagnosticPipeline` first obtains the neural prediction from the raw six-channel waveform, then forwards that frozen prediction through the existing transformer pipeline. Synchronization checks, wavelet/correlation evidence, spectral evidence, optional electrical rules, verification, confidence handling and abstention therefore remain available.

The SGAH model is trained as a **five-class classifier** (single-phase ground, inter-phase short circuit, two-phase ground, main-transformer fault, normal), but only `main_transformer_fault` is promoted to the public transformer-fault diagnosis label. Other grid-fault probabilities remain model evidence rather than being mislabeled as internal transformer faults.

## Install and run

```bash
pip install -e ".[dev,deep]"
python scripts/run_transformer_sgah_ad_tfm.py --download --epochs 30
```

For a quick smoke run:

```bash
python scripts/run_transformer_sgah_ad_tfm.py --download --epochs 2 --device cpu
```

Results are written to `outputs/transformer_sgah_ad_tfm/transformer_sgah_ad_tfm_benchmark.json` by default. The report includes multiclass accuracy/macro-F1, per-class precision/recall/F1, binary main-transformer-fault metrics, training history, probabilities and the peak attention sample for every frozen test event.

## Evaluation discipline

- Split by complete event, not by individual rows.
- Use contiguous per-class 60/20/20 train/development/frozen-test partitions, matching the repository's existing SGAH benchmark protocol.
- Fit normalization on training data only.
- Apply phase augmentation to training data only.
- Never use frozen test labels for training, hyperparameter fitting or normalization.
- Keep the same seed and training hyperparameters across the four ablation variants.
- Compare the winning deep model against the existing Random-Forest spectral-representation baseline before replacing the baseline in any headline result.
