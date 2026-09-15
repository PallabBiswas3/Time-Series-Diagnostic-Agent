# Paderborn CNN–physics bearing pipeline

The primary bearing benchmark uses Paderborn vibration and motor-current
waveforms at 64 kHz. Rotational speed is supporting metadata: it converts the
spectrum to shaft order and locates BPFO, BPFI, BSF and FTF families. This is an
adaptation of Guo, Yang and Huang (2022), not a reproduction of their private
2 kHz motor-speed dataset.

The implemented ablation is `physics → CNN → CNN+physics`. The neural branch is
a reproducible 1D replacement for the paper's arbitrary 100×100 reshape: two
convolution/ReLU/average-pooling blocks feed a 64-value representation. The
physics branch contains 1×–8× shaft orders, bearing-frequency harmonics,
vibration-envelope harmonics, current sidebands, kurtosis, crest factor, RMS
and spectral entropy. Fusion concatenates both representations before the
classifier.

Paderborn evaluation uses a bearing-specimen holdout. All windows and operating
conditions from a held-out physical bearing remain outside training. Mixed KB
damage is excluded by default because it is not a bearing-plus-misalignment
label. Enable it only with `--include-combined` and report that taxonomy.

Place the licensed Paderborn MATLAB files below `data/bearing_paderborn`, then:

```powershell
python scripts/run_bearing_hybrid_benchmark.py --dataset paderborn --epochs 15
```

Lenze-MB is evaluated separately. It uses the documented 16 kHz `StromBox_Werte`
channels and compares current-only, speed-only and current-plus-speed inputs.
The maximum RPM condition is held out, so this tests operating-condition
transfer rather than random windows from the same recording.

```powershell
python scripts/run_bearing_hybrid_benchmark.py --dataset lenze --download --epochs 15
```

Raw datasets, trained `.pt` models and reports live below `data/` and `outputs/`,
which are excluded by the repository `.gitignore`.

The benchmark prints record-preparation progress and one line per epoch with
training loss, epoch duration and estimated remaining time. After every
ablation model it prints record-level accuracy, balanced accuracy and macro-F1.
