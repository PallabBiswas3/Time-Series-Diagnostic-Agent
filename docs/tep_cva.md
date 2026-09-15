# Tennessee Eastman CVA diagnosis path

The TEP research path now separates three questions:

1. **Detection:** calibrated PCA, DPCA and CVA monitoring are compared under the
   same healthy-only false-alarm calibration protocol.
2. **Diagnosis:** CVA dynamic state features feed either an RBF SVM (the
   benchmark-selected default) or shrinkage FDA as an interpretable baseline.
   Only `dXX.dat` training simulations fit this stage.
3. **Root cause:** CVA alarms and residual contributions can feed the existing
   onset, Granger, process-topology and knowledge-catalog reasoning pipeline.

The CVA projection is fitted on normal operation. Past and future blocks learn
the projection, but inference uses past blocks only, so online statistics do not
look into future test samples. Control limits are selected using held-out healthy
data. Test labels never fit normalization, CVA, limits, FDA or SVM.

Faulty training sequences begin at sample 20 and frozen test sequences begin at
sample 160. The benchmark therefore does not incorrectly label the normal prefix
of each fault simulation as faulty.

Download the Braatz data and run faults 0 through 20:

```powershell
python scripts/run_tep_cva_benchmark.py --download
```

Run again from the local cache:

```powershell
python scripts/run_tep_cva_benchmark.py
```

For a faster development check:

```powershell
python scripts/run_tep_cva_benchmark.py --faults 0,1,4,6 --max-samples-per-class 300
```

The JSON report includes per-method detection rate, false-alarm rate and delay,
plus event-level fault-ID accuracy, balanced accuracy, macro-F1 and the complete
confusion matrix. Results should be reported per fault; an aggregate score must
not hide weak faults such as 3, 9 or 15.

On the frozen faults 0--20 run, CVA-SVM reached 81.0% event accuracy and 76.5%
macro-F1 versus 76.2% and 70.6% for CVA-FDA. With the fixed abstention policy,
SVM covered 76.2% of events at 93.8% covered accuracy. These are single-protocol
benchmark results, not a claim of universal superiority.

The deployed decision rule runs DPCA and CVA together. DPCA-only evidence is an
early warning; CVA confirmation or a strong persistent DPCA alarm produces a
confirmed fault. The frozen benchmark produced 81.68% mean detection coverage,
0.71% pre-fault false alarms and a 27.4-sample mean delay for this hybrid.

Classifier artifacts include normal-reference scaling, the CVA projection,
feature scaling and the fitted estimator:

```powershell
python scripts/run_tep_cva_benchmark.py --classifier svm --model-output outputs/tep_cva/tep_cva_svm.joblib
python scripts/run_tep_cva_benchmark.py --model-input outputs/tep_cva/tep_cva_svm.joblib
```

Only load trusted Joblib artifacts. Dataset and model-output directories remain
excluded by `.gitignore`.

Run the causal/root-cause evaluation separately:

```powershell
python scripts/run_tep_causal_benchmark.py
```

On the current real-data run it attempted all 20 faults, produced a root for 11,
and had 5 correct roots among 15 faults with engineering proxy targets. That is
71.4% conditional accuracy at 46.7% labeled-case coverage, or 33.3%
unconditional accuracy. The coverage figures must accompany the accuracy.
