from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..datasets.transformer_sgah import SGAH_CHANNEL_NAMES, SGAH_CLASSES, SgahEvent, load_sgah_events
from ..domains.domain_steps import transformer_corr, transformer_denoise, transformer_fuse, transformer_representation, transformer_spectral, transformer_sync, transformer_weights
from ..domains.runners import TransformerDiagnosticPipeline
from ..tools.transformer_rules import fit_transformer_rule_reference, transformer_electrical_features, transformer_rule_diagnosis


@dataclass
class SgahCase:
    class_id:int; label:str; event_id:int; true_transformer_fault:bool; predicted_transformer_fault:bool
    decision:str; abstained:bool; confidence:float; runtime_seconds:float


def _split(events:list[SgahEvent]):
    n=len(events); a=max(1,int(np.floor(.60*n))); b=min(max(a+1,int(np.floor(.80*n))) if n>=3 else n,n)
    return events[:a],events[a:b],events[b:]


def _feature_image(event:SgahEvent)->np.ndarray:
    s={"signal_matrix":event.signal_matrix,"sampling_rate_hz":1.0,"sensor_positions":list(SGAH_CHANNEL_NAMES)}
    for fn in (transformer_sync,transformer_denoise,transformer_corr,transformer_weights,transformer_fuse,transformer_spectral,transformer_representation):
        s.update(fn(s))
    return np.asarray(s["feature_image"],dtype=float)


class _FixedSgahClassifier:
    def __init__(self,estimator): self.estimator=estimator
    def __call__(self,feature_image):
        p=float(self.estimator.predict_proba(np.asarray(feature_image,dtype=float).reshape(1,-1))[0,1])
        return {"label":"main_transformer_fault" if p>=.5 else None,"confidence":max(p,1-p),"probabilities":{"main_transformer_fault":p,"not_transformer_fault":1-p}}


def _rule_metrics(events,predicted):
    y=np.asarray([e.class_id==4 for e in events],bool); p=np.asarray(predicted,bool)
    tp=int(np.sum(p&y)); fn=int(np.sum((~p)&y)); fp=int(np.sum(p&(~y))); tn=int(np.sum((~p)&(~y)))
    rec=0.0 if tp+fn==0 else tp/(tp+fn); spec=0.0 if tn+fp==0 else tn/(tn+fp)
    normal=np.asarray([e.class_id==5 for e in events]); competing=np.asarray([e.class_id in (1,2,3) for e in events])
    return {"recall":rec,"specificity":spec,"balanced_accuracy":.5*(rec+spec),"normal_false_positive_rate":float(np.mean(p[normal])) if np.any(normal) else 0.0,"competing_fault_false_positive_rate":float(np.mean(p[competing])) if np.any(competing) else 0.0}


def _calibrate_rules(reference,dev_events):
    rows=[]
    for threshold in np.arange(2.0,8.01,.25):
        for external in np.arange(2.0,12.01,.5):
            pred=[transformer_rule_diagnosis(e.signal_matrix,reference,threshold=float(threshold),external_asymmetry_threshold=float(external))["transformer_fault"] for e in dev_events]
            rows.append({"threshold":float(threshold),"external_asymmetry_threshold":float(external),**_rule_metrics(dev_events,pred)})
    feasible=[r for r in rows if r["recall"]>=.60 and r["normal_false_positive_rate"]<=.15 and r["competing_fault_false_positive_rate"]<=.30]
    if feasible: return max(feasible,key=lambda r:(r["balanced_accuracy"],r["recall"],-r["competing_fault_false_positive_rate"]))
    def violation(r): return max(.60-r["recall"],0)+max(r["normal_false_positive_rate"]-.15,0)+max(r["competing_fault_false_positive_rate"]-.30,0)
    return max(rows,key=lambda r:(-violation(r),r["balanced_accuracy"],r["recall"]))


def _binary(rows:list[SgahCase]):
    tp=sum(r.true_transformer_fault and r.predicted_transformer_fault for r in rows); fn=sum(r.true_transformer_fault and not r.predicted_transformer_fault for r in rows)
    fp=sum((not r.true_transformer_fault) and r.predicted_transformer_fault for r in rows); tn=sum((not r.true_transformer_fault) and not r.predicted_transformer_fault for r in rows)
    rec=None if tp+fn==0 else tp/(tp+fn); spec=None if tn+fp==0 else tn/(tn+fp); prec=None if tp+fp==0 else tp/(tp+fp)
    f1=None if prec is None or rec is None or prec+rec==0 else 2*prec*rec/(prec+rec)
    covered=sum(not r.abstained for r in rows)
    return {"case_count":len(rows),"positive_count":tp+fn,"negative_count":tn+fp,"true_positive":tp,"false_negative":fn,"false_positive":fp,"true_negative":tn,"recall":rec,"specificity":spec,"precision":prec,"f1":f1,"balanced_accuracy":None if rec is None or spec is None else .5*(rec+spec),"coverage":None if not rows else covered/len(rows),"abstention_rate":None if not rows else 1-covered/len(rows),"mean_runtime_seconds":None if not rows else float(np.mean([r.runtime_seconds for r in rows]))}


def run_sgah_transformer_benchmark(data_dir:str|Path,*,output_dir:str|Path|None=None)->dict:
    root=Path(data_dir); train_events=[]; train_by_class={}; dev_events=[]; test_events=[]; manifest=[]
    for cid,label in SGAH_CLASSES.items():
        events=load_sgah_events(root/f"{cid}-data.csv",cid); train,dev,test=_split(events)
        train_by_class[cid]=train; train_events.extend(train); dev_events.extend(dev); test_events.extend(test)
        manifest.append({"class_id":cid,"label":label,"event_count":len(events),"train_count":len(train),"dev_count":len(dev),"test_count":len(test)})

    train_x=np.stack([_feature_image(e).reshape(-1) for e in train_events]); train_y=np.asarray([e.class_id==4 for e in train_events],dtype=int)
    estimator=RandomForestClassifier(n_estimators=400,max_depth=None,min_samples_leaf=2,class_weight="balanced_subsample",random_state=0,n_jobs=-1)
    estimator.fit(train_x,train_y); classifier=_FixedSgahClassifier(estimator)

    reference=fit_transformer_rule_reference([transformer_electrical_features(e.signal_matrix) for e in train_by_class[5]])
    calibration=_calibrate_rules(reference,dev_events); threshold=calibration["threshold"]; external=calibration["external_asymmetry_threshold"]

    def raw_acc(events):
        x=np.stack([_feature_image(e).reshape(-1) for e in events]); y=np.asarray([e.class_id==4 for e in events],dtype=int)
        return float(np.mean(estimator.predict(x)==y))

    pipeline=TransformerDiagnosticPipeline(); cases=[]; failures=[]
    for e in test_events:
        try:
            start=perf_counter(); result=pipeline.run(e.signal_matrix,sampling_rate_hz=1.0,sensor_positions=list(SGAH_CHANNEL_NAMES),trained_image_model=classifier,transformer_rule_reference=reference,transformer_rule_threshold=threshold,transformer_external_asymmetry_threshold=external,hybrid_arbitration=True,anomaly_threshold=0.0,classifier_diagnosis_threshold=0.5); runtime=perf_counter()-start
            pred=bool(result.decision=="diagnose" and any(h.label=="main_transformer_fault" for h in result.hypotheses))
            cases.append(SgahCase(e.class_id,e.label,e.event_id,e.class_id==4,pred,result.decision,bool(result.abstained),float(result.confidence),float(runtime)))
        except Exception as exc: failures.append({"class_id":e.class_id,"event_id":e.event_id,"error":f"{type(exc).__name__}: {exc}"})

    normal=[r for r in cases if r.class_id==5]; competing=[r for r in cases if r.class_id in (1,2,3)]; summary=_binary(cases)
    summary.update({"normal_false_positive_rate":None if not normal else sum(r.predicted_transformer_fault for r in normal)/len(normal),"competing_fault_false_positive_rate":None if not competing else sum(r.predicted_transformer_fault for r in competing)/len(competing)})
    by_class={str(cid):{"label":SGAH_CLASSES[cid],**_binary([r for r in cases if r.class_id==cid])} for cid in SGAH_CLASSES}
    payload={"source":"smartlab-hfut/SGAH-datasets","protocol":{"upstream_commit":"bbe1020e3fade83f7861657bb3eaea41c25ec0c9","split":"per-class whole-event 60/20/20","positive_class":"main_transformer_fault","classifier":"fixed RandomForest, threshold 0.5","rules":"normal-reference electrical protection features with dev-only thresholds","arbiter":"hybrid_ml_physics_arbitration_v1","frozen_test_labels":"scoring only"},"manifest":manifest,"development":{"train_raw_classifier_accuracy":raw_acc(train_events),"dev_raw_classifier_accuracy":raw_acc(dev_events),"rule_calibration":calibration},"summary":summary,"by_class":by_class,"cases":[asdict(r) for r in cases],"failures":failures}
    if output_dir is not None:
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"transformer_sgah_benchmark.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
        if cases:
            with (out/"transformer_sgah_cases.csv").open("w",newline="",encoding="utf-8") as h:
                w=csv.DictWriter(h,fieldnames=list(asdict(cases[0]).keys())); w.writeheader(); w.writerows(asdict(r) for r in cases)
    return payload
