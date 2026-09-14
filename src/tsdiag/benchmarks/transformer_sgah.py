from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..datasets.transformer_sgah import SGAH_CHANNEL_NAMES, SGAH_CLASSES, SgahEvent, load_sgah_events
from ..domains.runners import TransformerDiagnosticPipeline
from ..tools.transformer_rules import fit_transformer_rule_reference, transformer_electrical_features, transformer_rule_diagnosis


@dataclass
class SgahCase:
    class_id:int; label:str; event_id:int; true_transformer_fault:bool; predicted_transformer_fault:bool
    decision:str; abstained:bool; confidence:float; verification_status:str|None; runtime_seconds:float


def _split(events:list[SgahEvent]):
    n=len(events); a=max(1,int(np.floor(.60*n))); b=min(max(a+1,int(np.floor(.80*n))) if n>=3 else n,n)
    return events[:a],events[a:b],events[b:]


def _metrics(events,predicted):
    y=np.asarray([e.class_id==4 for e in events],bool); p=np.asarray(predicted,bool)
    tp=int(np.sum(p&y)); fn=int(np.sum((~p)&y)); fp=int(np.sum(p&(~y))); tn=int(np.sum((~p)&(~y)))
    rec=0.0 if tp+fn==0 else tp/(tp+fn); spec=0.0 if tn+fp==0 else tn/(tn+fp)
    normal=np.asarray([e.class_id==5 for e in events]); competing=np.asarray([e.class_id in (1,2,3) for e in events])
    return {"recall":rec,"specificity":spec,"balanced_accuracy":.5*(rec+spec),"normal_false_positive_rate":float(np.mean(p[normal])) if np.any(normal) else 0.0,"competing_fault_false_positive_rate":float(np.mean(p[competing])) if np.any(competing) else 0.0,"tp":tp,"fn":fn,"fp":fp,"tn":tn}


def _calibrate(reference,dev_events):
    rows=[]
    for threshold in np.arange(2.0,8.01,.25):
        for external in np.arange(2.0,12.01,.5):
            pred=[transformer_rule_diagnosis(e.signal_matrix,reference,threshold=float(threshold),external_asymmetry_threshold=float(external))["transformer_fault"] for e in dev_events]
            rows.append({"threshold":float(threshold),"external_asymmetry_threshold":float(external),**_metrics(dev_events,pred)})
    feasible=[r for r in rows if r["recall"]>=.60 and r["normal_false_positive_rate"]<=.15 and r["competing_fault_false_positive_rate"]<=.30]
    if feasible:
        best=max(feasible,key=lambda r:(r["balanced_accuracy"],r["recall"],-r["competing_fault_false_positive_rate"],r["threshold"]))
    else:
        def violation(r): return max(.60-r["recall"],0)+max(r["normal_false_positive_rate"]-.15,0)+max(r["competing_fault_false_positive_rate"]-.30,0)
        best=max(rows,key=lambda r:(-violation(r),r["balanced_accuracy"],r["recall"]))
    return best


def _binary(rows):
    tp=sum(r.true_transformer_fault and r.predicted_transformer_fault for r in rows); fn=sum(r.true_transformer_fault and not r.predicted_transformer_fault for r in rows)
    fp=sum((not r.true_transformer_fault) and r.predicted_transformer_fault for r in rows); tn=sum((not r.true_transformer_fault) and not r.predicted_transformer_fault for r in rows)
    rec=None if tp+fn==0 else tp/(tp+fn); spec=None if tn+fp==0 else tn/(tn+fp); prec=None if tp+fp==0 else tp/(tp+fp)
    f1=None if prec is None or rec is None or prec+rec==0 else 2*prec*rec/(prec+rec); bal=None if rec is None or spec is None else .5*(rec+spec)
    sup=sum(r.verification_status=="SUPPORTED" for r in rows)
    return {"case_count":len(rows),"positive_count":tp+fn,"negative_count":tn+fp,"true_positive":tp,"false_negative":fn,"false_positive":fp,"true_negative":tn,"recall":rec,"specificity":spec,"precision":prec,"f1":f1,"balanced_accuracy":bal,"coverage":1.0 if rows else None,"abstention_rate":0.0 if rows else None,"verification_support_rate":None if not rows else sup/len(rows),"mean_runtime_seconds":None if not rows else float(np.mean([r.runtime_seconds for r in rows]))}


def run_sgah_transformer_benchmark(data_dir:str|Path,*,output_dir:str|Path|None=None)->dict:
    root=Path(data_dir); train_by_class={}; dev_events=[]; test_events=[]; manifest=[]
    for cid,label in SGAH_CLASSES.items():
        events=load_sgah_events(root/f"{cid}-data.csv",cid); train,dev,test=_split(events)
        if not train or not dev or not test: raise ValueError(f"SGAH class {cid} does not have enough whole events for frozen split")
        train_by_class[cid]=train; dev_events.extend(dev); test_events.extend(test); manifest.append({"class_id":cid,"label":label,"event_count":len(events),"train_count":len(train),"dev_count":len(dev),"test_count":len(test)})
    reference=fit_transformer_rule_reference([transformer_electrical_features(e.signal_matrix) for e in train_by_class[5]])
    calibration=_calibrate(reference,dev_events); threshold=calibration["threshold"]; external=calibration["external_asymmetry_threshold"]
    pipeline=TransformerDiagnosticPipeline(); cases=[]; failures=[]
    for e in test_events:
        try:
            start=perf_counter(); result=pipeline.run(e.signal_matrix,sampling_rate_hz=1.0,sensor_positions=list(SGAH_CHANNEL_NAMES),transformer_rule_reference=reference,transformer_rule_threshold=threshold,transformer_external_asymmetry_threshold=external,allow_ml_fallback=False); runtime=perf_counter()-start
            pred=bool(result.decision=="diagnose" and any(h.label=="main_transformer_fault" for h in result.hypotheses)); status=str(result.verification[0].status) if result.verification else None
            cases.append(SgahCase(e.class_id,e.label,e.event_id,e.class_id==4,pred,result.decision,bool(result.abstained),float(result.confidence),status,float(runtime)))
        except Exception as exc: failures.append({"class_id":e.class_id,"event_id":e.event_id,"error":f"{type(exc).__name__}: {exc}"})
    normal=[r for r in cases if r.class_id==5]; competing=[r for r in cases if r.class_id in (1,2,3)]; summary=_binary(cases)
    summary.update({"normal_false_positive_rate":None if not normal else sum(r.predicted_transformer_fault for r in normal)/len(normal),"competing_fault_false_positive_rate":None if not competing else sum(r.predicted_transformer_fault for r in competing)/len(competing)})
    by_class={str(cid):{"label":SGAH_CLASSES[cid],**_binary([r for r in cases if r.class_id==cid])} for cid in SGAH_CLASSES}
    payload={"source":"smartlab-hfut/SGAH-datasets (State Grid Corporation of China)","protocol":{"upstream_commit":"bbe1020e3fade83f7861657bb3eaea41c25ec0c9","samples_per_event":100,"split":"Per class, contiguous whole-event 60% train / 20% development / 20% frozen test; no row-level splitting.","positive_class":"main_transformer_fault (class 4)","negative_controls":"classes 1,2,3 competing grid faults plus class 5 normal","primary_method":"deterministic transformer protection rules; no ML classifier","normal_reference":"Robust median/MAD envelope fitted only from class-5 normal training events.","rule_features":["three-phase voltage/current RMS","positive/negative/zero sequence ratios","phase imbalance","voltage/current transient ratios","voltage/current THD","apparent impedance","impedance phase spread","power phase imbalance"],"threshold_calibration":"Two explicit protection-rule thresholds selected on development events only: disturbance/internal threshold and external sequence/asymmetry rejection threshold. Frozen test labels excluded.","rule_threshold":threshold,"external_asymmetry_threshold":external,"ml_fallback":False},"manifest":manifest,"development":{"dev_rule_calibration":calibration},"normal_reference":{"center":reference.center,"scale":reference.scale},"summary":summary,"by_class":by_class,"cases":[asdict(r) for r in cases],"failures":failures}
    if output_dir is not None:
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"transformer_sgah_benchmark.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
        if cases:
            with (out/"transformer_sgah_cases.csv").open("w",newline="",encoding="utf-8") as h:
                w=csv.DictWriter(h,fieldnames=list(asdict(cases[0]).keys())); w.writeheader(); w.writerows(asdict(r) for r in cases)
    return payload
