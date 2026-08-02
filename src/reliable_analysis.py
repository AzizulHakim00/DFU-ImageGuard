from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    y=np.asarray(y,int); p=np.asarray(p,float); edges=np.linspace(0,1,bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if m.any(): out += m.mean()*abs(float(y[m].mean())-float(p[m].mean()))
    return float(out)


def metric_dict(y, p, pred) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score
    y=np.asarray(y,int); p=np.asarray(p,float); pred=np.asarray(pred,int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"sensitivity":float(recall_score(y,pred,zero_division=0)),"specificity":float(tn/max(tn+fp,1)),"precision":float(precision_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"mcc":float(matthews_corrcoef(y,pred)),"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,np.clip(p,1e-7,1-1e-7))),"ece":ece(y,p),"tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}


def selective_prediction_table(frame: pd.DataFrame, coverages=(1.0,.95,.90,.80)) -> pd.DataFrame:
    rows=[]
    for (model,seed,fold),g in frame.groupby(["model_key","seed","outer_fold"]):
        confidence=np.maximum(g.prob_calibrated.to_numpy(),1-g.prob_calibrated.to_numpy())
        order=np.argsort(-confidence)
        for coverage in coverages:
            k=max(1,int(np.ceil(len(g)*coverage))); keep=order[:k]
            y=g.label.to_numpy()[keep]; p=g.prob_calibrated.to_numpy()[keep]; pred=g.pred.to_numpy()[keep]
            rows.append({"model_key":model,"seed":int(seed),"outer_fold":int(fold),"coverage":float(coverage),**metric_dict(y,p,pred),"referred":int(len(g)-k)})
    return pd.DataFrame(rows)


def error_audit(frame: pd.DataFrame) -> pd.DataFrame:
    x=frame.copy(); x["error_type"]=np.where((x.label==1)&(x.pred==0),"FN",np.where((x.label==0)&(x.pred==1),"FP","correct")); x["confidence"]=np.maximum(x.prob_calibrated,1-x.prob_calibrated); x["high_confidence_error"]=(x.error_type!="correct")&(x.confidence>=.90)
    return x.sort_values(["error_type","confidence"],ascending=[True,False])


def paired_group_bootstrap(frame: pd.DataFrame, model_a: str, model_b: str, reps: int = 1000, seed: int = 2026) -> dict[str, Any]:
    rng=np.random.default_rng(seed); merged=[]
    keys=["seed","outer_fold","image_id","group_id","label"]
    a=frame[frame.model_key==model_a][keys+["prob_calibrated","pred"]].rename(columns={"prob_calibrated":"pa","pred":"pda"})
    b=frame[frame.model_key==model_b][keys+["prob_calibrated","pred"]].rename(columns={"prob_calibrated":"pb","pred":"pdb"})
    m=a.merge(b,on=keys,how="inner"); groups=m.group_id.unique(); metrics=[]
    for _ in range(reps):
        sampled=rng.choice(groups,size=len(groups),replace=True); parts=[m[m.group_id==g] for g in sampled]; s=pd.concat(parts,ignore_index=True)
        ma=metric_dict(s.label,s.pa,s.pda); mb=metric_dict(s.label,s.pb,s.pdb)
        metrics.append({k:ma[k]-mb[k] for k in ["balanced_accuracy","sensitivity","roc_auc","brier","ece"]})
    out={"model_a":model_a,"model_b":model_b,"reps":reps,"n_pairs":int(len(m)),"metrics":{}}
    d=pd.DataFrame(metrics)
    for c in d: out["metrics"][c]={"mean_delta":float(d[c].mean()),"ci95":[float(d[c].quantile(.025)),float(d[c].quantile(.975))]}
    return out


def build_reports(run_root: str | Path) -> dict[str, str]:
    root=Path(run_root); pred=pd.read_csv(root/"tables"/"all_oof_predictions.csv")
    selective=selective_prediction_table(pred); audit=error_audit(pred)
    selective.to_csv(root/"tables"/"selective_prediction.csv",index=False); audit.to_csv(root/"tables"/"error_audit.csv",index=False)
    comparisons={}
    for baseline in [m for m in sorted(pred.model_key.unique()) if m!="convnextv2_tiny"]:
        comparisons[baseline]=paired_group_bootstrap(pred,"convnextv2_tiny",baseline)
    (root/"tables"/"paired_bootstrap.json").write_text(json.dumps(comparisons,indent=2),encoding="utf-8")
    return {"selective_prediction":str(root/"tables"/"selective_prediction.csv"),"error_audit":str(root/"tables"/"error_audit.csv"),"paired_bootstrap":str(root/"tables"/"paired_bootstrap.json")}
