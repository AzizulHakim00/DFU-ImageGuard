from __future__ import annotations
import math
from typing import Iterable
import numpy as np
import pandas as pd

REQ={"image_id","group_id","label","model","outer_fold","prob_raw","prob_calibrated","threshold","pred_calibrated"}

def finite(x:Iterable[float],name:str)->np.ndarray:
    a=np.asarray(x,dtype=float).reshape(-1)
    if not a.size or not np.isfinite(a).all(): raise ValueError(f"Invalid {name}")
    return a

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8); return np.log(p/(1-p))

def sigmoid(z):
    z=np.clip(np.asarray(z,dtype=float),-40,40); return 1/(1+np.exp(-z))

def validate(df:pd.DataFrame)->None:
    miss=REQ-set(df.columns)
    if miss: raise ValueError(f"Missing OOF columns: {sorted(miss)}")
    if df.empty or df.duplicated(["model","image_id"]).any(): raise ValueError("Invalid OOF rows")
    ref=None
    for name,f in df.groupby("model",sort=False):
        cur=f[["image_id","group_id","label","outer_fold"]].sort_values("image_id").reset_index(drop=True)
        if ref is None: ref=cur
        elif not cur.equals(ref): raise AssertionError(f"Locked assignments differ for {name}")

def add_threshold_columns(df:pd.DataFrame)->pd.DataFrame:
    validate(df); out=df.copy(); p=np.clip(finite(out.prob_calibrated,"probabilities"),0,1); t=np.clip(finite(out.threshold,"thresholds"),1e-8,1-1e-8)
    out["pred_calibrated"]=(p>=t).astype(int); out["correct_calibrated"]=(out.pred_calibrated.to_numpy()==out.label.to_numpy()).astype(int)
    margin=logit(p)-logit(t); out["probability_confidence"]=np.maximum(p,1-p); out["signed_threshold_logit_margin"]=margin
    out["absolute_threshold_logit_margin"]=np.abs(margin); out["decision_confidence"]=2*sigmoid(np.abs(margin))-1; out["decision_uncertainty"]=1-out.decision_confidence
    out["probability_distance_to_threshold"]=np.abs(p-t); q=np.clip(p,1e-8,1-1e-8); out["predictive_entropy"]=-(q*np.log(q)+(1-q)*np.log(1-q)); out["top_two_margin"]=np.abs(2*p-1)
    return out

def ece(y,p,bins=15):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),0,1); edges=np.linspace(0,1,bins+1); value=0.; maximum=0.
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&(p<(hi if hi<1 else hi+1e-12))
        if m.any():
            gap=abs(float(y[m].mean()-p[m].mean())); value+=gap*float(m.mean()); maximum=max(maximum,gap)
    return float(value),float(maximum)

def auc(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); np1=int(y.sum()); nn=len(y)-np1
    if not np1 or not nn:return float("nan")
    order=np.argsort(p,kind="mergesort"); sp=p[order]; ranks=np.empty(len(p),float); i=0
    while i<len(p):
        j=i+1
        while j<len(p) and sp[j]==sp[i]:j+=1
        ranks[order[i:j]]=0.5*((i+1)+j); i=j
    return float((ranks[y==1].sum()-np1*(np1+1)/2)/(np1*nn))

def ap(y,p):
    y=np.asarray(y,dtype=int); n=int(y.sum())
    if not n:return float("nan")
    sy=y[np.argsort(-np.asarray(p,dtype=float),kind="mergesort")]; precision=np.cumsum(sy)/np.arange(1,len(sy)+1)
    return float(precision[sy==1].sum()/n)

def metrics(y,p,pred,threshold=None,full=True)->dict[str,float]:
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),0,1); pred=np.asarray(pred,dtype=int)
    tp=int(np.sum((y==1)&(pred==1)));tn=int(np.sum((y==0)&(pred==0)));fp=int(np.sum((y==0)&(pred==1)));fn=int(np.sum((y==1)&(pred==0)))
    sens=tp/max(tp+fn,1);spec=tn/max(tn+fp,1);acc=(tp+tn)/max(len(y),1);prec=tp/max(tp+fp,1);npv=tn/max(tn+fn,1);f1=2*tp/max(2*tp+fp+fn,1);f2=5*tp/max(5*tp+4*fn+fp,1)
    den=math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn),0));mcc=((tp*tn)-(fp*fn))/den if den else 0.; obs=float(y.mean());prd=float(pred.mean());pe=obs*prd+(1-obs)*(1-prd);kappa=(acc-pe)/(1-pe) if 1-pe>1e-12 else float("nan")
    clip=np.clip(p,1e-7,1-1e-7); ec,mc=ece(y,p)
    out={"threshold":float(threshold) if threshold is not None else float("nan"),"accuracy":acc,"error_rate":1-acc,"balanced_accuracy":.5*(sens+spec),"precision_ppv":prec,"recall_sensitivity":sens,"specificity":spec,"npv":npv,"f1":f1,"f2":f2,"mcc":mcc,"cohen_kappa":kappa,"roc_auc":auc(y,p),"pr_auc":ap(y,p),"log_loss":float(-np.mean(y*np.log(clip)+(1-y)*np.log(1-clip))),"brier_score":float(np.mean((p-y)**2)),"ece":ec,"mce":mc,"fpr":fp/max(fp+tn,1),"fnr":fn/max(fn+tp,1),"tp":tp,"tn":tn,"fp":fp,"fn":fn,"n":len(y)}
    if full:
        try:
            from sklearn.linear_model import LogisticRegression
            lp=logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1); lr=LogisticRegression(C=1e6,max_iter=3000).fit(lp,y);out["calibration_slope"]=float(lr.coef_[0,0]);out["calibration_intercept"]=float(lr.intercept_[0])
        except Exception:out["calibration_slope"]=out["calibration_intercept"]=float("nan")
    return out

def metric_tables(df):
    rows=[];folds=[];classes=[]
    for name,f in df.groupby("model",sort=True):
        rows.append({"model":name,"state":"raw",**metrics(f.label,f.prob_raw,(f.prob_raw>=.5).astype(int),.5)})
        m=metrics(f.label,f.prob_calibrated,f.pred_calibrated);rows.append({"model":name,"state":"calibrated_fold_specific_thresholds","fold_thresholds":str({int(k)+1:float(v) for k,v in f.groupby("outer_fold").threshold.first().items()}),**m})
        for fold,g in f.groupby("outer_fold",sort=True):
            fm=metrics(g.label,g.prob_calibrated,g.pred_calibrated,float(g.threshold.iloc[0]));folds.append({"model":name,"fold":int(fold)+1,**fm});rows.append({"model":name,"state":f"fold_{int(fold)+1}_calibrated",**fm})
        classes.extend([{"model":name,"class":"Normal","precision":m["npv"],"recall":m["specificity"],"support":m["tn"]+m["fp"]},{"model":name,"class":"DFU","precision":m["precision_ppv"],"recall":m["recall_sensitivity"],"support":m["tp"]+m["fn"]}])
    return pd.DataFrame(rows),pd.DataFrame(folds),pd.DataFrame(classes)

def _indices(frame):
    groups=frame.group_id.unique();arr=frame.group_id.to_numpy();return groups,{g:np.flatnonzero(arr==g) for g in groups}

def group_bootstrap(df,reps=1000,seed=2027):
    rng=np.random.default_rng(seed);names=["accuracy","balanced_accuracy","recall_sensitivity","specificity","f1","mcc","roc_auc","pr_auc","brier_score","log_loss","ece"];rows=[]
    for model,f in df.groupby("model",sort=True):
        f=f.reset_index(drop=True);groups,idx=_indices(f);obs=metrics(f.label,f.prob_calibrated,f.pred_calibrated,full=False);samples={n:[] for n in names}
        for _ in range(reps):
            take=np.concatenate([idx[g] for g in rng.choice(groups,len(groups),replace=True)]);y=f.label.to_numpy()[take]
            if len(np.unique(y))<2:continue
            m=metrics(y,f.prob_calibrated.to_numpy()[take],f.pred_calibrated.to_numpy()[take],full=False)
            for n in names:samples[n].append(m[n])
        for n in names:
            a=np.asarray(samples[n]);rows.append({"model":model,"metric":n,"estimate":obs[n],"ci_low":float(np.percentile(a,2.5)),"ci_high":float(np.percentile(a,97.5)),"bootstrap_repetitions_requested":reps,"bootstrap_repetitions_valid":len(a),"resampling_unit":"duplicate_group"})
    return pd.DataFrame(rows)

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p);out=np.empty_like(p);running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(len(p)-rank)*p[i]));out[i]=running
    return out

def paired(df,primary="DFU-ImageGuard",reps=1000,perms=10000,seed=2028):
    rng=np.random.default_rng(seed);p=df[df.model==primary];names=["accuracy","balanced_accuracy","recall_sensitivity","specificity","f1","mcc","roc_auc","pr_auc","brier_score","ece"];rows=[]
    for base in sorted(set(df.model)-{primary}):
        m=p.merge(df[df.model==base],on=["image_id","group_id","label","outer_fold"],suffixes=("_p","_b"),validate="one_to_one");groups,idx=_indices(m);diff={n:[] for n in names}
        for _ in range(reps):
            take=np.concatenate([idx[g] for g in rng.choice(groups,len(groups),replace=True)]);y=m.label.to_numpy()[take]
            if len(np.unique(y))<2:continue
            a=metrics(y,m.prob_calibrated_p.to_numpy()[take],m.pred_calibrated_p.to_numpy()[take],full=False);b=metrics(y,m.prob_calibrated_b.to_numpy()[take],m.pred_calibrated_b.to_numpy()[take],full=False)
            for n in names:diff[n].append(a[n]-b[n])
        pc=(m.pred_calibrated_p.to_numpy()==m.label.to_numpy()).astype(float);bc=(m.pred_calibrated_b.to_numpy()==m.label.to_numpy()).astype(float);gd=np.array([(pc[i]-bc[i]).mean() for i in idx.values()]);observed=float(gd.mean());perm=1. if np.allclose(gd,0) else float((1+np.sum(np.abs((rng.choice([-1.,1.],size=(perms,len(gd)))*gd).mean(1))>=abs(observed)))/(perms+1))
        for n in names:
            a=np.asarray(diff[n]);rows.append({"comparison":f"{primary} vs {base}","baseline":base,"metric":n,"primary_minus_baseline":float(a.mean()),"difference_ci_low":float(np.percentile(a,2.5)),"difference_ci_high":float(np.percentile(a,97.5)),"paired_group_bootstrap_p_two_sided":float(min(1,2*min(np.mean(a<=0),np.mean(a>=0)))),"group_sign_flip_permutation_p":perm if n=="accuracy" else np.nan,"primary_analysis":"duplicate-group paired bootstrap"})
    out=pd.DataFrame(rows)
    for n,ix in out.groupby("metric").groups.items():out.loc[list(ix),"bootstrap_p_holm"]=holm(out.loc[list(ix),"paired_group_bootstrap_p_two_sided"])
    ix=out.index[out.metric=="accuracy"];out.loc[ix,"group_permutation_p_holm"]=holm(out.loc[ix,"group_sign_flip_permutation_p"])
    return out

def risk(df):
    rows=[];summary=[]
    for name,f in df.groupby("model",sort=True):
        f=f.sort_values("decision_confidence",ascending=False).reset_index(drop=True);mr=[]
        for cov in np.linspace(.05,1,20):
            n=max(1,int(round(len(f)*cov)));g=f.iloc[:n];m=metrics(g.label,g.prob_calibrated,g.pred_calibrated,full=False);r={"model":name,"coverage":n/len(f),"retained_n":n,"referred_n":len(f)-n,"risk":m["error_rate"],"selective_accuracy":m["accuracy"],"sensitivity":m["recall_sensitivity"],"specificity":m["specificity"],"false_negatives":m["fn"],"false_positives":m["fp"]};rows.append(r);mr.append(r)
        z=pd.DataFrame(mr);summary.append({"model":name,"eaurc_threshold_aware":float(np.trapezoid(z.risk,z.coverage)),"risk_at_80pct_coverage":float(z.iloc[(z.coverage-.8).abs().argmin()].risk),"sensitivity_at_80pct_coverage":float(z.iloc[(z.coverage-.8).abs().argmin()].sensitivity)})
    return pd.DataFrame(rows),pd.DataFrame(summary)
