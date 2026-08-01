from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .q1_metrics import add_threshold_columns,group_bootstrap,metric_tables,metrics,paired,risk

def dump(path:Path,data:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)),encoding="utf-8")

def uncertainty(df,risk_summary):
    from sklearn.metrics import roc_auc_score
    lookup=risk_summary.set_index("model").to_dict("index");rows=[]
    for name,f in df.groupby("model",sort=True):
        err=(f.correct_calibrated==0).astype(int);valid=err.nunique()==2
        rows.append({"model":name,"error_detection_auroc_entropy":float(roc_auc_score(err,f.predictive_entropy)) if valid else np.nan,"error_detection_auroc_threshold_aware":float(roc_auc_score(err,f.decision_uncertainty)) if valid else np.nan,"mean_probability_confidence":float(f.probability_confidence.mean()),"mean_decision_confidence":float(f.decision_confidence.mean()),"high_decision_confidence_errors_ge_0_80":int(((err==1)&(f.decision_confidence>=.8)).sum()),"total_errors":int(err.sum()),**lookup.get(name,{})})
    return pd.DataFrame(rows)

def errors(df,primary):
    f=df[df.model==primary].copy();f["error_type"]=np.select([(f.label==1)&(f.pred_calibrated==0),(f.label==0)&(f.pred_calibrated==1)],["false_negative","false_positive"],default="correct");f["high_decision_confidence_error"]=(f.error_type!="correct")&(f.decision_confidence>=.8);f["uncertain_prediction_top_decile"]=f.decision_uncertainty>=f.decision_uncertainty.quantile(.9);f["low_decision_confidence_correct"]=(f.error_type=="correct")&(f.decision_confidence<=f.decision_confidence.quantile(.1));return f

def calibration(df):
    rows=[]
    for name,f in df.groupby("model",sort=True):
        a=metrics(f.label,f.prob_raw,(f.prob_raw>=.5).astype(int),.5);b=metrics(f.label,f.prob_calibrated,f.pred_calibrated)
        rows.append({"model":name,"raw_ece":a["ece"],"calibrated_ece":b["ece"],"ece_change_calibrated_minus_raw":b["ece"]-a["ece"],"raw_brier":a["brier_score"],"calibrated_brier":b["brier_score"],"brier_change_calibrated_minus_raw":b["brier_score"]-a["brier_score"],"raw_log_loss":a["log_loss"],"calibrated_log_loss":b["log_loss"],"log_loss_change_calibrated_minus_raw":b["log_loss"]-a["log_loss"],"calibration_improved_ece":b["ece"]<a["ece"],"calibration_improved_brier":b["brier_score"]<a["brier_score"]})
    return pd.DataFrame(rows)

def readiness(metric_table,comparisons,primary):
    c=metric_table[metric_table.state=="calibrated_fold_specific_thresholds"].sort_values(["balanced_accuracy","roc_auc"],ascending=False).reset_index(drop=True);p=c[c.model==primary].iloc[0];rank=int(c.index[c.model==primary][0])+1;acc=comparisons[comparisons.metric=="accuracy"];superior=bool(not acc.empty and (acc.primary_minus_baseline>0).all() and (acc.difference_ci_low>0).all() and (acc.group_permutation_p_holm<.05).all())
    return {"primary_model":primary,"primary_rank_by_balanced_accuracy":rank,"best_model_by_balanced_accuracy":str(c.iloc[0].model),"primary_balanced_accuracy":float(p.balanced_accuracy),"primary_sensitivity":float(p.recall_sensitivity),"primary_specificity":float(p.specificity),"primary_roc_auc":float(p.roc_auc),"primary_false_negatives":int(p.fn),"superiority_supported_against_all_baselines":superior,"external_validation_performed":False,"multi_seed_performed":False,"q1_high_impact_ready":False,"mandatory_remaining_evidence":["genuine independent external validation with overlap audit","pre-registered five-seed final comparison","visual review of XAI overlays","independent error/label review","dataset licensing clarification"],"allowed_claim":"Competitive duplicate-group-aware OOF discrimination with high specificity; no statistically supported superiority or deployment claim.","blocked_claims":["state of the art","significantly outperforms all baselines","patient-level validation","external generalisation","clinical deployment readiness","XAI proves clinical correctness"]}

def savefigs(df,metric_table,folds,rc,out):
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True);cal=metric_table[metric_table.state=="calibrated_fold_specific_thresholds"]
    fig,ax=plt.subplots(figsize=(9,5))
    for name,f in folds.groupby("model"):ax.plot(f.fold,f.balanced_accuracy,marker="o",label=name)
    ax.set(xlabel="Outer fold",ylabel="Balanced accuracy",title="Corrected five-fold stability");ax.set_xticks(range(1,6));ax.legend(fontsize=7,ncol=2);_save(fig,out/"q1_corrected_fold_stability");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for name,f in rc.groupby("model"):ax.plot(f.coverage,f.risk,label=name)
    ax.set(xlabel="Coverage",ylabel="Risk",title="Threshold-aware risk–coverage");ax.legend(fontsize=7);_save(fig,out/"q1_threshold_aware_risk_coverage");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for name,f in rc.groupby("model"):ax.plot(f.coverage,f.sensitivity,label=name)
    ax.set(xlabel="Coverage",ylabel="Sensitivity",title="Sensitivity–coverage");ax.legend(fontsize=7);_save(fig,out/"q1_sensitivity_coverage");plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));cal.set_index("model")[["balanced_accuracy","recall_sensitivity","specificity","f1","mcc","roc_auc","pr_auc"]].plot(kind="bar",ax=ax);ax.set_ylim(.85,1.01);ax.set_title("Corrected calibrated OOF comparison");ax.legend(fontsize=7,ncol=2);_save(fig,out/"q1_corrected_metric_comparison");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for name,f in df.groupby("model"):ax.hist(f.decision_confidence,bins=20,alpha=.3,density=True,label=name)
    ax.set(xlabel="Threshold-aware decision confidence",ylabel="Density",title="Decision-confidence distribution");ax.legend(fontsize=7);_save(fig,out/"q1_decision_confidence_histogram");plt.close(fig)

def _save(fig,path):
    fig.savefig(path.with_suffix(".png"),dpi=300,bbox_inches="tight");fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")

def report(path,ready,metric_table,comparisons):
    cal=metric_table[metric_table.state=="calibrated_fold_specific_thresholds"][["model","accuracy","balanced_accuracy","recall_sensitivity","specificity","f1","mcc","roc_auc","pr_auc","brier_score","ece","fn","fp"]].sort_values("balanced_accuracy",ascending=False);acc=comparisons[comparisons.metric=="accuracy"][["comparison","primary_minus_baseline","difference_ci_low","difference_ci_high","group_sign_flip_permutation_p","group_permutation_p_holm"]]
    text=f"# DFU-ImageGuard Q1 Post-hoc Report\n\nQ1 high-impact ready: **{ready['q1_high_impact_ready']}**\n\nPrimary rank: **{ready['primary_rank_by_balanced_accuracy']}**; nominal best model: **{ready['best_model_by_balanced_accuracy']}**.\n\nSupported claim: {ready['allowed_claim']}\n\n## Corrected metrics\n\n{cal.to_markdown(index=False,floatfmt='.4f')}\n\n## Group-aware accuracy comparisons\n\n{acc.to_markdown(index=False,floatfmt='.4f')}\n\n## Mandatory remaining evidence\n\n"+"\n".join(f"- {x}" for x in ready["mandatory_remaining_evidence"])+"\n\n## Blocked claims\n\n"+"\n".join(f"- {x}" for x in ready["blocked_claims"])+"\n"
    path.write_text(text,encoding="utf-8")

def run_q1_posthoc_corrections(run_root,primary_model="DFU-ImageGuard",bootstrap_repetitions=1000,permutation_repetitions=10000,seed=2026,predictions=None,overwrite_primary_tables=False):
    root=Path(run_root);tables=root/"tables";pred_dir=root/"predictions";fig_dir=root/"figures";q=root/"q1_corrected";qt=q/"tables";qf=q/"figures";qt.mkdir(parents=True,exist_ok=True);qf.mkdir(parents=True,exist_ok=True)
    if predictions is None:predictions=pd.read_csv(pred_dir/"all_oof_predictions.csv")
    predictions=add_threshold_columns(predictions);mt,folds,classes=metric_tables(predictions);ci=group_bootstrap(predictions,bootstrap_repetitions,seed+1);comp=paired(predictions,primary_model,bootstrap_repetitions,permutation_repetitions,seed+2);rc,rs=risk(predictions);unc=uncertainty(predictions,rs);err=errors(predictions,primary_model);cal=calibration(predictions);ready=readiness(mt,comp,primary_model)
    frames={"q1_all_oof_predictions_threshold_aware.csv":predictions,"q1_corrected_metrics.csv":mt,"q1_corrected_fold_stability.csv":folds,"q1_per_class_metrics.csv":classes,"q1_group_bootstrap_95ci.csv":ci,"q1_paired_group_comparisons.csv":comp,"q1_threshold_aware_risk_coverage.csv":rc,"q1_risk_coverage_summary.csv":rs,"q1_uncertainty_summary.csv":unc,"q1_error_analysis.csv":err,"q1_calibration_before_after.csv":cal}
    for name,f in frames.items():f.to_csv(qt/name,index=False)
    paper={"reporting_version":"q1-posthoc-v1","primary_model":primary_model,"metrics":mt.to_dict("records"),"confidence_intervals":ci.to_dict("records"),"paired_group_comparisons":comp.to_dict("records"),"threshold_aware_uncertainty":unc.to_dict("records"),"calibration_before_after":cal.to_dict("records"),"q1_readiness":ready,"retraining_performed":False};dump(q/"q1_readiness.json",ready);dump(q/"paper_results_q1_corrected.json",paper);dump(root/"paper_results_q1_corrected.json",paper);report(q/"Q1_POSTHOC_REPORT.md",ready,mt,comp);report(root/"Q1_POSTHOC_REPORT.md",ready,mt,comp);savefigs(predictions,mt,folds,rc,qf)
    if overwrite_primary_tables:
        predictions.to_csv(pred_dir/"all_oof_predictions.csv",index=False);predictions[predictions.model==primary_model].to_csv(pred_dir/"dfu_imageguard_oof_predictions.csv",index=False);mt.to_csv(tables/"all_metrics_raw_and_calibrated.csv",index=False);folds.to_csv(tables/"fold_stability.csv",index=False);ci.to_csv(tables/"group_bootstrap_95ci.csv",index=False);comp.to_csv(tables/"statistical_comparisons.csv",index=False);rc.to_csv(tables/"risk_coverage.csv",index=False);unc.to_csv(tables/"uncertainty_summary.csv",index=False);err.to_csv(tables/"error_uncertainty_analysis.csv",index=False);cal.to_csv(tables/"calibration_before_after.csv",index=False);savefigs(predictions,mt,folds,rc,fig_dir)
    return {"predictions":predictions,"metrics":mt,"fold_metrics":folds,"class_metrics":classes,"confidence_intervals":ci,"comparisons":comp,"risk_coverage":rc,"risk_summary":rs,"uncertainty":unc,"errors":err,"calibration":cal,"readiness":ready}
