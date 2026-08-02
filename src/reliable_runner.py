from __future__ import annotations

import hashlib, json, os, pickle, shutil, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_data import Config, assign_duplicate_groups, build_manifest, download_dataset, make_inner_partition, make_outer_folds, seed_everything, sha256_file
from .evaluation import fit_temperature, select_threshold, sigmoid_np
from .models_training import build_loader, build_transforms
from .reliable_analysis import build_reports, metric_dict
from .reliable_models import build_reliable_model, parameter_summary


@dataclass
class ReliableSettings:
    run_id: str="RELIABLE_DFU_CV_V1"
    drive_root: str="/content/drive/MyDrive/DFU-ImageGuard"
    backup_root: str="/content/drive/MyDrive/DFU-ImageGuard-Backup"
    seeds: tuple[int,...]=(2026,2027,2028)
    folds: tuple[int,...]=(0,1,2,3,4)
    models: tuple[str,...>=("convnextv2_tiny","mobilenetv3_large","densenet121")
    max_epochs: int=30
    patience: int=7
    freeze_epochs: int=2
    batch_size: int=16
    num_workers: int=2
    target_sensitivity: float=.95


def _json(path: Path, payload: Any):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); os.replace(tmp,path)


def _csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); frame.to_csv(tmp,index=False); os.replace(tmp,path)


def _torch(path: Path, payload: dict[str,Any]):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,tmp); os.replace(tmp,path)


def _manifest(root: Path):
    out={}
    for p in sorted(root.rglob("*")):
        if p.is_file(): out[str(p.relative_to(root))]={"bytes":p.stat().st_size,"sha256":sha256_file(p)}
    return out


def _mirror(run: Path, settings: ReliableSettings):
    target=Path(settings.backup_root)/"runs"/settings.run_id
    if target.exists(): shutil.rmtree(target)
    shutil.copytree(run,target)
    if _manifest(run)!=_manifest(target): raise RuntimeError("Secondary backup SHA-256 mismatch")
    _json(run/"SECONDARY_BACKUP_STATUS.json",{"verified":True,"target":str(target),"files":len(_manifest(run))})


def _infer(model, loader, device):
    import torch
    logits=[]; labels=[]; indices=[]
    model.eval()
    with torch.inference_mode():
        for xb,yb,idx in loader:
            out=model(xb.to(device,non_blocking=True)).reshape(-1)
            logits.append(out.float().cpu().numpy()); labels.append(yb.numpy()); indices.append(idx.numpy())
    return np.concatenate(logits),np.concatenate(labels).astype(int),np.concatenate(indices).astype(int)


def _train_trial(train_df,selection_df,cal_df,test_df,model_key,seed,fold,cfg,settings,trial,backup_callback):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    done=trial/"COMPLETE.json"; pred_path=trial/"test_predictions.csv"
    if done.exists() and pred_path.exists(): return json.loads(done.read_text()),pd.read_csv(pred_path)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("GPU runtime required")
    seed_everything(seed); model=build_reliable_model(model_key,pretrained=True).to(device)
    for p in model.parameters(): p.requires_grad=True
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=cfg.WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings.max_epochs)
    scaler=torch.amp.GradScaler("cuda",enabled=cfg.USE_AMP)
    train_tf,eval_tf=build_transforms(cfg)
    tr=build_loader(train_df,train_tf,cfg,True,seed); va=build_loader(selection_df,eval_tf,cfg,False,seed+1)
    pos=float((train_df.label==1).sum()); neg=float((train_df.label==0).sum()); pos_weight=torch.tensor(neg/max(pos,1),device=device)
    last=trial/"last_resume.pt"; best=trial/"best_model.pt"; history=[]; start=1; best_auc=-1.; left=settings.patience
    if last.exists():
        x=torch.load(last,map_location=device,weights_only=False)
        if x.get("model_key")==model_key and x.get("seed")==seed and x.get("fold")==fold:
            model.load_state_dict(x["model"]); optimizer.load_state_dict(x["optimizer"]); scheduler.load_state_dict(x["scheduler"]); scaler.load_state_dict(x["scaler"]); start=int(x["epoch"])+1; best_auc=float(x["best_auc"]); left=int(x["patience_left"])
            if (trial/"history.csv").exists(): history=pd.read_csv(trial/"history.csv").to_dict("records")
    for epoch in range(start,settings.max_epochs+1):
        model.train(); total=0.; n=0
        for xb,yb,_ in tr:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=cfg.USE_AMP): loss=F.binary_cross_entropy_with_logits(model(xb).reshape(-1),yb,pos_weight=pos_weight)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update(); total+=float(loss)*len(yb); n+=len(yb)
        scheduler.step(); vl,vy,_=_infer(model,va,device); auc=float(roc_auc_score(vy,sigmoid_np(vl))); row={"epoch":epoch,"train_loss":total/max(n,1),"selection_auc":auc}; history.append(row); _csv(trial/"history.csv",pd.DataFrame(history)); print(model_key,seed,fold+1,row)
        if auc>best_auc+1e-5:
            best_auc=auc; left=settings.patience; _torch(best,{"model_key":model_key,"seed":seed,"fold":fold,"epoch":epoch,"best_auc":auc,"model":model.state_dict(),"params":parameter_summary(model)})
        else: left-=1
        _torch(last,{"model_key":model_key,"seed":seed,"fold":fold,"epoch":epoch,"best_auc":best_auc,"patience_left":left,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict()}); backup_callback()
        if left<=0: break
    payload=torch.load(best,map_location=device,weights_only=False); model.load_state_dict(payload["model"])
    portable={k:(v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu()) for k,v in model.state_dict().items()}; _torch(trial/"best_model_portable_fp16.pt",{"model_key":model_key,"seed":seed,"fold":fold,"state_dict":portable,"params":payload["params"]})
    cl,cy,_=_infer(model,build_loader(cal_df,eval_tf,cfg,False,seed+300),device); tl,ty,_=_infer(model,build_loader(test_df,eval_tf,cfg,False,seed+400),device)
    temperature=fit_temperature(cl,cy); cp=sigmoid_np(cl/temperature); threshold,rule=select_threshold(cy,cp,settings.target_sensitivity); prob=sigmoid_np(tl/temperature); pred=(prob>=threshold).astype(int)
    frame=test_df.reset_index(drop=True)[["image_id","group_id","label","label_name","relative_path"]].copy(); frame["model_key"]=model_key; frame["seed"]=seed; frame["outer_fold"]=fold+1; frame["logit"]=tl; frame["prob_calibrated"]=prob; frame["pred"]=pred; frame["temperature"]=temperature; frame["threshold"]=threshold
    _csv(pred_path,frame); metrics={**metric_dict(ty,prob,pred),"model_key":model_key,"seed":seed,"outer_fold":fold+1,"temperature":temperature,"threshold":threshold,"threshold_rule":rule,"best_model_sha256":sha256_file(best),"last_resume_sha256":sha256_file(last)}; _json(done,metrics); return metrics,frame


def run_reliable_framework(settings: ReliableSettings|None=None):
    settings=settings or ReliableSettings(); start=time.time(); root=Path(settings.drive_root); run=root/"runs"/settings.run_id; run.mkdir(parents=True,exist_ok=True)
    marker=run/"DRIVE_SENTINEL.txt"; marker.write_text(str(time.time_ns())); assert marker.read_text(); backup=Path(settings.backup_root)/"runs"/settings.run_id; backup.mkdir(parents=True,exist_ok=True)
    cfg=Config(); cfg.DRIVE_ROOT=settings.drive_root; cfg.ALLOW_LOCAL_FALLBACK=False; cfg.N_FOLDS=5; cfg.SEED=2026; cfg.BATCH_SIZE=settings.batch_size; cfg.NUM_WORKERS=settings.num_workers; cfg.MAX_EPOCHS=settings.max_epochs; cfg.PATIENCE=settings.patience; cfg.TARGET_SENSITIVITY=settings.target_sensitivity
    dirs={"root":run,**{n:run/n for n in ["tables","figures","models","xai","predictions","logs","configs","manifests","cache"]}}
    for p in dirs.values(): Path(p).mkdir(parents=True,exist_ok=True)
    data=make_outer_folds(assign_duplicate_groups(build_manifest(download_dataset(cfg,dirs),cfg,dirs),cfg,dirs),cfg,dirs)
    rows=[]; preds=[]
    for fold in settings.folds:
        outer_train=data[data.outer_fold!=fold].copy(); test=data[data.outer_fold==fold].copy().reset_index(drop=True); inner=make_inner_partition(outer_train,cfg,fold); train=inner[inner.inner_role=="train"]; selection=inner[inner.inner_role=="selection"]; cal=inner[inner.inner_role=="calibration"]
        for seed in settings.seeds:
            for model_key in settings.models:
                trial=run/"trials"/model_key/f"seed_{seed}"/f"fold_{fold+1}"; trial.mkdir(parents=True,exist_ok=True)
                m,p=_train_trial(train,selection,cal,test,model_key,seed,fold,cfg,settings,trial,lambda:_mirror(run,settings)); rows.append(m); preds.append(p); _mirror(run,settings)
                _csv(run/"tables"/"fold_seed_metrics.csv",pd.DataFrame(rows)); _csv(run/"tables"/"all_oof_predictions.csv",pd.concat(preds,ignore_index=True))
    metrics=pd.DataFrame(rows); all_pred=pd.concat(preds,ignore_index=True); reports=build_reports(run); summary=metrics.groupby("model_key").agg({"balanced_accuracy":["mean","std"],"sensitivity":["mean","std"],"specificity":["mean","std"],"roc_auc":["mean","std"],"pr_auc":["mean","std"],"brier":["mean","std"],"ece":["mean","std"]}); summary.to_csv(run/"tables"/"model_summary.csv")
    with (run/"reliable_dfu_reproducibility.pkl").open("wb") as f: pickle.dump({"settings":asdict(settings),"metrics":metrics.to_dict("records"),"predictions":all_pred.to_dict("records"),"reports":reports},f,pickle.HIGHEST_PROTOCOL)
    _json(run/"ARTIFACT_MANIFEST.json",_manifest(run)); _mirror(run,settings); final={"run_id":settings.run_id,"completed_trials":len(metrics),"expected_trials":len(settings.folds)*len(settings.seeds)*len(settings.models),"primary_drive":str(run),"secondary_drive":str(backup),"reports":reports,"elapsed_minutes":(time.time()-start)/60}; _json(run/"FINAL_VERIFICATION.json",final); print(json.dumps(final,indent=2)); return final
