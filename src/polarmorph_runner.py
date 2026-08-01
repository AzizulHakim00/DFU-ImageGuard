from __future__ import annotations

import gc
import json
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .artifacts import artifact_manifest, push_to_github, software_hardware_versions
from .config_data import Config, assign_duplicate_groups, build_manifest, download_dataset, make_inner_partition, make_outer_folds, mount_drive, now_run_id, prepare_run_dirs, seed_everything, sha256_file, write_json
from .evaluation import create_prediction_frame, fit_temperature, metric_dict, select_threshold
from .models_training import build_loader, build_transforms
from .polarmorph_model import build_polarmorph_model


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_checkpoint(model, cfg, fold, epoch, auc, path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"architecture":"DFU-PolarMorphNet","version":1,"fold":fold+1,"epoch":epoch,"auc":auc,"config":asdict(cfg),"model_state_dict":model.state_dict()}, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path, cfg):
    import torch
    device = _device()
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("architecture") != "DFU-PolarMorphNet":
        raise RuntimeError(f"Invalid architecture tag in {path}")
    model = build_polarmorph_model(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _infer(model, loader, auxiliary=False):
    import torch
    logits, labels, indexes, aux = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for xb, yb, idx in loader:
            out = model(xb.to(_device()), return_aux=auxiliary)
            z = out["logits"] if auxiliary else out
            logits.append(z.detach().cpu().numpy().reshape(-1)); labels.append(yb.numpy()); indexes.append(idx.numpy())
            if auxiliary:
                for j in range(len(yb)):
                    aux.append({"loader_index":int(idx[j]),"center_x":float(out["center"][j,0].cpu()),"center_y":float(out["center"][j,1].cpu()),"lesion_scale":float(out["scale"][j].cpu()),"fusion_radial":float(out["fusion_weights"][j,0].cpu()),"fusion_contour":float(out["fusion_weights"][j,1].cpu()),"fusion_global":float(out["fusion_weights"][j,2].cpu())})
    return np.concatenate(logits), np.concatenate(labels).astype(int), np.concatenate(indexes), pd.DataFrame(aux)


def _train(model, train_df, val_df, cfg, fold, checkpoint, history_path):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    seed_everything(int(cfg.SEED + 1009 * fold))
    device = _device(); model = model.to(device)
    train_tf, eval_tf = build_transforms(cfg)
    train_loader = build_loader(train_df, train_tf, cfg, True, int(cfg.SEED + fold))
    val_loader = build_loader(val_df, eval_tf, cfg, False, int(cfg.SEED + fold + 1))
    pos = float((train_df.label==1).sum()); neg = float((train_df.label==0).sum())
    pos_weight = torch.tensor(neg/max(pos,1), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.LEARNING_RATE), weight_decay=float(cfg.WEIGHT_DECAY))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,int(cfg.MAX_EPOCHS)))
    amp = bool(cfg.USE_AMP and device.type=="cuda"); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best=-1.0; patience=int(cfg.PATIENCE); rows=[]
    for epoch in range(1,int(cfg.MAX_EPOCHS)+1):
        model.train(); totals=np.zeros(4); seen=0
        for xb,yb,_ in train_loader:
            xb=xb.to(device); yb=yb.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out=model(xb,return_aux=True)
                cls=F.binary_cross_entropy_with_logits(out["logits"],yb,pos_weight=pos_weight)
                bg=F.binary_cross_entropy_with_logits(out["background_logits"],yb)
                m=out["mask"]; tv=(m[:,:,1:]-m[:,:,:-1]).abs().mean()+(m[:,:,:,1:]-m[:,:,:,:-1]).abs().mean(); area=(m.mean((1,2,3))-.25).abs().mean()
                reg=tv+.25*area; loss=cls+.10*bg+.05*reg
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.GRAD_CLIP_NORM)); scaler.step(optimizer); scaler.update()
            n=len(yb); totals += np.array([float(loss.detach()),float(cls.detach()),float(bg.detach()),float(reg.detach())])*n; seen += n
        scheduler.step(); val_logits,val_y,_,_=_infer(model,val_loader); prob=1/(1+np.exp(-np.clip(val_logits,-30,30))); auc=float(roc_auc_score(val_y,prob))
        row={"fold":fold+1,"epoch":epoch,"total_loss":totals[0]/seen,"classification_loss":totals[1]/seen,"background_loss":totals[2]/seen,"mask_regularization":totals[3]/seen,"selection_auc":auc,"lr":optimizer.param_groups[0]["lr"]}; rows.append(row); pd.DataFrame(rows).to_csv(history_path,index=False)
        print(pd.DataFrame([row]).round(5).to_string(index=False))
        if auc>best+1e-5: best=auc; patience=int(cfg.PATIENCE); _save_checkpoint(model,cfg,fold,epoch,auc,checkpoint)
        else:
            patience-=1
            if patience<=0: break
    return _load_checkpoint(checkpoint,cfg)[0], pd.DataFrame(rows)


def _display_fold(history, metrics, fold, dirs):
    import matplotlib.pyplot as plt
    from IPython.display import display
    display(metrics.round(5))
    if history.empty: return
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(history.epoch,history.total_loss,label="loss"); ax2=ax.twinx(); ax2.plot(history.epoch,history.selection_auc,label="selection AUC"); ax.set(xlabel="epoch",ylabel="loss",title=f"DFU-PolarMorphNet fold {fold+1}"); ax2.set_ylabel("AUC"); fig.savefig(dirs["figures"]/f"live_fold_{fold+1}.png",dpi=300,bbox_inches="tight"); plt.show(); plt.close(fig)


def _xai(model, frame, cfg, fold, dirs):
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    _,tf=build_transforms(cfg); rows=[]
    for _,r in frame.sample(n=min(3,len(frame)),random_state=int(cfg.SEED+fold)).iterrows():
        with Image.open(r.image_path) as im: original=im.convert("RGB").resize((cfg.IMAGE_SIZE,cfg.IMAGE_SIZE))
        with torch.inference_mode(): out=model(tf(original).unsqueeze(0).to(_device()),return_aux=True); p=float(torch.sigmoid(out["logits"])[0]); mask=out["mask"][0,0].cpu().numpy()
        fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(original); ax.imshow(mask,alpha=.45,extent=(0,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,0)); ax.axis("off"); ax.set_title(f"fold {fold+1} | {r.label_name} | p={p:.3f}"); path=dirs["xai"]/f"weak_map_f{fold+1}_{r.image_id}.png"; fig.savefig(path,dpi=300,bbox_inches="tight"); plt.show(); plt.close(fig); rows.append({"image_id":r.image_id,"fold":fold+1,"probability":p,"path":str(path),"method":"weak_lesion_map","clinical_localization_claim":False})
    return rows


def _final_figures(oof, metrics, dirs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
    fig,ax=plt.subplots(figsize=(5,5)); ConfusionMatrixDisplay.from_predictions(oof.label,oof.pred_calibrated,display_labels=["Normal","DFU"],ax=ax); fig.savefig(dirs["figures"]/"oof_confusion_matrix.png",dpi=300,bbox_inches="tight"); fig.savefig(dirs["figures"]/"oof_confusion_matrix.pdf",bbox_inches="tight"); plt.show(); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); RocCurveDisplay.from_predictions(oof.label,oof.prob_calibrated,ax=ax,name="DFU-PolarMorphNet"); fig.savefig(dirs["figures"]/"oof_roc.png",dpi=300,bbox_inches="tight"); plt.show(); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); PrecisionRecallDisplay.from_predictions(oof.label,oof.prob_calibrated,ax=ax,name="DFU-PolarMorphNet"); fig.savefig(dirs["figures"]/"oof_pr.png",dpi=300,bbox_inches="tight"); plt.show(); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); fold=metrics[metrics.scope!="OOF"]; ax.plot(range(1,len(fold)+1),fold.balanced_accuracy,marker="o",label="balanced accuracy"); ax.plot(range(1,len(fold)+1),fold.recall_sensitivity,marker="o",label="sensitivity"); ax.legend(); ax.set(xlabel="fold",ylabel="score",title="Fold-wise live final performance"); fig.savefig(dirs["figures"]/"fold_stability.png",dpi=300,bbox_inches="tight"); plt.show(); plt.close(fig)


def run_polarmorph(overrides: Optional[dict[str,Any]]=None):
    cfg=Config()
    for k,v in (overrides or {}).items():
        if hasattr(cfg,k): setattr(cfg,k,v)
    cfg.PRIMARY_MODEL_NAME="DFU-PolarMorphNet"; cfg.RUN_ID=cfg.RUN_ID or f"POLAR_{now_run_id()}"; mount_drive(cfg); seed_everything(cfg.SEED); dirs=prepare_run_dirs(cfg); started=time.time()
    root=download_dataset(cfg,dirs); manifest=make_outer_folds(assign_duplicate_groups(build_manifest(root,cfg,dirs),cfg,dirs),cfg,dirs); all_predictions=[]; calibration=[]; xai=[]
    for fold in range(cfg.N_FOLDS):
        outer_train=manifest[manifest.outer_fold!=fold].copy(); outer_test=manifest[manifest.outer_fold==fold].copy().reset_index(drop=True); inner=make_inner_partition(outer_train,cfg,fold); train=inner[inner.inner_role=="train"]; val=inner[inner.inner_role=="selection"]; cal=inner[inner.inner_role=="calibration"]
        checkpoint=dirs["models"]/f"dfu_polarmorphnet_fold_{fold+1}.pt"; history_path=dirs["logs"]/f"history_polarmorph_fold_{fold+1}.csv"; trained=False
        if checkpoint.exists() and not cfg.FORCE_RETRAIN: model,_=_load_checkpoint(checkpoint,cfg); history=pd.read_csv(history_path) if history_path.exists() else pd.DataFrame(); print(f"REUSED fold {fold+1}: {checkpoint}")
        else: model,history=_train(build_polarmorph_model(cfg),train,val,cfg,fold,checkpoint,history_path); trained=True
        _,tf=build_transforms(cfg); cal_loader=build_loader(cal,tf,cfg,False,cfg.SEED+300+fold); test_loader=build_loader(outer_test,tf,cfg,False,cfg.SEED+400+fold); cal_logits,cal_y,_,_=_infer(model,cal_loader); temp=fit_temperature(cal_logits,cal_y); cal_prob=1/(1+np.exp(-np.clip(cal_logits/temp,-30,30))); threshold,rule=select_threshold(cal_y,cal_prob,cfg.TARGET_SENSITIVITY); logits,y,_,aux=_infer(model,test_loader,True); pred=create_prediction_frame(outer_test,logits,temp,threshold,cfg.PRIMARY_MODEL_NAME,fold).reset_index(drop=True); pred=pd.concat([pred,aux.drop(columns="loader_index").reset_index(drop=True)],axis=1); pred.to_csv(dirs["predictions"]/f"oof_fold_{fold+1}.csv",index=False); all_predictions.append(pred); pd.concat(all_predictions).to_csv(dirs["predictions"]/"all_oof_predictions.csv",index=False)
        fm=pd.DataFrame([{"scope":f"fold_{fold+1}",**metric_dict(y,pred.prob_calibrated,threshold)}]); fm.to_csv(dirs["tables"]/f"fold_{fold+1}_metrics.csv",index=False); cumulative=pd.concat([pd.read_csv(p) for p in sorted(dirs["tables"].glob("fold_*_metrics.csv"))]); _display_fold(history,cumulative,fold,dirs); xai.extend(_xai(model,outer_test,cfg,fold,dirs)); calibration.append({"fold":fold+1,"temperature":temp,"threshold":threshold,"threshold_rule":rule,"checkpoint":str(checkpoint),"sha256":sha256_file(checkpoint),"trained_now":trained}); write_json(dirs["configs"]/f"fold_{fold+1}_calibration.json",calibration[-1]); del model; gc.collect()
    oof=pd.concat(all_predictions,ignore_index=True); rows=[]
    for fold,f in oof.groupby("outer_fold"): rows.append({"scope":f"fold_{fold+1}",**metric_dict(f.label,f.prob_calibrated,float(f.threshold.iloc[0]))})
    from sklearn.metrics import confusion_matrix
    tn,fp,fn,tp=confusion_matrix(oof.label,oof.pred_calibrated,labels=[0,1]).ravel(); agg=metric_dict(oof.label,oof.prob_calibrated,.5); agg.update({"scope":"OOF","accuracy":float((oof.label==oof.pred_calibrated).mean()),"tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}); rows.append(agg); metrics=pd.DataFrame(rows); metrics.to_csv(dirs["tables"]/"fold_and_oof_metrics.csv",index=False); pd.DataFrame(xai).to_csv(dirs["tables"]/"small_xai_metadata.csv",index=False); _final_figures(oof,metrics,dirs)
    payload={"run_id":cfg.RUN_ID,"architecture":"DFU-PolarMorphNet","config":asdict(cfg),"manifest":manifest.to_dict("records"),"predictions":oof.to_dict("records"),"metrics":metrics.to_dict("records"),"calibration":calibration,"xai":xai,"raw_images_stored":False}; pkl=dirs["root"]/"dfu_polarmorphnet_complete_reproducibility.pkl"; pickle.dump(payload,pkl.open("wb"),protocol=pickle.HIGHEST_PROTOCOL); write_json(dirs["root"]/"software_versions.json",software_hardware_versions()); write_json(dirs["root"]/"manifest.json",artifact_manifest(dirs["root"])); github=push_to_github(dirs["root"],cfg,dirs); verification={"run_id":cfg.RUN_ID,"valid_folds":int(oof.outer_fold.nunique()),"valid_checkpoints":sum(Path(c["checkpoint"]).exists() for c in calibration),"fits_executed_now":sum(c["trained_now"] for c in calibration),"pkl_path":str(pkl),"drive_path":str(dirs["root"]),"github":github,"elapsed_minutes":(time.time()-started)/60}; write_json(dirs["root"]/"final_verification.json",verification); print(json.dumps(verification,indent=2,default=str)); return verification


def regenerate_from_saved(run_root: str|Path):
    run_root=Path(run_root); predictions=pd.read_csv(run_root/"predictions"/"all_oof_predictions.csv"); metrics=pd.read_csv(run_root/"tables"/"fold_and_oof_metrics.csv"); dirs={"figures":run_root/"figures"}; dirs["figures"].mkdir(exist_ok=True); _final_figures(predictions,metrics,dirs); print("Figures regenerated without training."); return {"run_root":str(run_root),"rows":len(predictions)}
