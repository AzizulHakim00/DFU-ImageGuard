from __future__ import annotations

import json, os
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .config_data import Config, seed_everything, sha256_file
from .evaluation import ece_mce, fit_temperature, select_threshold, sigmoid_np
from .models_training import build_loader, build_transforms
from .radial_adapter_model import build_model, model_parameter_summary


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch(payload: dict[str, Any], path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _device():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU runtime is required; training did not start.')
    return torch.device('cuda')


def _infer(model, loader, device):
    import torch
    rows = {k: [] for k in ['logit','base_logit','adapter_logit','adapter_contribution','label','index']}
    gate = 0.0
    model.eval()
    with torch.inference_mode():
        for xb, yb, idx in loader:
            out = model(xb.to(device, non_blocking=True), return_aux=True)
            rows['logit'].append(out['logits'].float().cpu().numpy())
            rows['base_logit'].append(out['base_logits'].float().cpu().numpy())
            rows['adapter_logit'].append(out['adapter_logit'].float().cpu().numpy())
            rows['adapter_contribution'].append(out['adapter_contribution'].float().cpu().numpy())
            rows['label'].append(yb.numpy()); rows['index'].append(idx.numpy())
            gate = float(out['gate'].float().cpu())
    result = {k: np.concatenate(v).reshape(-1) for k,v in rows.items()}
    result['label'] = result['label'].astype(int); result['index'] = result['index'].astype(int)
    result['gate'] = gate
    return result


def _metrics(y, p, pred):
    from sklearn.metrics import (accuracy_score, average_precision_score,
        balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
        log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
    y=np.asarray(y,int); p=np.asarray(p,float); pred=np.asarray(pred,int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); ece,mce=ece_mce(y,p,15)
    return {
        'n':len(y),'accuracy':accuracy_score(y,pred),
        'balanced_accuracy':balanced_accuracy_score(y,pred),
        'sensitivity':recall_score(y,pred,zero_division=0),
        'specificity':tn/max(tn+fp,1),'precision':precision_score(y,pred,zero_division=0),
        'f1':f1_score(y,pred,zero_division=0),'mcc':matthews_corrcoef(y,pred),
        'roc_auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),
        'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-7,1-1e-7)),
        'ece':ece,'mce':mce,'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn)
    }


def _optimizer(model, cfg: Config):
    import torch
    groups={'backbone':[],'head':[],'adapter':[],'gate':[]}
    for name,p in model.named_parameters():
        if name.endswith('adapter.alpha'): groups['gate'].append(p)
        elif 'adapter' in name: groups['adapter'].append(p)
        elif 'backbone' in name: groups['backbone'].append(p)
        else: groups['head'].append(p)
    rates={'backbone':1e-5,'head':1e-4,'adapter':2e-4,'gate':1e-3}
    params=[{'params':v,'lr':rates[k],'name':k} for k,v in groups.items() if v]
    return torch.optim.AdamW(params, weight_decay=cfg.WEIGHT_DECAY)


def train_evaluate_trial(train_df, selection_df, calibration_df, test_df,
                         model_kind: str, seed: int, cfg: Config,
                         trial_dir: Path, max_epochs: int=25,
                         patience: int=7, freeze_epochs: int=2, checkpoint_callback=None):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    trial_dir.mkdir(parents=True, exist_ok=True)
    done=trial_dir/'COMPLETE.json'; pred_path=trial_dir/'test_predictions.csv'
    if done.exists() and pred_path.exists():
        return json.loads(done.read_text()), pd.read_csv(pred_path)

    seed_everything(seed); device=_device()
    model=build_model(model_kind,pretrained=True).to(device)
    for p in model.backbone.parameters(): p.requires_grad=False
    optimizer=_optimizer(model,cfg)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs)
    amp=bool(cfg.USE_AMP); scaler=torch.amp.GradScaler('cuda',enabled=amp)
    train_tf,eval_tf=build_transforms(cfg)
    tr=build_loader(train_df,train_tf,cfg,True,seed)
    va=build_loader(selection_df,eval_tf,cfg,False,seed+1)
    positives=float((train_df.label==1).sum()); negatives=float((train_df.label==0).sum())
    pos_weight=torch.tensor(negatives/max(positives,1),device=device)
    last=trial_dir/'last_resume.pt'; best=trial_dir/'best_model.pt'; history=[]
    start=1; best_auc=-1.0; left=patience
    if last.exists():
        payload=torch.load(last,map_location=device,weights_only=False)
        if payload.get('model_kind')==model_kind and payload.get('seed')==seed:
            model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
            scheduler.load_state_dict(payload['scheduler']); scaler.load_state_dict(payload['scaler'])
            start=int(payload['epoch'])+1; best_auc=float(payload['best_auc']); left=int(payload['patience_left'])
            if (trial_dir/'history.csv').exists(): history=pd.read_csv(trial_dir/'history.csv').to_dict('records')
    if start>freeze_epochs:
        for p in model.backbone.parameters(): p.requires_grad=True

    for epoch in range(start,max_epochs+1):
        if epoch==freeze_epochs+1:
            for p in model.backbone.parameters(): p.requires_grad=True
        model.train(); total=0.; n=0
        for xb,yb,_ in tr:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=amp):
                logits=model(xb); loss=F.binary_cross_entropy_with_logits(logits,yb,pos_weight=pos_weight)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update(); total+=float(loss)*len(yb); n+=len(yb)
        scheduler.step(); val=_infer(model,va,device)
        auc=float(roc_auc_score(val['label'],sigmoid_np(val['logit'])))
        row={'epoch':epoch,'train_loss':total/max(n,1),'selection_auc':auc,'gate':val['gate']}
        history.append(row); atomic_csv(pd.DataFrame(history),trial_dir/'history.csv'); print(pd.DataFrame([row]).round(6).to_string(index=False))
        improved=auc>best_auc+1e-5
        if improved:
            best_auc=auc; left=patience
            atomic_torch({'model_kind':model_kind,'seed':seed,'epoch':epoch,'best_auc':auc,
                          'model':model.state_dict(),'params':model_parameter_summary(model)},best)
        else: left-=1
        atomic_torch({'model_kind':model_kind,'seed':seed,'epoch':epoch,'best_auc':best_auc,
                      'patience_left':left,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                      'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict()},last)
        if checkpoint_callback is not None:
            checkpoint_callback()
        if left<=0: break

    payload=torch.load(best,map_location=device,weights_only=False); model.load_state_dict(payload['model']); model.eval()
    portable={k:(v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu()) for k,v in model.state_dict().items()}
    atomic_torch({'model_kind':model_kind,'seed':seed,'state_dict':portable,'params':payload['params']},trial_dir/'best_model_portable_fp16.pt')
    ca=_infer(model,build_loader(calibration_df,eval_tf,cfg,False,seed+300),device)
    te=_infer(model,build_loader(test_df,eval_tf,cfg,False,seed+400),device)
    temperature=fit_temperature(ca['logit'],ca['label']); cp=sigmoid_np(ca['logit']/temperature)
    threshold,rule=select_threshold(ca['label'],cp,cfg.TARGET_SENSITIVITY)
    raw=sigmoid_np(te['logit']); prob=sigmoid_np(te['logit']/temperature); pred=(prob>=threshold).astype(int)
    frame=test_df.reset_index(drop=True)[['image_id','group_id','label','label_name','relative_path']].copy()
    frame['model_kind']=model_kind; frame['seed']=seed; frame['logit']=te['logit']; frame['prob_raw']=raw
    frame['prob_calibrated']=prob; frame['pred']=pred; frame['temperature']=temperature; frame['threshold']=threshold
    frame['base_logit']=te['base_logit']; frame['adapter_logit']=te['adapter_logit']; frame['adapter_contribution']=te['adapter_contribution']
    atomic_csv(frame,pred_path)
    raw_metrics=_metrics(te['label'],raw,(raw>=.5).astype(int)); metrics=_metrics(te['label'],prob,pred)
    metrics.update({'model_kind':model_kind,'seed':seed,'temperature':temperature,'threshold':threshold,
                    'threshold_rule':rule,'gate':te['gate'],
                    'mean_abs_adapter_contribution':float(np.mean(np.abs(te['adapter_contribution']))),
                    'raw_brier':raw_metrics['brier'],'raw_ece':raw_metrics['ece'],'raw_log_loss':raw_metrics['log_loss'],
                    'best_model_sha256':sha256_file(best),'last_resume_sha256':sha256_file(last)})
    atomic_json(done,metrics); atomic_json(trial_dir/'metrics.json',metrics)
    return metrics,frame
