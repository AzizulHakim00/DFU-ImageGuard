from __future__ import annotations

import base64, hashlib, json, os, pickle, shutil, subprocess, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .checkpoint_backup import chunk_file_for_git
from .config_data import Config, assign_duplicate_groups, build_manifest, download_dataset, make_inner_partition, make_outer_folds, seed_everything, sha256_file
from .radial_pilot_core import atomic_csv, atomic_json, train_evaluate_trial


@dataclass
class PilotSettings:
    run_id: str='RADIAL_ADAPTER_PILOT_V1'
    drive_root: str='/content/drive/MyDrive/DFU-ImageGuard'
    secondary_drive_root: str='/content/drive/MyDrive/DFU-ImageGuard-Backup'
    outer_fold: int=0
    seeds: tuple[int,...]=(2026,2027)
    model_kinds: tuple[str,...]=('convnextv2_baseline','dfu_radial_adapter')
    max_epochs: int=25
    patience: int=7
    freeze_epochs: int=2
    batch_size: int=16
    num_workers: int=2
    github_branch: str='radial-pilot-results'
    github_export: bool=True
    github_export_required: bool=True
    github_chunk_full_checkpoints: bool=True
    github_export_after_each_trial: bool=True
    require_secondary_drive_backup: bool=True
    secondary_backup_required: bool=True
    github_chunk_bytes: int=48*1024*1024


def _under(path: Path, parent: Path)->bool:
    try: path.resolve().relative_to(parent.resolve()); return True
    except ValueError: return False


def _preflight(settings: PilotSettings)->Path:
    root=Path(settings.drive_root)
    valid=[Path('/content/drive/MyDrive'),Path('/content/drive/Shareddrives')]
    if not any(p.exists() and _under(root,p) for p in valid): raise RuntimeError(f'Persistent Drive path required: {root}')
    run=root/'runs'/settings.run_id; run.mkdir(parents=True,exist_ok=True)
    token=os.getenv('GITHUB_TOKEN')
    if not token:
        try:
            from google.colab import userdata
            token=userdata.get('GITHUB_TOKEN')
        except Exception: token=None
    if not token: raise RuntimeError('GITHUB_TOKEN missing; no training started.')
    os.environ['GITHUB_TOKEN']=token
    nonce=hashlib.sha256(f'{time.time_ns()}'.encode()).hexdigest(); marker=run/'DRIVE_SENTINEL.json'
    atomic_json(marker,{'nonce':nonce,'run':str(run)})
    if json.loads(marker.read_text())['nonce']!=nonce: raise RuntimeError('Drive write/read verification failed')
    if settings.require_secondary_drive_backup or settings.secondary_backup_required:
        second=Path(settings.secondary_drive_root)/'runs'/settings.run_id; second.mkdir(parents=True,exist_ok=True)
        marker2=second/'SECONDARY_SENTINEL.json'; atomic_json(marker2,{'nonce':nonce})
        if json.loads(marker2.read_text())['nonce']!=nonce: raise RuntimeError('Secondary Drive verification failed')
    print('PERSISTENCE PREFLIGHT: PASS',run)
    return run


def _manifest(root: Path)->dict[str,dict[str,Any]]:
    out={}
    for p in sorted(root.rglob('*')):
        if p.is_file() and '.git' not in p.parts:
            h=hashlib.sha256(p.read_bytes()).hexdigest(); out[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h}
    return out


def mirror_secondary(run: Path, settings: PilotSettings)->dict[str,Any]:
    target=Path(settings.secondary_drive_root)/'runs'/settings.run_id
    if target.exists(): shutil.rmtree(target)
    shutil.copytree(run,target)
    primary=_manifest(run); secondary=_manifest(target)
    if primary!=secondary: raise RuntimeError('Secondary Drive SHA-256 verification failed')
    status={'verified':True,'target':str(target),'files':len(primary)}; atomic_json(run/'SECONDARY_BACKUP_STATUS.json',status); return status


def _git(args:list[str],check=True,**kw):
    return subprocess.run(args,text=True,capture_output=True,check=check,**kw)


def export_github(run: Path, settings: PilotSettings, repository_dir: Path)->dict[str,Any]:
    token=os.environ['GITHUB_TOKEN']; work=Path('/content/DFU-radial-results-export')
    if work.exists(): shutil.rmtree(work)
    _git(['git','clone','https://github.com/AzizulHakim00/DFU-ImageGuard.git',str(work)])
    exists=_git(['git','-C',str(work),'ls-remote','--heads','origin',settings.github_branch],check=False).stdout.strip()
    if exists:
        _git(['git','-C',str(work),'fetch','origin',settings.github_branch])
        _git(['git','-C',str(work),'checkout','-B',settings.github_branch,f'origin/{settings.github_branch}'])
    else: _git(['git','-C',str(work),'checkout','-b',settings.github_branch])
    target=work/'results'/'radial_pilot'/settings.run_id
    if target.exists(): shutil.rmtree(target)
    target.mkdir(parents=True)
    for src in run.rglob('*'):
        if not src.is_file(): continue
        rel=src.relative_to(run)
        if src.suffix=='.pt':
            if settings.github_chunk_full_checkpoints:
                chunk_file_for_git(src,target/(str(rel)+'.chunks'),settings.github_chunk_bytes)
            if 'portable_fp16' not in src.name: continue
        dst=target/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    for rel in ['src/radial_adapter_model.py','src/radial_pilot_core.py','src/radial_pilot_runner.py','src/checkpoint_backup.py']:
        src=repository_dir/rel
        if src.exists():
            dst=target/'source_snapshot'/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    atomic_json(target/'GITHUB_EXPORT_MANIFEST.json',_manifest(target))
    _git(['git','-C',str(work),'config','user.name','DFU Radial Pilot'])
    _git(['git','-C',str(work),'config','user.email','actions@users.noreply.github.com'])
    _git(['git','-C',str(work),'add','--','results/radial_pilot'])
    diff=_git(['git','-C',str(work),'diff','--cached','--quiet'],check=False)
    if diff.returncode==1: _git(['git','-C',str(work),'commit','-m',f'Backup radial pilot {settings.run_id}'])
    raw=base64.b64encode(f'x-access-token:{token}'.encode()).decode()
    push=_git(['git','-C',str(work),'-c',f'http.extraHeader=Authorization: Basic {raw}','push','origin',f'HEAD:{settings.github_branch}'],check=False)
    if push.returncode!=0: raise RuntimeError('GitHub backup failed: '+push.stderr[-500:])
    sha=_git(['git','-C',str(work),'rev-parse','HEAD']).stdout.strip()
    status={'success':True,'branch':settings.github_branch,'commit':sha,'target':str(target.relative_to(work))}
    atomic_json(run/'GITHUB_BACKUP_STATUS.json',status); return status


def _verdict(metrics: pd.DataFrame):
    rows=[]
    for seed in sorted(metrics.seed.unique()):
        b=metrics[(metrics.seed==seed)&(metrics.model_kind=='convnextv2_baseline')].iloc[0]
        a=metrics[(metrics.seed==seed)&(metrics.model_kind=='dfu_radial_adapter')].iloc[0]
        rows.append({'seed':int(seed),'delta_balanced_accuracy':a.balanced_accuracy-b.balanced_accuracy,
                     'delta_sensitivity':a.sensitivity-b.sensitivity,'delta_roc_auc':a.roc_auc-b.roc_auc,
                     'delta_ece':a.ece-b.ece,'gate':a.gate,'mean_abs_adapter_contribution':a.mean_abs_adapter_contribution})
    d=pd.DataFrame(rows); avg=d.mean(numeric_only=True)
    noninferior=(avg.delta_balanced_accuracy>=-.005 and avg.delta_sensitivity>=-.010 and avg.delta_roc_auc>=-.005 and avg.delta_ece<=.010 and abs(avg.gate)>=1e-4 and avg.mean_abs_adapter_contribution>=1e-5)
    benefit=(avg.delta_balanced_accuracy>=.002 or avg.delta_sensitivity>=.005 or avg.delta_ece<=-.005)
    status='STRONG_PASS_ELIGIBLE_FOR_FULL_CV' if noninferior and benefit else ('CONDITIONAL_PASS_NEEDS_ONE_MORE_PILOT_FOLD' if noninferior else 'FAIL_DO_NOT_RUN_FULL_CV')
    return d,{'status':status,'average_deltas':avg.to_dict(),'full_cv_started':False}


def run_radial_adapter_pilot(settings: PilotSettings|None=None, repository_dir: str|Path='/content/DFU-ImageGuard-radial'):
    settings=settings or PilotSettings(); started=time.time(); run=_preflight(settings); repo=Path(repository_dir)
    cfg=Config(); cfg.DRIVE_ROOT=settings.drive_root; cfg.ALLOW_LOCAL_FALLBACK=False; cfg.N_FOLDS=5; cfg.SEED=2026
    cfg.BATCH_SIZE=settings.batch_size; cfg.NUM_WORKERS=settings.num_workers; cfg.MAX_EPOCHS=settings.max_epochs; cfg.PATIENCE=settings.patience
    seed_everything(cfg.SEED)
    dirs={'root':run,**{n:run/n for n in ['tables','figures','models','xai','predictions','logs','configs','manifests','cache']}}
    for p in dirs.values(): Path(p).mkdir(parents=True,exist_ok=True)
    root=download_dataset(cfg,dirs); data=make_outer_folds(assign_duplicate_groups(build_manifest(root,cfg,dirs),cfg,dirs),cfg,dirs)
    train_outer=data[data.outer_fold!=settings.outer_fold].copy(); test=data[data.outer_fold==settings.outer_fold].copy().reset_index(drop=True)
    inner=make_inner_partition(train_outer,cfg,settings.outer_fold); train=inner[inner.inner_role=='train']; select=inner[inner.inner_role=='selection']; cal=inner[inner.inner_role=='calibration']
    atomic_csv(inner,run/'manifests'/'pilot_inner_partition.csv')
    metric_rows=[]; predictions=[]
    for seed in settings.seeds:
        for kind in settings.model_kinds:
            trial=run/'trials'/kind/f'seed_{seed}'/f'fold_{settings.outer_fold+1}'
            def epoch_backup():
                if settings.require_secondary_drive_backup or settings.secondary_backup_required:
                    mirror_secondary(run,settings)
            m,p=train_evaluate_trial(train,select,cal,test,kind,seed,cfg,trial,settings.max_epochs,settings.patience,settings.freeze_epochs,checkpoint_callback=epoch_backup)
            metric_rows.append(m); predictions.append(p)
            if settings.require_secondary_drive_backup or settings.secondary_backup_required:
                mirror_secondary(run,settings)
            if settings.github_export and settings.github_export_after_each_trial:
                export_github(run,settings,repo)
    metrics=pd.DataFrame(metric_rows); all_pred=pd.concat(predictions,ignore_index=True); deltas,verdict=_verdict(metrics)
    atomic_csv(metrics,run/'tables'/'pilot_metrics.csv'); atomic_csv(all_pred,run/'tables'/'all_test_predictions.csv'); atomic_csv(deltas,run/'tables'/'paired_seed_deltas.csv'); atomic_json(run/'PILOT_VERDICT.json',verdict)
    pkl=run/'radial_adapter_pilot_reproducibility.pkl'
    with pkl.open('wb') as f: pickle.dump({'settings':asdict(settings),'metrics':metrics.to_dict('records'),'predictions':all_pred.to_dict('records'),'verdict':verdict},f,pickle.HIGHEST_PROTOCOL)
    atomic_json(run/'ARTIFACT_MANIFEST.json',_manifest(run))
    second=mirror_secondary(run,settings) if (settings.require_secondary_drive_backup or settings.secondary_backup_required) else {'verified':False,'disabled':True}
    github=export_github(run,settings,repo) if settings.github_export else {'success':False,'disabled':True}
    final={'run_id':settings.run_id,'completed_trials':len(metrics),'expected_trials':len(settings.seeds)*len(settings.model_kinds),'verdict':verdict,'primary_drive':str(run),'secondary_backup':second,'github_backup':github,'pkl':str(pkl),'elapsed_minutes':(time.time()-started)/60,'full_cv_started':False}
    atomic_json(run/'FINAL_PILOT_VERIFICATION.json',final)
    if settings.require_secondary_drive_backup or settings.secondary_backup_required: mirror_secondary(run,settings)
    if settings.github_export: export_github(run,settings,repo)
    print(json.dumps(final,indent=2,default=str)); return final


def regenerate_from_saved(run_root: str|Path):
    run=Path(run_root); metrics=pd.read_csv(run/'tables'/'pilot_metrics.csv'); deltas,verdict=_verdict(metrics)
    atomic_csv(deltas,run/'tables'/'paired_seed_deltas.csv'); atomic_json(run/'PILOT_VERDICT.json',verdict)
    return {'run_root':str(run),'metrics':metrics.to_dict('records'),'verdict':verdict}
