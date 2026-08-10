from __future__ import annotations
import ast, base64, hashlib, importlib.util, io, json, os, shutil, sys, time, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

VERSION="DFU_REPAIR45_GOOD38_BRIDGE_V3_20260810"
BASE_COMMIT="0413eec1acea851664f52e8af7cc7934182aa24b"
BASE_SHA256="be3de60220b677da3d9bad5d8a06dcbb3e67f498a2fe97972a475a74666cab99"
BASE_URL=f"https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/{BASE_COMMIT}/scripts/dfu_repair45_reconcile_locked_v1.py"
GOOD38_NAME="DFU_Repair7_SELF_CONTAINED_GOOD38_RECOVERY_CPU.ipynb"
GOOD38_SHA="6b0c149456f5b29a4a132534d0a6e71ac5fb31550d2cc4ef056131219e62ea5b"
FOLD_SIZES={1:209,2:206,3:217,4:214,5:209}
HIST_URL="https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/11881ee007149b23e3664298c0d09c1f988acdd1/notebooks/repair7_evidence/mobilenetv3_large_seed2028_fold1_logits.csv"
HIST_SHA="c5f696aa89b0436ad100911fe8d4f478b91b86b94da6d269823d51c8308b4020"

def h(b): return hashlib.sha256(b).hexdigest()

raw=urllib.request.urlopen(BASE_URL,timeout=120).read()
if h(raw)!=BASE_SHA256: raise RuntimeError("Base Repair45 SHA mismatch")
p=Path("/content/dfu_repair45_base_v1_v3.py"); p.write_bytes(raw)
spec=importlib.util.spec_from_file_location("dfu_repair45_base_v1_v3",p)
base=importlib.util.module_from_spec(spec); sys.modules[spec.name]=base; spec.loader.exec_module(base)
base.VERSION=VERSION
ORIG_VALIDATE=base.validate_locked_manifest
STATE={"done":False}

def const_program(code,var):
    value=None
    for n in ast.parse(code).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==var and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
            value=n.value.value
        elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id==var and isinstance(n.op,ast.Add) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
            value=(value or "")+n.value.value
    return value

def parse_notebook_bytes(nbraw,source):
    nb=json.loads(nbraw.decode("utf-8"))
    code="\n".join("".join(c.get("source",[])) for c in nb.get("cells",[]) if c.get("cell_type")=="code")
    if const_program(code,"EVIDENCE_PACKAGE_SHA256")!=GOOD38_SHA:
        raise ValueError("wrong embedded evidence package identifier")
    b64=const_program(code,"_EVIDENCE_B64")
    if not b64: raise ValueError("embedded GOOD38 payload not found")
    pkg=base64.b64decode(b64,validate=True)
    if h(pkg)!=GOOD38_SHA: raise ValueError("embedded GOOD38 package SHA mismatch")
    with zipfile.ZipFile(io.BytesIO(pkg)) as z:
        man=json.loads(z.read("MANIFEST.json").decode())
        pb=z.read("good38_predictions.csv"); mb=z.read("good38_metrics.csv")
    if h(pb)!=man["predictions_sha256"] or h(mb)!=man["metrics_sha256"]:
        raise ValueError("GOOD38 inner SHA mismatch")
    pred=pd.read_csv(io.BytesIO(pb)); met=pd.read_csv(io.BytesIO(mb))
    if len(pred)!=8032 or len(met)!=38: raise ValueError(f"GOOD38 size mismatch {len(pred)}/{len(met)}")
    return {"source":source,"pred":pred,"met":met,"manifest":man}

def find_or_upload_bundle():
    roots=[Path("/content/drive/MyDrive"),Path("/content/drive/Shareddrives"),Path("/content/drive/.shortcut-targets-by-id")]
    roots=[r for r in roots if r.exists()]
    print("Searching whole mounted Drive for",GOOD38_NAME)
    for root in roots:
        for current,dirs,names in os.walk(root,topdown=True,followlinks=False):
            dirs[:]=[d for d in dirs if d not in {".Trash",".cache","__pycache__",".ipynb_checkpoints","node_modules"}]
            if GOOD38_NAME in names:
                q=Path(current)/GOOD38_NAME
                try:
                    b=parse_notebook_bytes(q.read_bytes(),str(q))
                    print("GOOD38 evidence found on Drive:",q); return b
                except Exception as e:
                    print("Rejected candidate:",q,type(e).__name__,e)
    print("GOOD38 notebook was not found on Drive.")
    print("Upload the exact file:",GOOD38_NAME)
    from google.colab import files
    uploaded=files.upload()
    for name,data in uploaded.items():
        try:
            b=parse_notebook_bytes(data,f"manual_upload:{name}")
            print("GOOD38 uploaded evidence: PASS"); return b
        except Exception as e:
            print("Rejected upload:",name,type(e).__name__,e)
    raise RuntimeError("Valid GOOD38 evidence was not supplied. GPU remains blocked.")

def lock_from_good38(bundle):
    x=bundle["pred"].copy()
    need={"image_id","group_id","label","model_key","seed","outer_fold","prob_calibrated","pred"}
    if not need.issubset(x.columns): raise RuntimeError(f"GOOD38 prediction columns missing: {sorted(need-set(x.columns))}")
    x["image_id"]=x.image_id.astype(str); x["group_id"]=x.group_id.astype(str)
    x["label"]=pd.to_numeric(x.label,errors="raise").astype(int); x["outer_fold"]=pd.to_numeric(x.outer_fold,errors="raise").astype(int)
    ids=x[["model_key","seed","outer_fold"]].drop_duplicates()
    if len(ids)!=38: raise RuntimeError(f"GOOD38 identity count={len(ids)} not 38")
    for c in ("group_id","label","outer_fold"):
        if (x.groupby("image_id")[c].nunique(dropna=False)>1).any(): raise RuntimeError(f"GOOD38 inconsistent {c} by image_id")
    cols=["image_id","group_id","label","outer_fold"]
    if "label_name" in x.columns: cols.append("label_name")
    if "relative_path" in x.columns: cols.append("relative_path")
    locked=x[cols].drop_duplicates("image_id").copy()
    if len(locked)!=1055 or locked.image_id.nunique()!=1055: raise RuntimeError(f"GOOD38 does not cover 1055 images: {len(locked)}/{locked.image_id.nunique()}")
    locked["outer_fold"]=locked.outer_fold-1
    if "label_name" not in locked: locked["label_name"]=np.where(locked.label==1,"DFU","Normal")
    sizes={int(f)+1:int(n) for f,n in locked.groupby("outer_fold").size().sort_index().items()}
    if sizes!=FOLD_SIZES: raise RuntimeError(f"historical fold-size mismatch: {sizes}")
    if locked.groupby("group_id").outer_fold.nunique().max()!=1: raise RuntimeError("duplicate group crosses folds in GOOD38 evidence")
    if "relative_path" not in locked or locked.relative_path.isna().any():
        from src.config_data import Config,build_manifest,download_dataset,seed_everything
        tmp=Path("/content/dfu_v3_path_map"); shutil.rmtree(tmp,ignore_errors=True)
        names=("tables","figures","models","xai","predictions","logs","configs","manifests","cache")
        dirs={"root":tmp,**{n:tmp/n for n in names}}
        for d in dirs.values(): Path(d).mkdir(parents=True,exist_ok=True)
        cfg=Config(); cfg.DRIVE_ROOT=str(tmp); cfg.LOCAL_FALLBACK_ROOT=str(tmp); cfg.MOUNT_DRIVE=False; cfg.ALLOW_LOCAL_FALLBACK=True; cfg.SEED=2026; cfg.NUM_WORKERS=2
        seed_everything(cfg.SEED); ds=download_dataset(cfg,dirs); mf=build_manifest(ds,cfg,dirs)
        mp=mf[["image_id","relative_path"]].copy(); mp.image_id=mp.image_id.astype(str)
        locked=locked.drop(columns=["relative_path"],errors="ignore").merge(mp,on="image_id",how="left",validate="one_to_one")
        if locked.relative_path.isna().any(): raise RuntimeError("could not recover relative_path for all 1055 images")
    hr=urllib.request.urlopen(HIST_URL,timeout=120).read()
    if h(hr)!=HIST_SHA: raise RuntimeError("historical Fold-1 GitHub evidence SHA mismatch")
    hp=pd.read_csv(io.BytesIO(hr))
    got=set(hp.image_id.astype(str)); exp=set(locked.loc[locked.outer_fold==0,"image_id"].astype(str))
    if len(hp)!=209 or len(got)!=209 or got!=exp: raise RuntimeError(f"historical Fold-1 proof failed: hist={len(got)} recovered={len(exp)} overlap={len(got&exp)}")
    print("Locked split proof: PASS | 1055 images | fold sizes",sizes,"| historical Fold-1 209/209")
    return locked

def restore38(bundle,locked):
    pred=bundle["pred"].copy(); met=bundle["met"].copy()
    pred["image_id"]=pred.image_id.astype(str)
    ids=sorted({(str(r.model_key),int(r.seed),int(r.outer_fold)-1) for r in met[["model_key","seed","outer_fold"]].itertuples(index=False)})
    if len(ids)!=38: raise RuntimeError("GOOD38 metric identities !=38")
    already=restored=0
    for ident in ids:
        m,s,f=ident
        if ident not in set(base.EXPECTED): raise RuntimeError(f"unexpected GOOD38 identity {ident}")
        ok,_,_,_=base.direct_trial_candidate(locked,m,s,f)
        if ok: already+=1; continue
        pp=pred[(pred.model_key.astype(str)==m)&(pd.to_numeric(pred.seed,errors="coerce")==s)&(pd.to_numeric(pred.outer_fold,errors="coerce")==f+1)].copy()
        mm=met[(met.model_key.astype(str)==m)&(pd.to_numeric(met.seed,errors="coerce")==s)&(pd.to_numeric(met.outer_fold,errors="coerce")==f+1)]
        if len(mm)!=1: raise RuntimeError(f"GOOD38 metric row count invalid for {ident}: {len(mm)}")
        ref=locked[["image_id","group_id","label","label_name","relative_path"]].copy(); ref.image_id=ref.image_id.astype(str)
        for c in ("group_id","label","label_name","relative_path"):
            if c not in pp.columns: pp=pp.merge(ref[["image_id",c]],on="image_id",how="left",validate="many_to_one")
        ok,reason,norm=base.validate_predictions(pp,locked,m,s,f)
        if not ok: raise RuntimeError(f"GOOD38 validation failed {ident}: {reason}")
        metric={k:base.jsonable(v) for k,v in mm.iloc[0].to_dict().items()}
        okid,reasonid=base.complete_identity_valid(metric,m,s,f)
        if not okid: raise RuntimeError(f"GOOD38 metric identity failed {ident}: {reasonid}")
        t=base.trial_path(m,s,f)
        if t.exists(): base.quarantine_trial(t,ident,"replaced_by_verified_GOOD38_embedded_evidence")
        t.mkdir(parents=True,exist_ok=True); base.write_json_atomic(t/"COMPLETE.json",metric); base.write_csv_atomic(t/"test_predictions.csv",norm)
        base.write_json_atomic(t/"REPAIR45_RECOVERY.json",{"version":VERSION,"source":bundle["source"],"evidence_package_sha256":GOOD38_SHA,"training_performed":False,"model_key":m,"seed":s,"outer_fold":f+1,"recovered_at_ns":time.time_ns()})
        restored+=1
    bad=[]
    for m,s,f in ids:
        ok,reason,_,_=base.direct_trial_candidate(locked,m,s,f)
        if not ok: bad.append((m,s,f+1,reason))
    if bad: raise RuntimeError(f"GOOD38 final gate failed: {bad[:5]}")
    print(f"GOOD38 preservation gate: PASS | valid old trials=38 | already={already} restored_without_training={restored}")
    return {"already_valid":already,"restored_without_training":restored,"source":bundle["source"]}

def v3_validate():
    if STATE["done"]: return ORIG_VALIDATE()
    bundle=find_or_upload_bundle()
    locked=lock_from_good38(bundle)
    base.LOCKED_SPLIT.parent.mkdir(parents=True,exist_ok=True)
    base.write_csv_atomic(base.LOCKED_SPLIT,locked)
    locked,sha=ORIG_VALIDATE()
    summary=restore38(bundle,locked)
    base.write_json_atomic(base.RUN_ROOT/"REPAIR45_V3_RECOVERY.json",{"version":VERSION,"status":"PASS","locked_split_sha256":sha,"good38_package_sha256":GOOD38_SHA,"historical_fold1_sha256":HIST_SHA,"fold_sizes":FOLD_SIZES,"good38":summary,"created_at_ns":time.time_ns()})
    STATE["done"]=True
    print("V3 pre-GPU evidence recovery: PASS")
    print("Now the normal 45-trial audit will decide the remaining train/resume set.")
    return locked,sha

base.validate_locked_manifest=v3_validate
print(VERSION)
print("Base Repair45 SHA256: PASS")
print("Policy: recover exact GOOD38 package -> reconstruct/prove lock -> restore 38 without training -> audit 45 -> GPU only remaining non-GOOD.")
print("GPU is blocked until all 38 historical compatible trials validate.")
base.main()
