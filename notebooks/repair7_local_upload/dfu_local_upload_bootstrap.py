import os,io,json,time,shutil,hashlib,zipfile,ast,urllib.request
from pathlib import Path
import numpy as np,pandas as pd
from google.colab import drive,files

RUN_ID="RELIABLE_DFU_CV_V3_MISSING38"
ROOT=Path("/content/drive/MyDrive/DFU-ImageGuard")
RUN=ROOT/"runs"/RUN_ID
LOCK=RUN/"manifests"/"locked_outer_fold_assignments.csv"
EVID="DFU_GOOD38_RECOVERY_EVIDENCE.zip"
EVID_SHA="6b0c149456f5b29a4a132534d0a6e71ac5fb31550d2cc4ef056131219e62ea5b"
RUNNER="https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/177ab6fb7311f5e864145ec4f2ea3e156fa05db6/notebooks/DFU_Repair7_CPU_FINAL_FIXED.ipynb"
MODELS=("convnextv2_tiny","mobilenetv3_large","densenet121")
SEEDS=(2026,2027,2028)
BAD7={
("convnextv2_tiny",2026,1),("convnextv2_tiny",2027,1),("convnextv2_tiny",2028,1),
("densenet121",2026,1),("densenet121",2027,1),
("mobilenetv3_large",2026,1),("mobilenetv3_large",2027,1)}
GOOD=sorted({(m,s,f) for m in MODELS for s in SEEDS for f in range(1,6)}-BAD7)
assert len(GOOD)==38

def hbytes(b): return hashlib.sha256(b).hexdigest()
def hfile(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()
def acsv(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");d.to_csv(q,index=False);os.replace(q,p)
def ajson(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(o,indent=2,default=str));os.replace(q,p)
def tpath(m,s,f): return RUN/"trials"/m/f"seed_{s}"/f"fold_{f}"

print("="*78)
print("DFU LOCAL-UPLOAD BOOTSTRAP + REPAIR-7 CPU")
print("Upload ONE local file:",EVID)
print("GOOD38 training: FORBIDDEN | BAD7 training only")
print("="*78)
if not Path("/content/drive/MyDrive").is_dir(): drive.mount("/content/drive")
print("Google Drive mount: PASS")

print("\nChoose the evidence ZIP from your computer.")
up=files.upload()
if EVID in up: blob=up[EVID]
elif len(up)==1:
    n,blob=next(iter(up.items()));print("Filename:",n,"| validating SHA")
else: raise RuntimeError("Upload exactly one ZIP.")
if hbytes(blob)!=EVID_SHA: raise RuntimeError("Wrong/corrupt evidence ZIP (SHA mismatch).")
print("Evidence ZIP SHA: PASS")

with zipfile.ZipFile(io.BytesIO(blob)) as z:
    man=json.loads(z.read("MANIFEST.json"))
    pb=z.read("good38_predictions.csv");mb=z.read("good38_metrics.csv")
if hbytes(pb)!=man["predictions_sha256"] or hbytes(mb)!=man["metrics_sha256"]:
    raise RuntimeError("Evidence inner SHA mismatch.")
pred=pd.read_csv(io.BytesIO(pb));met=pd.read_csv(io.BytesIO(mb))
if len(pred)!=8032 or len(met)!=38: raise RuntimeError(f"Evidence size mismatch: {len(pred)}, {len(met)}")
ids=set(zip(met.model_key.astype(str),met.seed.astype(int),met.outer_fold.astype(int)))
if ids!=set(GOOD): raise RuntimeError("Evidence identities are not exact GOOD38.")
print("GOOD38 evidence: PASS | 38 identities / 8032 rows")

cols=["image_id","group_id","label","relative_path","outer_fold"]
x=pred[cols].copy();x["image_id"]=x.image_id.astype(str)
c=x.groupby("image_id").agg(group_id=("group_id","nunique"),label=("label","nunique"),
                            relative_path=("relative_path","nunique"),outer_fold=("outer_fold","nunique"))
if len(c)!=1055 or (c.max(axis=1)!=1).any(): raise RuntimeError("V4 evidence metadata/fold inconsistency.")
locked=x.drop_duplicates("image_id").copy()
locked["outer_fold"]=locked.outer_fold.astype(int)-1
locked=locked.sort_values("image_id").reset_index(drop=True)
sizes=locked.outer_fold.value_counts().sort_index().to_dict()
if sizes!={0:209,1:206,2:217,3:214,4:209}: raise RuntimeError(f"Bad locked fold sizes: {sizes}")
exp={f:set(locked.loc[locked.outer_fold==f,"image_id"].astype(str)) for f in range(5)}
for m,s,f in GOOD:
    p=pred[(pred.model_key.astype(str)==m)&(pred.seed.astype(int)==s)&(pred.outer_fold.astype(int)==f)]
    g=set(p.image_id.astype(str))
    if len(p)!=len(exp[f-1]) or p.image_id.astype(str).duplicated().any() or g!=exp[f-1]:
        raise RuntimeError(f"GOOD evidence/locked mismatch: {(m,s,f)}")
print("Locked assignments rehydrated from V4 evidence: PASS")

RUN.mkdir(parents=True,exist_ok=True);LOCK.parent.mkdir(parents=True,exist_ok=True)
if LOCK.exists():
    old=pd.read_csv(LOCK)
    if not set(cols).issubset(old.columns): raise RuntimeError("Existing locked split missing required columns.")
    a=old[cols].copy();a.image_id=a.image_id.astype(str);a=a.sort_values("image_id").reset_index(drop=True)
    b=locked[cols].copy();b.image_id=b.image_id.astype(str);b=b.sort_values("image_id").reset_index(drop=True)
    if any(a[k].astype(str).tolist()!=b[k].astype(str).tolist() for k in cols):
        raise RuntimeError("Existing locked split conflicts with durable V4 evidence; refusing overwrite.")
    print("Existing locked split matches V4 evidence: PASS")
else:
    acsv(LOCK,locked[cols]);print("Locked split restored from V4 evidence: PASS")
locksha=hfile(LOCK);print("RUN_ROOT:",RUN);print("Locked SHA:",locksha)

def valid(m,s,f):
    t=tpath(m,s,f);cp=t/"COMPLETE.json";pp=t/"test_predictions.csv"
    if not(cp.is_file() and pp.is_file()): return False
    try:
        q=json.loads(cp.read_text());p=pd.read_csv(pp);p.image_id=p.image_id.astype(str)
        if str(q.get("model_key"))!=m or int(q.get("seed"))!=s or int(q.get("outer_fold"))!=f:return False
        need={"image_id","group_id","label","model_key","seed","outer_fold","prob_calibrated","pred"}
        if not need.issubset(p.columns):return False
        g=set(p.image_id)
        if len(p)!=len(exp[f-1]) or p.image_id.duplicated().any() or g!=exp[f-1]:return False
        u=p[["model_key","seed","outer_fold"]].drop_duplicates()
        return len(u)==1 and str(u.iloc[0].model_key)==m and int(u.iloc[0].seed)==s and int(u.iloc[0].outer_fold)==f
    except Exception:return False

kept=[];rec=[]
for m,s,f in GOOD:
    if valid(m,s,f): kept.append((m,s,f));continue
    p=pred[(pred.model_key.astype(str)==m)&(pred.seed.astype(int)==s)&(pred.outer_fold.astype(int)==f)].copy()
    q=met[(met.model_key.astype(str)==m)&(met.seed.astype(int)==s)&(met.outer_fold.astype(int)==f)].copy()
    if len(q)!=1 or len(p)!=len(exp[f-1]) or set(p.image_id.astype(str))!=exp[f-1]:
        raise RuntimeError(f"Recovery evidence invalid: {(m,s,f)}")
    t=tpath(m,s,f);t.mkdir(parents=True,exist_ok=True)
    bk=RUN/"_good38_recovery_backup"/m/f"seed_{s}"/f"fold_{f}"
    for n in ("COMPLETE.json","test_predictions.csv","TRIAL_VERIFICATION.json"):
        o=t/n
        if o.exists():
            bk.mkdir(parents=True,exist_ok=True);shutil.copy2(o,bk/f"{n}.pre_{time.time_ns()}")
    r={}
    for k,v in q.iloc[0].to_dict().items():
        r[k]=None if pd.isna(v) else (v.item() if isinstance(v,np.generic) else v)
    tr=r.get("threshold_rule")
    if isinstance(tr,str) and tr.startswith("{"):
        try:r["threshold_rule"]=ast.literal_eval(tr)
        except Exception:pass
    acsv(t/"test_predictions.csv",p.reset_index(drop=True));ajson(t/"COMPLETE.json",r)
    ajson(t/"TRIAL_VERIFICATION.json",{"status":"RECOVERED_FROM_LOCAL_V4_GOOD38","model_key":m,"seed":s,
          "outer_fold":f,"training_performed":False,"prediction_rows":len(p),
          "evidence_zip_sha256":EVID_SHA,"locked_split_sha256":locksha,"recovered_at_ns":time.time_ns()})
    if not valid(m,s,f): raise RuntimeError(f"Post-recovery validation failed: {(m,s,f)}")
    rec.append((m,s,f));print("RECOVERED GOOD — NO TRAINING:",m,s,f,len(p))

bad=[x for x in GOOD if not valid(*x)]
if bad: raise RuntimeError(f"GOOD38 final validation failed: {bad[:3]}")
print("="*78);print("GOOD38 BOOTSTRAP FINAL: PASS");print("Already valid:",len(kept),"Recovered:",len(rec))
print("Launching CPU Repair-7; ONLY BAD7 may train.");print("="*78)

nb=json.loads(urllib.request.urlopen(RUNNER,timeout=120).read().decode())
cells=[c for c in nb["cells"] if c.get("cell_type")=="code"]
if len(cells)!=1:raise RuntimeError("Pinned CPU runner structure changed.")
code="".join(cells[0]["source"]);compile(code,"cpu_runner.py","exec");exec(compile(code,"cpu_runner.py","exec"),globals())
