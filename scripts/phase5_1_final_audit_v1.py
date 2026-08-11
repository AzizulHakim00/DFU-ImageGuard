# DFU Phase-5.1 Final Evidence Audit V1
# Existing artifacts only: XAI quantitative audit + computational evidence + external paired stats (Holm)
# NO TRAINING. NO DATASET CNN INFERENCE. NO THRESHOLD/CALIBRATION FITTING.
from pathlib import Path
import os, json, hashlib, shutil, zipfile, math, warnings
from itertools import combinations
import numpy as np
import pandas as pd

VERSION = "DFU_PHASE5_1_FINAL_AUDIT_V1_20260812"
RUN_ID = "RELIABLE_DFU_CV_V3_MISSING38"
EXPECTED_EXTERNAL_FULL = 662
EXPECTED_EXTERNAL_FILTERED = 634
EXPECTED_NEAR_REMOVED = 28
EXPECTED_MODELS = 3

def banner(msg):
    print("\n" + "="*100)
    print(msg)
    print("="*100)

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None

def find_one(root, filename):
    hits = list(Path(root).rglob(filename))
    return hits[0] if hits else None

def norm(s):
    return str(s).strip().lower().replace(" ","_").replace("-","_")

def find_col(df, candidates=(), contains=()):
    mapping = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in mapping:
            return mapping[norm(c)]
    for col in df.columns:
        n = norm(col)
        if any(tok in n for tok in contains):
            return col
    return None

def boolish(x):
    if pd.isna(x): return False
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, np.integer, float, np.floating)): return float(x) != 0
    return str(x).strip().lower() in {"1","true","yes","y","near","candidate","pass"}

def display_df(name, df, max_rows=30):
    print(f"\n## {name}")
    try:
        from IPython.display import display
        display(df.head(max_rows) if len(df) > max_rows else df)
    except Exception:
        print(df.head(max_rows).to_string(index=False))

def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m-rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj

def exact_mcnemar_p(b, c):
    n = int(b+c)
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(int(b),int(c)), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        k = min(int(b), int(c))
        prob = sum(math.comb(n,i) for i in range(k+1)) / (2**n)
        return float(min(1.0, 2*prob))

def safe_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}

def resolve_root():
    override = os.environ.get("DFU_PHASE5_1_ROOT", "").strip()
    if override:
        root = Path(override)
        if not root.exists():
            raise FileNotFoundError(f"DFU_PHASE5_1_ROOT does not exist: {root}")
        return root, "override"
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:
        if not Path("/content/drive/MyDrive").exists():
            raise RuntimeError("Google Drive is not mounted and DFU_PHASE5_1_ROOT override was not supplied.") from exc
    root = Path("/content/drive/MyDrive/DFU-ImageGuard/runs") / RUN_ID
    if not root.exists():
        raise FileNotFoundError(f"Run root not found: {root}")
    return root, "colab_drive"

ROOT, MODE = resolve_root()
OUT = ROOT / "PHASE5_1_FINAL_AUDIT_V1"
TABLES = OUT / "tables"
FIGPNG = OUT / "figures_png"
FIGPDF = OUT / "figures_pdf"
for d in (OUT, TABLES, FIGPNG, FIGPDF):
    d.mkdir(parents=True, exist_ok=True)

banner(VERSION)
print("Mode:", MODE)
print("Run root:", ROOT)
print("Output root:", OUT)
print("Training performed: FALSE")
print("Dataset CNN inference performed: FALSE")
print("Threshold/calibration fitting performed: FALSE")

verification_candidates = {
    "primary": list(ROOT.rglob("REPAIR45_FINAL_VERIFICATION.json")),
    "phase2": list(ROOT.rglob("PHASE2_FULL_VERIFICATION.json")),
    "phase4": list(ROOT.rglob("PHASE4_EXTERNAL_V2_VERIFICATION.json")) + list(ROOT.rglob("PHASE4_EXTERNAL_V6_VERIFICATION.json")),
    "posthoc": list(ROOT.rglob("PHASE4_POSTHOC_VERIFICATION.json")),
    "phase5": list(ROOT.rglob("PHASE5_PAPER_EVIDENCE_VERIFICATION.json")),
}
verification_rows = []
for phase, hits in verification_candidates.items():
    if not hits:
        verification_rows.append({"phase":phase,"found":False,"path":"","sha256":"","pass_like":False})
        continue
    p = hits[0]
    data = safe_json(p)
    text = json.dumps(data).lower()
    pass_like = ("pass" in text) or bool(data.get("pass", False)) or bool(data.get("verification_pass", False))
    verification_rows.append({"phase":phase,"found":True,"path":str(p.relative_to(ROOT)),"sha256":sha256_file(p),"pass_like":pass_like})
ver_df = pd.DataFrame(verification_rows)
ver_df.to_csv(TABLES/"00_verification_chain.csv", index=False)
display_df("00_verification_chain.csv", ver_df)

phase3 = first_existing([ROOT/"PHASE3_XAI_FULL", ROOT/"PHASE3_XAI"])
if phase3 is None:
    raise FileNotFoundError("Phase-3 XAI directory not found.")

xai_files = {
    "method_summary": find_one(phase3, "06_xai_method_summary.csv"),
    "agreement_summary": find_one(phase3, "07_xai_agreement_summary.csv"),
    "randomization_sanity": find_one(phase3, "08_xai_parameter_randomization_sanity.csv"),
    "faithfulness_case": find_one(phase3, "04_xai_faithfulness_case_metrics.csv"),
    "method_status": find_one(phase3, "03_xai_method_status.csv"),
}
for key in ("method_summary","agreement_summary","randomization_sanity"):
    if xai_files[key] is None:
        raise FileNotFoundError(f"Required Phase-3 XAI file missing: {key}")

xai_method = pd.read_csv(xai_files["method_summary"])
xai_agree = pd.read_csv(xai_files["agreement_summary"])
xai_rand = pd.read_csv(xai_files["randomization_sanity"])
xai_method.to_csv(TABLES/"01_xai_method_summary_exact.csv", index=False)
xai_agree.to_csv(TABLES/"02_xai_agreement_summary_exact.csv", index=False)
xai_rand.to_csv(TABLES/"03_xai_randomization_sanity_exact.csv", index=False)

method_col = find_col(xai_method, candidates=("method","xai_method","explainer","method_name"), contains=("method","explainer"))
metric_rows = []
for c in xai_method.columns:
    if c == method_col:
        continue
    if pd.api.types.is_numeric_dtype(xai_method[c]):
        n = norm(c)
        if "insertion" in n and ("auc" in n or "aopc" in n):
            direction = "higher_is_better"
        elif "deletion" in n and ("auc" in n or "aopc" in n):
            direction = "lower_is_better_if_metric_is_deletion_auc"
        elif any(t in n for t in ("faithfulness","correlation","spearman","pearson")):
            direction = "higher_is_better"
        else:
            direction = "no_automatic_ranking"
        metric_rows.append({
            "column":c, "direction":direction,
            "min":float(pd.to_numeric(xai_method[c], errors="coerce").min()),
            "max":float(pd.to_numeric(xai_method[c], errors="coerce").max()),
            "mean":float(pd.to_numeric(xai_method[c], errors="coerce").mean()),
        })
xai_metric_dict = pd.DataFrame(metric_rows)
xai_metric_dict.to_csv(TABLES/"04_xai_metric_dictionary.csv", index=False)

status_col = find_col(xai_rand, candidates=("status","decision","result","pass"), contains=("status","decision","result","pass"))
rand_summary = {
    "rows": int(len(xai_rand)),
    "status_column": status_col if status_col else "",
    "pass_like_rows": None,
}
if status_col:
    vals = xai_rand[status_col].astype(str).str.lower()
    rand_summary["pass_like_rows"] = int(vals.str.contains("pass|ok|valid|success", regex=True).sum())

agree_numeric = [c for c in xai_agree.columns if pd.api.types.is_numeric_dtype(xai_agree[c])]
xai_summary = {
    "phase3_path": str(phase3.relative_to(ROOT)),
    "method_summary_rows": int(len(xai_method)),
    "method_summary_columns": list(map(str,xai_method.columns)),
    "agreement_summary_rows": int(len(xai_agree)),
    "agreement_numeric_columns": agree_numeric,
    "randomization_rows": int(len(xai_rand)),
    "randomization_status": rand_summary,
    "automatic_best_method_claim": False,
    "reason": "Exact Phase-3 tables are packaged; method ranking is only generated when metric direction is unambiguous."
}
(Path(OUT/"XAI_QUANTITATIVE_AUDIT.json")).write_text(json.dumps(xai_summary, indent=2))

display_df("01_xai_method_summary_exact.csv", xai_method, 50)
display_df("02_xai_agreement_summary_exact.csv", xai_agree, 50)
display_df("03_xai_randomization_sanity_exact.csv", xai_rand, 50)
display_df("04_xai_metric_dictionary.csv", xai_metric_dict, 50)

try:
    import matplotlib.pyplot as plt
    if method_col and len(metric_rows) >= 2 and len(xai_method) >= 2:
        numeric_cols = [r["column"] for r in metric_rows][:10]
        M = xai_method[[method_col]+numeric_cols].copy()
        Z = M[numeric_cols].apply(pd.to_numeric, errors="coerce")
        Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0,np.nan)
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)*1.2), max(4, len(M)*0.45)))
        im = ax.imshow(Z.fillna(0).to_numpy(), aspect="auto")
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(M)))
        ax.set_yticklabels(M[method_col].astype(str).tolist())
        ax.set_title("Phase-3 XAI quantitative summary (column-wise z-scores)")
        fig.colorbar(im, ax=ax, label="z-score")
        fig.tight_layout()
        fig.savefig(FIGPNG/"FigP51_01_XAI_Quantitative_Summary.png", dpi=220, bbox_inches="tight")
        fig.savefig(FIGPDF/"FigP51_01_XAI_Quantitative_Summary.pdf", bbox_inches="tight")
        plt.close(fig)
except Exception as exc:
    warnings.warn(f"XAI figure skipped: {exc}")

inventory = find_one(ROOT, "14_checkpoint_inventory.csv")
comp_rows = []
if inventory is not None:
    inv = pd.read_csv(inventory)
    model_key_col = find_col(inv, candidates=("model_key","model","architecture"), contains=("model_key","model","arch"))
    model_name_col = find_col(inv, candidates=("model_name",), contains=("model_name",))
    path_col = find_col(inv, candidates=("checkpoint_path","path","bundle_relative_path","relative_path"), contains=("checkpoint","path"))
    size_cols = [c for c in inv.columns if any(t in norm(c) for t in ("size","bytes","mb"))]
    param_cols = [c for c in inv.columns if any(t in norm(c) for t in ("param","parameter"))]
    flop_cols = [c for c in inv.columns if any(t in norm(c) for t in ("flop","mac"))]
    latency_cols = [c for c in inv.columns if any(t in norm(c) for t in ("latency","inference_time","time_per_image"))]
    keep_cols = [c for c in [model_key_col,model_name_col,path_col] if c] + size_cols + param_cols + flop_cols + latency_cols
    inv_keep = inv[keep_cols].copy() if keep_cols else inv.copy()
    inv_keep.to_csv(TABLES/"05_checkpoint_inventory_computational_evidence.csv", index=False)
    if model_key_col:
        for mk, g in inv.groupby(model_key_col):
            row = {"model_key": mk, "n_checkpoints": int(len(g))}
            if model_name_col: row["model_name"] = str(g[model_name_col].iloc[0])
            for c in size_cols + param_cols + flop_cols + latency_cols:
                vals = pd.to_numeric(g[c], errors="coerce")
                if vals.notna().any():
                    row[c+"_mean"] = float(vals.mean())
            if path_col:
                resolved_sizes=[]
                for val in g[path_col].astype(str):
                    cand = Path(val)
                    if not cand.is_absolute():
                        c1 = ROOT/cand
                        if c1.exists(): cand=c1
                        else:
                            hits=list(ROOT.rglob(cand.name))
                            cand=hits[0] if hits else cand
                    if cand.exists() and cand.is_file():
                        resolved_sizes.append(cand.stat().st_size)
                if resolved_sizes:
                    row["resolved_checkpoint_size_mb_mean"] = float(np.mean(resolved_sizes)/(1024**2))
            comp_rows.append(row)
else:
    inv_keep = pd.DataFrame()

if not comp_rows:
    ckpts = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".pt",".pth",".ckpt"}]
    for p in ckpts:
        low = p.as_posix().lower()
        mk = "unknown"
        if "convnext" in low: mk="convnextv2_tiny"
        elif "densenet" in low: mk="densenet121"
        elif "mobilenet" in low: mk="mobilenetv3_large"
        if mk!="unknown":
            comp_rows.append({"model_key":mk,"n_checkpoints":1,"resolved_checkpoint_size_mb_mean":p.stat().st_size/(1024**2)})
    if comp_rows:
        cdf=pd.DataFrame(comp_rows)
        comp_rows=[]
        for mk,g in cdf.groupby("model_key"):
            comp_rows.append({"model_key":mk,"n_checkpoints":int(g["n_checkpoints"].sum()),
                              "resolved_checkpoint_size_mb_mean":float(g["resolved_checkpoint_size_mb_mean"].mean())})

comp_df = pd.DataFrame(comp_rows)
if not comp_df.empty:
    if "model_key" in comp_df.columns and comp_df["model_key"].duplicated().any():
        agg={}
        for c in comp_df.columns:
            if c=="model_key": continue
            agg[c]="mean" if pd.api.types.is_numeric_dtype(comp_df[c]) else "first"
        comp_df=comp_df.groupby("model_key",as_index=False).agg(agg)
comp_df["forward_pass_profiled"] = False if not comp_df.empty else pd.Series(dtype=bool)
comp_df["paper_guidance"] = "Report only fields actually present/measured here; FLOPs/MACs/latency remain NOT REPORTED if absent from frozen evidence." if not comp_df.empty else pd.Series(dtype=str)
comp_df.to_csv(TABLES/"06_computational_evidence_summary.csv", index=False)
display_df("06_computational_evidence_summary.csv", comp_df, 30)

phase4 = first_existing([ROOT/"PHASE4_EXTERNAL_V6", ROOT/"PHASE4_EXTERNAL_V2"])
if phase4 is None:
    pred_path = find_one(ROOT, "06_external_ensemble_predictions.csv")
    phase4 = pred_path.parent if pred_path else None
else:
    pred_path = find_one(phase4, "06_external_ensemble_predictions.csv")
if pred_path is None:
    pred_path = find_one(ROOT, "06_external_ensemble_predictions.csv")
if pred_path is None:
    raise FileNotFoundError("06_external_ensemble_predictions.csv not found.")

pred = pd.read_csv(pred_path)
model_col = find_col(pred, candidates=("model_key","model","architecture"), contains=("model_key","model","arch"))
name_col = find_col(pred, candidates=("model_name",), contains=("model_name",))
y_col = find_col(pred, candidates=("y_true","label","true_label","target"), contains=("y_true","true_label","target","label"))
yp_col = find_col(pred, candidates=("y_pred","predicted_label","ensemble_pred","majority_vote_pred","frozen_pred","decision","pred"), contains=("y_pred","predicted_label","majority_vote","frozen_pred","ensemble_pred"))
id_col = find_col(pred, candidates=("external_relpath","relative_path","image_path","filepath","path","image_id","sample_id"), contains=("relpath","image_path","filepath","sample_id","image_id"))
near_col = find_col(pred, candidates=("near_primary_candidate","near_candidate","is_near_primary"), contains=("near_primary","near_candidate"))
group_col = find_col(pred, candidates=("similarity_group","external_similarity_group","group_id"), contains=("similarity_group","group_id"))

if yp_col is not None:
    yy = pd.to_numeric(pred[yp_col], errors="coerce").dropna().unique()
    if not set(map(float,yy)).issubset({0.0,1.0}):
        yp_col = None
if yp_col is None:
    for c in pred.columns:
        n=norm(c)
        if any(tok in n for tok in ("pred","decision","label")) and c != y_col:
            vals=pd.to_numeric(pred[c], errors="coerce").dropna().unique()
            if len(vals) and set(map(float,vals)).issubset({0.0,1.0}):
                yp_col=c
                break

if model_col is None or y_col is None or yp_col is None:
    raise RuntimeError(f"Could not resolve binary prediction schema. Columns={list(pred.columns)}")
if id_col is None:
    pred = pred.copy()
    pred["_row_index_within_model"] = pred.groupby(model_col).cumcount()
    id_col = "_row_index_within_model"

pred["_sample_id"] = pred[id_col].astype(str)
pred["_model_key"] = pred[model_col].astype(str)
pred["_y_true"] = pd.to_numeric(pred[y_col], errors="raise").astype(int)
pred["_y_pred"] = pd.to_numeric(pred[yp_col], errors="raise").astype(int)

near_ids=set()
if near_col is not None:
    near_ids=set(pred.loc[pred[near_col].map(boolish), "_sample_id"].astype(str))
if not near_ids:
    near_paths_file=find_one(ROOT/"PHASE4_POSTHOC_V1" if (ROOT/"PHASE4_POSTHOC_V1").exists() else ROOT, "02_near_candidate_paths.csv")
    if near_paths_file:
        nd=pd.read_csv(near_paths_file)
        n_id_col=find_col(nd, candidates=(id_col,"external_relpath","relative_path","image_path","filepath","path","image_id","sample_id"),
                          contains=("relpath","image_path","filepath","sample_id","image_id"))
        if n_id_col:
            near_ids=set(nd[n_id_col].astype(str))

def build_paired(df):
    truth=df.groupby("_sample_id")["_y_true"].nunique()
    if (truth>1).any():
        raise RuntimeError("Inconsistent y_true across model rows for same sample.")
    wide_pred=df.pivot_table(index="_sample_id", columns="_model_key", values="_y_pred", aggfunc="first")
    wide_y=df.groupby("_sample_id")["_y_true"].first()
    common=wide_pred.dropna().index
    wide_pred=wide_pred.loc[common].astype(int)
    wide_y=wide_y.loc[common].astype(int)
    return wide_pred, wide_y

wide_full, y_full = build_paired(pred)
model_keys=list(wide_full.columns)
if len(model_keys) != EXPECTED_MODELS:
    warnings.warn(f"Expected {EXPECTED_MODELS} models, found {len(model_keys)}: {model_keys}")

filtered_ids=[i for i in wide_full.index if str(i) not in near_ids]
wide_f=wide_full.loc[filtered_ids]
y_f=y_full.loc[filtered_ids]

if (len(wide_full)-len(wide_f)) != EXPECTED_NEAR_REMOVED and near_ids:
    near_basenames={Path(str(x)).name for x in near_ids}
    filtered_ids=[i for i in wide_full.index if Path(str(i)).name not in near_basenames]
    wide_f=wide_full.loc[filtered_ids]
    y_f=y_full.loc[filtered_ids]

if len(y_full) != EXPECTED_EXTERNAL_FULL:
    raise RuntimeError(f"External full paired N mismatch: expected={EXPECTED_EXTERNAL_FULL}, actual={len(y_full)}")
if len(y_f) != EXPECTED_EXTERNAL_FILTERED:
    raise RuntimeError(f"External filtered paired N mismatch: expected={EXPECTED_EXTERNAL_FILTERED}, actual={len(y_f)}")
if (len(y_full)-len(y_f)) != EXPECTED_NEAR_REMOVED:
    raise RuntimeError(f"Near-primary removal mismatch: expected={EXPECTED_NEAR_REMOVED}, actual={len(y_full)-len(y_f)}")

def paired_stats(wide, y, set_name):
    rows=[]
    for a,b in combinations(wide.columns,2):
        ca=(wide[a].to_numpy()==y.to_numpy())
        cb=(wide[b].to_numpy()==y.to_numpy())
        bw=int(np.sum((~ca)&cb))
        cw=int(np.sum(ca&(~cb)))
        p=exact_mcnemar_p(bw,cw)
        acc_a=float(ca.mean()); acc_b=float(cb.mean())
        rows.append({
            "set":set_name,"model_a":a,"model_b":b,"n":int(len(y)),
            "accuracy_a":acc_a,"accuracy_b":acc_b,
            "delta_accuracy_a_minus_b":acc_a-acc_b,
            "a_wrong_b_correct":bw,"a_correct_b_wrong":cw,
            "discordant_total":bw+cw,
            "mcnemar_exact_p_raw":p,
        })
    out=pd.DataFrame(rows)
    if len(out):
        out["mcnemar_p_holm"]=holm_adjust(out["mcnemar_exact_p_raw"].to_numpy())
        out["holm_significant_0_05"]=out["mcnemar_p_holm"]<0.05
    return out

stats_full=paired_stats(wide_full,y_full,"external_full")
stats_filtered=paired_stats(wide_f,y_f,"external_filtered_no_near")
stats_all=pd.concat([stats_full,stats_filtered],ignore_index=True)

name_map={}
if name_col:
    name_map=(pred[[model_col,name_col]].drop_duplicates().set_index(model_col)[name_col].astype(str).to_dict())
for c in ("model_a","model_b"):
    stats_all[c+"_name"]=stats_all[c].map(name_map).fillna(stats_all[c])
stats_all.to_csv(TABLES/"07_external_pairwise_mcnemar_holm.csv",index=False)
display_df("07_external_pairwise_mcnemar_holm.csv", stats_all, 30)

acc_rows=[]
for set_name,wide,y in [("external_full",wide_full,y_full),("external_filtered_no_near",wide_f,y_f)]:
    for mk in wide.columns:
        acc_rows.append({"set":set_name,"model_key":mk,"model_name":name_map.get(mk,mk),
                         "n":len(y),"accuracy":float((wide[mk].to_numpy()==y.to_numpy()).mean())})
acc_df=pd.DataFrame(acc_rows)
acc_df.to_csv(TABLES/"08_external_paired_accuracy_summary.csv",index=False)
display_df("08_external_paired_accuracy_summary.csv", acc_df, 30)

claim_rows=[]
for _,r in stats_filtered.iterrows():
    if r["holm_significant_0_05"]:
        if r["delta_accuracy_a_minus_b"]>0:
            direction=f"{name_map.get(r['model_a'],r['model_a'])} had higher paired accuracy than {name_map.get(r['model_b'],r['model_b'])}"
        elif r["delta_accuracy_a_minus_b"]<0:
            direction=f"{name_map.get(r['model_b'],r['model_b'])} had higher paired accuracy than {name_map.get(r['model_a'],r['model_a'])}"
        else:
            direction="paired accuracies were equal"
        status="SUPPORTED"
    else:
        direction="No Holm-corrected significant paired accuracy difference was established"
        status="NON_SIGNIFICANT"
    claim_rows.append({
        "comparison":f"{name_map.get(r['model_a'],r['model_a'])} vs {name_map.get(r['model_b'],r['model_b'])}",
        "status":status,
        "paper_safe_interpretation":direction,
        "mcnemar_p_holm":r["mcnemar_p_holm"],
    })
claims=pd.DataFrame(claim_rows)
claims.to_csv(TABLES/"09_external_statistical_claims.csv",index=False)
display_df("09_external_statistical_claims.csv", claims, 30)

try:
    import matplotlib.pyplot as plt
    p=acc_df[acc_df["set"]=="external_filtered_no_near"].copy()
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(p["model_name"],p["accuracy"])
    ax.set_ylim(max(0, p["accuracy"].min()-0.08),1.005)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overlap-filtered external paired accuracy")
    ax.tick_params(axis="x",rotation=20)
    for i,v in enumerate(p["accuracy"]):
        ax.text(i,v+0.004,f"{v:.3f}",ha="center")
    fig.tight_layout()
    fig.savefig(FIGPNG/"FigP51_02_External_Paired_Accuracy.png",dpi=220,bbox_inches="tight")
    fig.savefig(FIGPDF/"FigP51_02_External_Paired_Accuracy.pdf",bbox_inches="tight")
    plt.close(fig)

    s=stats_filtered.copy()
    labels=[f"{name_map.get(a,a)}\nvs\n{name_map.get(b,b)}" for a,b in zip(s["model_a"],s["model_b"])]
    fig,ax=plt.subplots(figsize=(8,5))
    vals=-np.log10(np.clip(s["mcnemar_p_holm"].to_numpy(float),1e-300,1))
    ax.bar(labels,vals)
    ax.axhline(-np.log10(0.05),linestyle="--",label="Holm p = 0.05")
    ax.set_ylabel("-log10(Holm-adjusted p)")
    ax.set_title("External paired McNemar tests after near-overlap exclusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGPNG/"FigP51_03_External_McNemar_Holm.png",dpi=220,bbox_inches="tight")
    fig.savefig(FIGPDF/"FigP51_03_External_McNemar_Holm.pdf",bbox_inches="tight")
    plt.close(fig)

    if not comp_df.empty and "resolved_checkpoint_size_mb_mean" in comp_df.columns:
        q=comp_df.dropna(subset=["resolved_checkpoint_size_mb_mean"]).copy()
        if len(q):
            fig,ax=plt.subplots(figsize=(8,5))
            ax.bar(q.get("model_name",q["model_key"]),q["resolved_checkpoint_size_mb_mean"])
            ax.set_ylabel("Mean checkpoint size (MB)")
            ax.set_title("Frozen checkpoint storage footprint")
            ax.tick_params(axis="x",rotation=20)
            fig.tight_layout()
            fig.savefig(FIGPNG/"FigP51_04_Checkpoint_Size.png",dpi=220,bbox_inches="tight")
            fig.savefig(FIGPDF/"FigP51_04_Checkpoint_Size.pdf",bbox_inches="tight")
            plt.close(fig)
except Exception as exc:
    warnings.warn(f"Figure generation warning: {exc}")

guidance = f"""# Phase-5.1 Paper Guidance

## External paired statistics
- Analysis set: overlap-filtered external set.
- N = {len(y_f)} paired images.
- Pairwise test: exact McNemar test on paired predictions.
- Multiple-comparison control: Holm correction across the three architecture comparisons.
- Only comparisons with Holm-adjusted p < 0.05 may be called statistically significant.

## XAI
- Exact Phase-3 quantitative tables were copied without rewriting values.
- Do not claim a single universally best explainer unless the metric direction and statistical comparison support it.
- Parameter-randomization/sanity conclusions must follow `03_xai_randomization_sanity_exact.csv`.

## Computational evidence
- No model forward pass was executed in Phase-5.1.
- Report only computational fields already present in frozen inventories/checkpoints.
- If FLOPs/MACs or latency are absent, write `not measured in the frozen study` rather than importing numbers from unrelated implementations.

## Prohibited changes
- No retraining.
- No external fine-tuning.
- No external threshold fitting.
- No external calibration fitting.
- No new model selection using external labels.
"""
(OUT/"PAPER_GUIDANCE_PHASE5_1.md").write_text(guidance)

input_files=[p for p in xai_files.values() if p is not None] + [pred_path]
if inventory: input_files.append(inventory)
posthoc_ver=find_one(ROOT,"PHASE4_POSTHOC_VERIFICATION.json")
if posthoc_ver: input_files.append(posthoc_ver)
manifest=[]
for p in sorted(set(map(Path,input_files))):
    if p.exists():
        manifest.append({"relative_path":str(p.relative_to(ROOT)),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
manifest_df=pd.DataFrame(manifest)
manifest_df.to_csv(TABLES/"10_input_artifact_manifest.csv",index=False)

checks = {
    "version": VERSION,
    "training_performed": False,
    "dataset_cnn_inference_performed": False,
    "model_forward_pass_performed": False,
    "external_threshold_fitting": False,
    "external_calibration_fitting": False,
    "xai_required_files_found": all(xai_files[k] is not None for k in ("method_summary","agreement_summary","randomization_sanity")),
    "external_full_paired_n": int(len(y_full)),
    "external_filtered_paired_n": int(len(y_f)),
    "near_candidate_ids_removed": int(len(set(wide_full.index)-set(wide_f.index))),
    "external_models": list(map(str,wide_full.columns)),
    "external_model_count": int(len(wide_full.columns)),
    "paired_comparisons_filtered": int(len(stats_filtered)),
    "holm_correction_applied": True,
    "phase3_method_summary_rows": int(len(xai_method)),
    "phase3_agreement_summary_rows": int(len(xai_agree)),
    "phase3_randomization_rows": int(len(xai_rand)),
    "computational_summary_rows": int(len(comp_df)),
}
if len(wide_full.columns) != EXPECTED_MODELS:
    raise RuntimeError(f"Expected 3 architecture ensembles; got {len(wide_full.columns)}.")
if len(stats_filtered) != 3:
    raise RuntimeError(f"Expected 3 pairwise architecture comparisons; got {len(stats_filtered)}.")

checks["verification_pass"] = (
    checks["training_performed"] is False and
    checks["dataset_cnn_inference_performed"] is False and
    checks["model_forward_pass_performed"] is False and
    checks["xai_required_files_found"] and
    checks["external_model_count"]==3 and
    checks["paired_comparisons_filtered"]==3 and
    checks["holm_correction_applied"]
)
(OUT/"PHASE5_1_FINAL_AUDIT_VERIFICATION.json").write_text(json.dumps(checks,indent=2))

zip_path=OUT/"PHASE5_1_FINAL_AUDIT_EXPORT.zip"
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and p != zip_path:
            z.write(p,p.relative_to(OUT))

banner("PHASE-5.1 FINAL AUDIT VERIFICATION: " + ("PASS" if checks["verification_pass"] else "FAIL"))
print("Training performed: FALSE")
print("Dataset CNN inference performed: FALSE")
print("Model forward pass performed: FALSE")
print("External filtered paired N:", len(y_f))
print("External architecture ensembles:", len(wide_f.columns))
print("Pairwise McNemar comparisons:", len(stats_filtered))
print("Holm correction: TRUE")
print("Phase-3 XAI summary rows:", len(xai_method))
print("Final export:", zip_path)
