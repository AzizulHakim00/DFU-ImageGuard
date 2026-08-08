# Replace broken embedded-hex recovery BEFORE the direct runner executes.
import ast
def _replace_top_level_function(_source, _name, _replacement):
    _tree = ast.parse(_source)
    _nodes = [
        _n for _n in _tree.body
        if isinstance(_n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _n.name == _name
    ]
    if len(_nodes) != 1:
        raise RuntimeError(f"Expected exactly one function {_name}, found {len(_nodes)}")
    _node = _nodes[0]
    _lines = _source.splitlines(keepends=True)
    return "".join(
        _lines[:_node.lineno - 1]
        + [_replacement.rstrip() + "\n"]
        + _lines[_node.end_lineno:]
    )

_recovery_source = 'def recover_good_identity_from_aggregate(model, seed, fold_zero):\n    import ast as _ast\n    import hashlib as _hashlib\n    import io as _io\n    import urllib.request as _urlreq\n\n    identity = (model, seed, fold_zero)\n    if identity not in RECOVER_GOOD_FROM_AGGREGATE:\n        raise RuntimeError(f"Plain-evidence reconstruction is not authorized for {identity}")\n\n    already = validate_completed_identity(model, seed, fold_zero, require_current_split=True)\n    if already.get("valid"):\n        print(f"Plain-evidence recovery not needed: {model} seed={seed} fold={fold_zero+1}")\n        return "already_present"\n\n    if identity != ("mobilenetv3_large", 2028, 0):\n        raise RuntimeError(f"No plain evidence is authorized for {identity}")\n\n    ef = fold_zero + 1\n    logits_bytes = _urlreq.urlopen(EVIDENCE_LOGITS_URL, timeout=120).read()\n    metric_bytes = _urlreq.urlopen(EVIDENCE_METRIC_URL, timeout=120).read()\n\n    got_logits_sha = _hashlib.sha256(logits_bytes).hexdigest()\n    got_metric_sha = _hashlib.sha256(metric_bytes).hexdigest()\n    if got_logits_sha != EVIDENCE_LOGITS_SHA256:\n        raise RuntimeError(\n            f"Plain logits evidence SHA mismatch: {got_logits_sha} != {EVIDENCE_LOGITS_SHA256}"\n        )\n    if got_metric_sha != EVIDENCE_METRIC_SHA256:\n        raise RuntimeError(\n            f"Plain metric evidence SHA mismatch: {got_metric_sha} != {EVIDENCE_METRIC_SHA256}"\n        )\n\n    logits_df = pd.read_csv(_io.BytesIO(logits_bytes))\n    metric_df = pd.read_csv(_io.BytesIO(metric_bytes))\n    if len(logits_df) != 209 or logits_df.image_id.astype(str).nunique() != 209:\n        raise RuntimeError(\n            f"Plain logits evidence invalid: rows={len(logits_df)}, "\n            f"unique={logits_df.image_id.astype(str).nunique()}"\n        )\n    if len(metric_df) != 1:\n        raise RuntimeError(f"Plain metric evidence expected 1 row, found {len(metric_df)}")\n\n    expected = expected_by_fold[fold_zero]\n    got_ids = set(logits_df.image_id.astype(str))\n    if got_ids != expected:\n        raise RuntimeError(\n            f"Plain evidence does not match locked fold {ef}: "\n            f"unexpected={len(got_ids-expected)}, missing={len(expected-got_ids)}"\n        )\n\n    base = locked.loc[\n        locked.outer_fold.astype(int) == fold_zero,\n        ["image_id", "group_id", "label", "relative_path"]\n    ].copy()\n    base["image_id"] = base["image_id"].astype(str)\n    logits_df["image_id"] = logits_df["image_id"].astype(str)\n    p = base.merge(logits_df, on="image_id", how="inner", validate="one_to_one")\n    if len(p) != 209:\n        raise RuntimeError(f"Locked/evidence merge expected 209 rows, got {len(p)}")\n\n    metric_record = {k: _python_scalar(v) for k, v in metric_df.iloc[0].to_dict().items()}\n    metric_record["model_key"] = model\n    metric_record["seed"] = int(seed)\n    metric_record["outer_fold"] = int(ef)\n\n    temp = float(metric_record["temperature"])\n    threshold = float(metric_record["threshold"])\n    z = p["logit"].astype(float).to_numpy()\n    p["label_name"] = np.where(p["label"].astype(int).to_numpy() == 1, "DFU", "Normal")\n    p["model_key"] = model\n    p["seed"] = int(seed)\n    p["outer_fold"] = int(ef)\n    p["prob_raw"] = 1.0 / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))\n    p["prob_calibrated"] = 1.0 / (1.0 + np.exp(-np.clip(z / temp, -80.0, 80.0)))\n    p["pred"] = (p["prob_calibrated"].to_numpy() >= threshold).astype(int)\n    p["temperature"] = temp\n    p["threshold"] = threshold\n    p = p[\n        [\n            "image_id", "group_id", "label", "label_name", "relative_path",\n            "model_key", "seed", "outer_fold", "logit", "prob_raw",\n            "prob_calibrated", "pred", "temperature", "threshold"\n        ]\n    ].copy()\n\n    tr = metric_record.get("threshold_rule")\n    if isinstance(tr, str) and tr.startswith("{"):\n        try:\n            metric_record["threshold_rule"] = _ast.literal_eval(tr)\n        except Exception:\n            pass\n\n    t = trial_path(model, seed, fold_zero)\n    t.mkdir(parents=True, exist_ok=True)\n    backup_dir = (\n        RUN_ROOT / "_plain_evidence_recovery_backup" /\n        model / f"seed_{seed}" / f"fold_{ef}"\n    )\n    for name in ("COMPLETE.json", "test_predictions.csv", "TRIAL_VERIFICATION.json"):\n        existing = t / name\n        if existing.exists():\n            backup_dir.mkdir(parents=True, exist_ok=True)\n            shutil.copy2(existing, backup_dir / f"{name}.pre_plain_{time.time_ns()}")\n\n    atomic_csv(t / "test_predictions.csv", p.reset_index(drop=True))\n    atomic_json(t / "COMPLETE.json", metric_record)\n    atomic_json(\n        t / "TRIAL_VERIFICATION.json",\n        {\n            "status": "RECOVERED_FROM_PINNED_PLAIN_V4_EVIDENCE",\n            "model_key": model,\n            "seed": int(seed),\n            "outer_fold": int(ef),\n            "plain_logits_sha256": EVIDENCE_LOGITS_SHA256,\n            "plain_metric_sha256": EVIDENCE_METRIC_SHA256,\n            "locked_split_sha256": LOCKED_SPLIT_SHA256,\n            "prediction_rows": int(len(p)),\n            "training_performed": False,\n            "recovered_at_ns": time.time_ns(),\n        },\n    )\n\n    verified = validate_completed_identity(\n        model, seed, fold_zero, require_current_split=True\n    )\n    if not verified.get("valid"):\n        raise RuntimeError(\n            f"Plain-evidence recovery failed verification for {identity}: {verified}"\n        )\n    print(\n        f"RECOVERED FROM PINNED PLAIN V4 EVIDENCE (NO TRAINING): "\n        f"{model} seed={seed} fold={ef} | rows={len(p)}"\n    )\n    return "recovered"\n'
script = _replace_top_level_function(
    script, "recover_good_identity_from_aggregate", _recovery_source
)
print("Plain CSV recovery patch: PASS — embedded hex recovery removed")

# Replace the only real rr.build_manifest() usage with locked relative-path rehydration.
_manifest_old = """dataset_root=rr.download_dataset(cfg,dirs)
current_manifest=rr.build_manifest(dataset_root,cfg,dirs)
current_manifest=current_manifest.loc[~current_manifest.exclude].copy()
path_map=current_manifest[["image_id","image_path"]].drop_duplicates("image_id")
data=locked.drop(columns=["image_path"],errors="ignore").merge(path_map,on="image_id",how="left",validate="one_to_one")
if data.image_path.isna().any():
    raise RuntimeError(f"Current dataset is missing {int(data.image_path.isna().sum())} locked images; refusing repair.")
"""
_manifest_new = """dataset_root=rr.download_dataset(cfg,dirs)
data=locked.drop(columns=["image_path"],errors="ignore").copy()
data["image_path"]=[str((Path(dataset_root)/str(rp)).resolve()) for rp in data["relative_path"].astype(str)]
_missing_paths=[p for p in data["image_path"].tolist() if not Path(p).is_file()]
if _missing_paths:
    raise RuntimeError(f"Current dataset is missing {len(_missing_paths)} locked image paths; refusing repair. First missing: {_missing_paths[0]}")
"""
if script.count(_manifest_old) != 1:
    raise RuntimeError(
        f"Locked-path patch expected exactly one rr.build_manifest block, found {script.count(_manifest_old)}"
    )
script = script.replace(_manifest_old, _manifest_new, 1)
print("Locked relative-path rehydration patch: PASS")

# AST-based safety assertions: inspect REAL rr.* calls only.
_tree = ast.parse(script)
_rr_calls = []
for _node in ast.walk(_tree):
    if (
        isinstance(_node, ast.Call)
        and isinstance(_node.func, ast.Attribute)
        and isinstance(_node.func.value, ast.Name)
        and _node.func.value.id == "rr"
    ):
        _rr_calls.append(_node.func.attr)

if _rr_calls.count("train_trial") != 1:
    raise RuntimeError(
        f"AST safety check: expected exactly one real rr.train_trial() call, found {_rr_calls.count('train_trial')}"
    )

_forbidden = {"make_outer_folds", "assign_duplicate_groups", "build_manifest"}
_bad = sorted(_forbidden.intersection(_rr_calls))
if _bad:
    raise RuntimeError(f"AST safety check: forbidden real rr.* call(s) found after patch: {_bad}")

if "bytes.fromhex(" in script or "base64.b64decode" in script:
    raise RuntimeError("Safety check: encoded evidence recovery still present after replacement")

print("AST safety check: PASS — no split-regeneration or encoded-evidence calls")
