# DFU Repair-7 LOCAL UPLOAD DIRECT CPU v3
# Plain-Python runner: no nested malformed notebook JSON.

import os, io, json, time, shutil, hashlib, zipfile, ast, urllib.request, re
from pathlib import Path
import numpy as np
import pandas as pd

BOOTSTRAP_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/"
    "d24b690d1b7a12eb6e0f1c01edbb5ac59dbd5487/"
    "notebooks/repair7_local_upload/dfu_local_upload_bootstrap.py"
)
DIRECT_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/"
    "b0f7d6953ab59f02d9ec36ff5784bbf6145c8da7/"
    "notebooks/DFU_Repair7_Only_Preserve38_v1_6_DIRECT_Colab.ipynb"
)

print("=" * 78)
print("DFU Repair-7 PLAIN-PY CPU v3")
print("Notebook-JSON nesting: REMOVED")
print("Local evidence upload: ENABLED")
print("GOOD38 training: FORBIDDEN")
print("Authorized training: original BAD7 only")
print("=" * 78)

# A. Run only the proven GOOD38 bootstrap prefix.
bootstrap = urllib.request.urlopen(BOOTSTRAP_URL, timeout=120).read().decode("utf-8")
launch_marker = 'nb=json.loads(urllib.request.urlopen(RUNNER,timeout=120).read().decode())'
cut = bootstrap.find(launch_marker)
if cut < 0:
    raise RuntimeError("Bootstrap launch marker not found; refusing execution.")

bootstrap_prefix = bootstrap[:cut]
compile(bootstrap_prefix, "dfu_good38_bootstrap_prefix.py", "exec")
print("GOOD38 bootstrap prefix compile: PASS")
exec(compile(bootstrap_prefix, "dfu_good38_bootstrap_prefix.py", "exec"), globals())

for _name in ("RUN", "LOCK", "GOOD", "BAD7"):
    if _name not in globals():
        raise RuntimeError(f"Bootstrap did not define required global: {_name}")
if len(GOOD) != 38 or len(BAD7) != 7:
    raise RuntimeError("Bootstrap protocol identity count changed unexpectedly.")
if not Path(LOCK).is_file():
    raise RuntimeError(f"Locked split missing after bootstrap: {LOCK}")

print("GOOD38 bootstrap handoff: PASS")
print("Preparing pinned direct runner patch...")

# B. Fetch pinned direct v1.6 runner.
direct_raw = urllib.request.urlopen(DIRECT_URL, timeout=120).read()
try:
    direct_nb = json.loads(direct_raw.decode("utf-8"))
except Exception as e:
    raise RuntimeError(
        f"Pinned DIRECT v1.6 notebook JSON is unreadable: {type(e).__name__}: {e}"
    ) from e

direct_cells = [
    c for c in direct_nb.get("cells", [])
    if c.get("cell_type") == "code"
]
if len(direct_cells) != 1:
    raise RuntimeError(
        f"Expected exactly one DIRECT v1.6 code cell; found {len(direct_cells)}"
    )
script = "".join(direct_cells[0]["source"])

required_markers = [
    "DFU REPAIR-7 v1.6 DIRECT SELF-CONTAINED RUNNER",
    "RECOVER_GOOD_FROM_AGGREGATE",
    "rr.train_trial(",
    "current_manifest=rr.build_manifest(dataset_root,cfg,dirs)",
    '["image_id", "group_id", "label", "label_name", "relative_path"]',
]
_missing = [m for m in required_markers if m not in script]
if _missing:
    raise RuntimeError(
        f"Unexpected DIRECT v1.6 source; missing marker(s): {_missing}"
    )

# C. Remove notebook-level GPU gate.
_gpu_gate = re.compile(
    r'import torch\n'
    r'if not torch\.cuda\.is_available\(\):\n'
    r'\s+raise RuntimeError\("GPU runtime required\.[^\n]*"\)\n'
    r'print\("GPU:", torch\.cuda\.get_device_name\(0\)\)'
)
script, _n = _gpu_gate.subn(
    'import torch\nprint("Compute device: CPU (forced repair mode)")',
    script,
    count=1,
)
if _n != 1:
    raise RuntimeError(f"GPU-gate patch expected once; found {_n}")

# D. Remove old embedded GOOD recovery completely.
def _replace_top_level_function(_source, _name, _replacement):
    _tree = ast.parse(_source)
    _nodes = [
        n for n in _tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == _name
    ]
    if len(_nodes) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level function {_name}; found {len(_nodes)}"
        )
    _node = _nodes[0]
    _lines = _source.splitlines(keepends=True)
    return "".join(
        _lines[:_node.lineno - 1]
        + [_replacement.rstrip() + "\n"]
        + _lines[_node.end_lineno:]
    )

script = _replace_top_level_function(
    script,
    "recover_good_identity_from_aggregate",
    '''def recover_good_identity_from_aggregate(model, seed, fold_zero):
    raise RuntimeError(
        "Embedded GOOD recovery is disabled; GOOD38 was restored from "
        "the verified local V4 evidence before this direct runner."
    )'''
)

_old_recovery_set = (
    'RECOVER_GOOD_FROM_AGGREGATE = '
    '{("mobilenetv3_large", 2028, 0)}'
)
if script.count(_old_recovery_set) != 1:
    raise RuntimeError(
        "Expected exactly one DIRECT recovery authorization set; "
        f"found {script.count(_old_recovery_set)}"
    )
script = script.replace(
    _old_recovery_set,
    "RECOVER_GOOD_FROM_AGGREGATE = set()",
    1,
)

# E. CPU-safe train_trial + exhausted-patience resume guard.
_anchor = "\nrr.atomic_torch"
_pos = script.find(_anchor)
if _pos < 0:
    raise RuntimeError("CPU patch anchor rr.atomic_torch not found.")

_cpu_patch = r'''
# ---- CPU-safe train_trial v3 patch ----
import inspect as _repair_inspect
_cpu_train_src = _repair_inspect.getsource(rr.train_trial)

_cpu_replacements = [
    (
        '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
        '    if device.type != "cuda":\n'
        '        raise RuntimeError("GPU runtime is required; training was not started.")\n',
        '    device = torch.device("cpu")\n'
    ),
    (
        '    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)\n',
        '    scaler = torch.amp.GradScaler("cpu", enabled=False)\n'
    ),
    (
        '            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):\n',
        '            with torch.amp.autocast("cpu", enabled=False):\n'
    ),
]

for _old, _new in _cpu_replacements:
    _count = _cpu_train_src.count(_old)
    if _count != 1:
        raise RuntimeError(
            f"CPU train patch expected one source match, found {_count}: "
            f"{_old[:90]!r}"
        )
    _cpu_train_src = _cpu_train_src.replace(_old, _new, 1)

_loop = '    for epoch in range(start_epoch, settings.max_epochs + 1):\n'
if _cpu_train_src.count(_loop) != 1:
    raise RuntimeError(
        "Resume-loop patch expected exactly once; "
        f"found {_cpu_train_src.count(_loop)}"
    )

_resume_guard = (
    '    if start_epoch > 1 and patience_left <= 0:\n'
    '        print(\n'
    '            f"RESUME: early stopping already satisfied — {model_key} "\n'
    '            f"seed={seed} fold={fold + 1}; finalizing saved best "\n'
    '            "checkpoint without another epoch"\n'
    '        )\n'
    '        start_epoch = settings.max_epochs + 1\n'
    '\n'
    '    for epoch in range(start_epoch, settings.max_epochs + 1):\n'
)
_cpu_train_src = _cpu_train_src.replace(_loop, _resume_guard, 1)

exec(
    compile(
        _cpu_train_src,
        "reliable_runner_v2_cpu_resume_safe_v3.py",
        "exec",
    ),
    rr.__dict__,
)
print("CPU-safe train_trial + exhausted-patience resume guard: PASS")
# ---- end CPU-safe train_trial v3 patch ----
'''
script = script[:_pos] + "\n" + _cpu_patch + script[_pos:]

# F. CPU config.
_cfg_anchor = "cfg = rr.build_config(settings)\n"
if script.count(_cfg_anchor) != 1:
    raise RuntimeError(
        f"CPU config anchor expected once; found {script.count(_cfg_anchor)}"
    )
script = script.replace(
    _cfg_anchor,
    _cfg_anchor
    + 'cfg.USE_AMP = False\n'
    + 'cfg.BATCH_SIZE = 4\n'
    + 'cfg.NUM_WORKERS = min(2, max(0, (os.cpu_count() or 1) - 1))\n'
    + 'settings.batch_size = 4\n'
    + 'settings.num_workers = cfg.NUM_WORKERS\n'
    + 'torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))\n'
    + 'print(f"CPU training config: batch={cfg.BATCH_SIZE}, '
      'workers={cfg.NUM_WORKERS}, AMP={cfg.USE_AMP}, '
      'threads={torch.get_num_threads()}")\n',
    1,
)

# G. Locked split only + dataset path rehydration + label_name.
_manifest_old = '''dataset_root=rr.download_dataset(cfg,dirs)
current_manifest=rr.build_manifest(dataset_root,cfg,dirs)
current_manifest=current_manifest.loc[~current_manifest.exclude].copy()
path_map=current_manifest[["image_id","image_path"]].drop_duplicates("image_id")
data=locked.drop(columns=["image_path"],errors="ignore").merge(path_map,on="image_id",how="left",validate="one_to_one")
if data.image_path.isna().any():
    raise RuntimeError(f"Current dataset is missing {int(data.image_path.isna().sum())} locked images; refusing repair.")
'''

_manifest_new = '''dataset_root=rr.download_dataset(cfg,dirs)
data=locked.drop(columns=["image_path"],errors="ignore").copy()
data["image_path"]=[
    str((Path(dataset_root)/str(rp)).resolve())
    for rp in data["relative_path"].astype(str)
]
_missing_paths=[
    p for p in data["image_path"].tolist()
    if not Path(p).is_file()
]
if _missing_paths:
    raise RuntimeError(
        f"Current dataset is missing {len(_missing_paths)} locked image paths; "
        f"refusing repair. First missing: {_missing_paths[0]}"
    )
data["label_name"]=np.where(
    data["label"].astype(int).to_numpy()==1,
    "DFU",
    "Normal",
)
if set(data["label_name"].unique())-{"DFU","Normal"}:
    raise RuntimeError("label_name reconstruction produced unexpected values")
'''

if script.count(_manifest_old) != 1:
    raise RuntimeError(
        "Locked-path/label_name patch expected once; "
        f"found {script.count(_manifest_old)}"
    )
script = script.replace(_manifest_old, _manifest_new, 1)

# H. Preserve a CURRENT incomplete BAD7 repair checkpoint for resume.
_old_repair = '''        existing=validate_completed_identity(model_key,seed,fold,require_current_split=True)
        if existing.get("valid"):
            print(f"SKIP already repaired: {model_key} seed={seed} fold=1")
            continue
        quarantine_bad_identity(model_key,seed,fold)
        t=trial_path(model_key,seed,fold)
        t.mkdir(parents=True,exist_ok=True)
'''

_new_repair = '''        existing=validate_completed_identity(model_key,seed,fold,require_current_split=True)
        if existing.get("valid"):
            print(f"SKIP already repaired: {model_key} seed={seed} fold=1")
            continue

        t=trial_path(model_key,seed,fold)
        _resume_current_partial=False
        _last=t/"last_resume.pt"
        _best=t/"best_model.pt"
        _complete=t/"COMPLETE.json"
        _pred=t/"test_predictions.csv"

        if (
            (not _complete.exists())
            and (not _pred.exists())
            and _last.is_file()
            and _best.is_file()
        ):
            try:
                _lr=torch.load(_last,map_location="cpu",weights_only=False)
                _bp=torch.load(_best,map_location="cpu",weights_only=False)
                _resume_current_partial=(
                    _lr.get("model_key")==model_key
                    and int(_lr.get("seed"))==int(seed)
                    and int(_lr.get("fold"))==int(fold)
                    and _lr.get("source_commit")==settings.source_commit
                    and _bp.get("model_key")==model_key
                    and int(_bp.get("seed"))==int(seed)
                    and int(_bp.get("fold"))==int(fold)
                    and _bp.get("source_commit")==settings.source_commit
                )
            except Exception as _e:
                print(
                    "Partial-checkpoint inspection failed; will quarantine: "
                    f"{type(_e).__name__}: {_e}"
                )
                _resume_current_partial=False

        if _resume_current_partial:
            print(
                "PRESERVE CURRENT PARTIAL REPAIR FOR RESUME: "
                f"{model_key} seed={seed} fold=1"
            )
        else:
            quarantine_bad_identity(model_key,seed,fold)
            t=trial_path(model_key,seed,fold)

        t.mkdir(parents=True,exist_ok=True)
'''

if script.count(_old_repair) != 1:
    raise RuntimeError(
        "Partial-resume repair-loop patch expected once; "
        f"found {script.count(_old_repair)}"
    )
script = script.replace(_old_repair, _new_repair, 1)

# I. AST safety.
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
        "Expected exactly one real rr.train_trial() call; "
        f"found {_rr_calls.count('train_trial')}"
    )

_forbidden = {"make_outer_folds", "assign_duplicate_groups", "build_manifest"}
_bad_calls = sorted(_forbidden.intersection(_rr_calls))
if _bad_calls:
    raise RuntimeError(
        f"Forbidden split-regeneration rr.* call(s) remain: {_bad_calls}"
    )

if "bytes.fromhex(" in script or "base64.b64decode" in script:
    raise RuntimeError("Encoded old GOOD evidence recovery still remains.")
if 'data["label_name"]=np.where' not in script:
    raise RuntimeError("label_name reconstruction patch is missing.")
if "PRESERVE CURRENT PARTIAL REPAIR FOR RESUME" not in script:
    raise RuntimeError("Partial-resume preservation patch is missing.")

compile(script, "DFU_Repair7_DIRECT_CPU_V3.py", "exec")

print("=" * 78)
print("DIRECT PATCH VALIDATION: PASS")
print("Split regeneration calls: NONE")
print("label_name reconstruction: ENABLED")
print("Partial repair checkpoint resume: ENABLED")
print("Exhausted-patience extra epoch: DISABLED")
print("Authorized training: EXACT original BAD7 only")
print("=" * 78)

exec(compile(script, "DFU_Repair7_DIRECT_CPU_V3.py", "exec"), globals())
