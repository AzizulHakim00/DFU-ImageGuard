from __future__ import annotations

import ast
import base64
import errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "DFU_REPAIR45_GOOD38_PROVEN_V4_20260810"
BASE_COMMIT = "0413eec1acea851664f52e8af7cc7934182aa24b"
BASE_SHA256 = "be3de60220b677da3d9bad5d8a06dcbb3e67f498a2fe97972a475a74666cab99"
BASE_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/"
    + BASE_COMMIT
    + "/scripts/dfu_repair45_reconcile_locked_v1.py"
)
GOOD38_NOTEBOOK_NAME = "DFU_Repair7_SELF_CONTAINED_GOOD38_RECOVERY_CPU.ipynb"
GOOD38_PACKAGE_SHA256 = "6b0c149456f5b29a4a132534d0a6e71ac5fb31550d2cc4ef056131219e62ea5b"
GOOD38_PREDICTION_ROWS = 8032
GOOD38_METRIC_ROWS = 38
EXPECTED_IMAGE_COUNT = 1055
EXPECTED_FOLD_SIZES_ONE = {1: 209, 2: 206, 3: 217, 4: 214, 5: 209}
HISTORICAL_FOLD1_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/"
    "11881ee007149b23e3664298c0d09c1f988acdd1/"
    "notebooks/repair7_evidence/mobilenetv3_large_seed2028_fold1_logits.csv"
)
HISTORICAL_FOLD1_SHA256 = "c5f696aa89b0436ad100911fe8d4f478b91b86b94da6d269823d51c8308b4020"

MODELS = ("convnextv2_tiny", "mobilenetv3_large", "densenet121")
SEEDS = (2026, 2027, 2028)
FOLDS_ZERO = (0, 1, 2, 3, 4)
EXPECTED45 = {
    (m, int(s), int(f))
    for f in FOLDS_ZERO
    for s in SEEDS
    for m in MODELS
}
BAD7 = {
    ("convnextv2_tiny", 2026, 0),
    ("convnextv2_tiny", 2027, 0),
    ("convnextv2_tiny", 2028, 0),
    ("densenet121", 2026, 0),
    ("densenet121", 2027, 0),
    ("mobilenetv3_large", 2026, 0),
    ("mobilenetv3_large", 2027, 0),
}
GOOD38 = EXPECTED45 - BAD7
if len(EXPECTED45) != 45 or len(BAD7) != 7 or len(GOOD38) != 38:
    raise RuntimeError("Protocol identity constants are inconsistent")

DRIVE_PROBE_BYTES = 64 * 1024 * 1024
MIN_REPORTED_FREE_BYTES = 1200 * 1024 * 1024
MAX_REASONABLE_DRIVE_TOTAL_BYTES = 100 * 1024**4
LOCAL_RECON_ROOT = Path("/content/dfu_repair45_v4_reconstruct")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_quota_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EDQUOT}:
        return True
    text = str(exc).lower()
    return any(token in text for token in ("quota", "storage full", "no space left", "disk full"))


def _static_string(node: ast.AST, known: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return known.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, known)
        right = _static_string(node.right, known)
        if left is not None and right is not None:
            return left + right
    return None


def extract_static_string_assignments(code: str) -> dict[str, str]:
    known: dict[str, str] = {}
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _static_string(node.value, known)
            if value is not None:
                known[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            value = _static_string(node.value, known)
            if value is not None:
                known[node.target.id] = value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            value = _static_string(node.value, known)
            if value is not None:
                known[node.target.id] = known.get(node.target.id, "") + value
    return known


def normalize_identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    for col in ("model_key", "seed", "outer_fold"):
        if col not in x.columns:
            raise ValueError(f"missing identity column: {col}")
    x["model_key"] = x["model_key"].astype(str)
    x["seed"] = pd.to_numeric(x["seed"], errors="raise").astype(int)
    x["outer_fold"] = pd.to_numeric(x["outer_fold"], errors="raise").astype(int)
    return x


def validate_good38_tables(pred: pd.DataFrame, met: pd.DataFrame) -> None:
    pred = normalize_identity_frame(pred)
    met = normalize_identity_frame(met)
    pred_need = {
        "image_id", "group_id", "label", "model_key", "seed", "outer_fold",
        "prob_calibrated", "pred",
    }
    missing = sorted(pred_need - set(pred.columns))
    if missing:
        raise ValueError(f"GOOD38 prediction columns missing: {missing}")
    if len(pred) != GOOD38_PREDICTION_ROWS or len(met) != GOOD38_METRIC_ROWS:
        raise ValueError(
            f"GOOD38 evidence size mismatch: predictions={len(pred)}, metrics={len(met)}"
        )

    pred_ids_one = {
        (str(r.model_key), int(r.seed), int(r.outer_fold) - 1)
        for r in pred[["model_key", "seed", "outer_fold"]].drop_duplicates().itertuples(index=False)
    }
    met_ids_one = {
        (str(r.model_key), int(r.seed), int(r.outer_fold) - 1)
        for r in met[["model_key", "seed", "outer_fold"]].drop_duplicates().itertuples(index=False)
    }
    if pred_ids_one != GOOD38:
        raise ValueError(
            f"GOOD38 prediction identity set mismatch: missing={sorted(GOOD38-pred_ids_one)} "
            f"unexpected={sorted(pred_ids_one-GOOD38)}"
        )
    if met_ids_one != GOOD38:
        raise ValueError(
            f"GOOD38 metric identity set mismatch: missing={sorted(GOOD38-met_ids_one)} "
            f"unexpected={sorted(met_ids_one-GOOD38)}"
        )

    p = pred.copy()
    p["image_id"] = p["image_id"].astype(str)
    p["group_id"] = p["group_id"].astype(str)
    p["label"] = pd.to_numeric(p["label"], errors="raise").astype(int)
    probs = pd.to_numeric(p["prob_calibrated"], errors="coerce").to_numpy(float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("GOOD38 prob_calibrated contains invalid values")
    hard = pd.to_numeric(p["pred"], errors="coerce")
    if hard.isna().any() or not set(hard.astype(int).unique()).issubset({0, 1}):
        raise ValueError("GOOD38 pred contains values outside {0,1}")
    if not set(p["label"].unique()).issubset({0, 1}):
        raise ValueError("GOOD38 label contains values outside {0,1}")

    for identity in sorted(GOOD38):
        m, s, f0 = identity
        part = p[
            (p["model_key"] == m)
            & (p["seed"] == s)
            & (p["outer_fold"] == f0 + 1)
        ]
        expected_n = EXPECTED_FOLD_SIZES_ONE[f0 + 1]
        if len(part) != expected_n:
            raise ValueError(f"GOOD38 row count mismatch for {identity}: {len(part)} != {expected_n}")
        if part["image_id"].duplicated().any() or part["image_id"].nunique() != expected_n:
            raise ValueError(f"GOOD38 duplicate/missing image IDs for {identity}")

    for column in ("group_id", "label", "outer_fold"):
        n = p.groupby("image_id")[column].nunique(dropna=False)
        if (n > 1).any():
            bad = n[n > 1].index.astype(str).tolist()[:5]
            raise ValueError(f"GOOD38 inconsistent {column} for image IDs: {bad}")
    for optional in ("label_name", "relative_path"):
        if optional in p.columns:
            n = p.groupby("image_id")[optional].nunique(dropna=False)
            if (n > 1).any():
                bad = n[n > 1].index.astype(str).tolist()[:5]
                raise ValueError(f"GOOD38 inconsistent {optional} for image IDs: {bad}")

    unique_images = p["image_id"].nunique()
    if unique_images != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"GOOD38 unique image count {unique_images} != {EXPECTED_IMAGE_COUNT}")


def parse_good38_notebook(nbraw: bytes, source: str) -> dict[str, Any]:
    try:
        nb = json.loads(nbraw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"notebook JSON unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise ValueError("notebook JSON does not contain a cells list")
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )
    constants = extract_static_string_assignments(code)
    package_declared = constants.get("EVIDENCE_PACKAGE_SHA256")
    if package_declared != GOOD38_PACKAGE_SHA256:
        raise ValueError(
            f"wrong evidence package identifier: {package_declared!r}"
        )
    b64 = constants.get("_EVIDENCE_B64")
    if not b64:
        raise ValueError("embedded GOOD38 payload was not found")
    try:
        package = base64.b64decode(b64, validate=True)
    except Exception as exc:
        raise ValueError(f"embedded GOOD38 base64 invalid: {type(exc).__name__}: {exc}") from exc
    if sha256_bytes(package) != GOOD38_PACKAGE_SHA256:
        raise ValueError("embedded GOOD38 package SHA-256 mismatch")

    try:
        with zipfile.ZipFile(io.BytesIO(package), "r") as zf:
            names = set(zf.namelist())
            required = {"MANIFEST.json", "good38_predictions.csv", "good38_metrics.csv"}
            if not required.issubset(names):
                raise ValueError(f"GOOD38 ZIP missing members: {sorted(required-names)}")
            manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))
            pred_bytes = zf.read("good38_predictions.csv")
            metric_bytes = zf.read("good38_metrics.csv")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"GOOD38 ZIP unreadable: {type(exc).__name__}: {exc}") from exc

    if sha256_bytes(pred_bytes) != str(manifest.get("predictions_sha256")):
        raise ValueError("GOOD38 predictions inner SHA-256 mismatch")
    if sha256_bytes(metric_bytes) != str(manifest.get("metrics_sha256")):
        raise ValueError("GOOD38 metrics inner SHA-256 mismatch")

    pred = pd.read_csv(io.BytesIO(pred_bytes))
    met = pd.read_csv(io.BytesIO(metric_bytes))
    validate_good38_tables(pred, met)
    return {
        "source": source,
        "package_sha256": GOOD38_PACKAGE_SHA256,
        "pred": pred,
        "met": met,
        "manifest": manifest,
    }


def load_base_module():
    raw = urllib.request.urlopen(BASE_URL, timeout=120).read()
    actual = sha256_bytes(raw)
    if actual != BASE_SHA256:
        raise RuntimeError(
            f"Base Repair45 SHA mismatch: expected={BASE_SHA256} actual={actual}"
        )
    path = Path("/content/dfu_repair45_base_v1_for_v4.py")
    path.write_bytes(raw)
    compile(raw.decode("utf-8"), str(path), "exec")
    spec = importlib.util.spec_from_file_location("dfu_repair45_base_v1_for_v4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create base Repair45 module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def drive_capacity_probe(base) -> None:
    root = Path("/content/drive/MyDrive")
    try:
        usage = shutil.disk_usage(root)
        free_gib = usage.free / 1024**3
        total_gib = usage.total / 1024**3
        print(f"DriveFS reported capacity: total={total_gib:.2f} GiB free={free_gib:.2f} GiB")
        if 0 < usage.total <= MAX_REASONABLE_DRIVE_TOTAL_BYTES and usage.free < MIN_REPORTED_FREE_BYTES:
            raise RuntimeError(
                f"Drive reports only {free_gib:.2f} GiB free; at least "
                f"{MIN_REPORTED_FREE_BYTES/1024**3:.2f} GiB headroom is required before GPU repair."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"DriveFS free-space report unavailable ({type(exc).__name__}); continuing to write probe.")

    probe = base.RUN_ROOT / ".REPAIR45_V4_QUOTA_PROBE.bin"
    block = bytes(4 * 1024 * 1024)
    written = 0
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        with probe.open("wb") as handle:
            while written < DRIVE_PROBE_BYTES:
                handle.write(block)
                written += len(block)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        if probe.stat().st_size != DRIVE_PROBE_BYTES:
            raise RuntimeError(
                f"Drive quota probe size mismatch: {probe.stat().st_size} != {DRIVE_PROBE_BYTES}"
            )
        print(f"Drive quota write probe: PASS ({DRIVE_PROBE_BYTES/1024**2:.0f} MiB write verified)")
    except Exception as exc:
        kind = "quota/storage" if is_quota_error(exc) else type(exc).__name__
        raise RuntimeError(
            f"Drive preflight failed before training ({kind}): {exc}"
        ) from exc
    finally:
        try:
            probe.unlink(missing_ok=True)
        except Exception:
            pass


def quota_aware_quarantine(base, trial: Path, identity: tuple[str, int, int], reason: str) -> Path | None:
    trial = Path(trial)
    if not trial.exists():
        return None
    model, seed, fold0 = identity
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()%1_000_000:06d}"
    target = (
        base.RUN_ROOT / "quarantine_metadata" / "repair45_v4" / stamp
        / model / f"seed_{seed}" / f"fold_{fold0+1}"
    )
    target.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []

    for path in sorted([p for p in trial.rglob("*") if p.is_file()]):
        rel = path.relative_to(trial)
        entry: dict[str, Any] = {
            "relative_path": rel.as_posix(),
            "bytes": int(path.stat().st_size),
        }
        try:
            entry["sha256"] = sha256_file(path)
        except Exception as exc:
            entry["sha256_error"] = f"{type(exc).__name__}: {exc}"
        if path.suffix.lower() == ".pt":
            entry["action"] = "deleted_invalid_checkpoint_after_hash"
            path.unlink(missing_ok=True)
        else:
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(path, dst)
                entry["action"] = "renamed_to_metadata_quarantine"
            except Exception:
                entry["action"] = "deleted_after_hash_rename_failed"
                path.unlink(missing_ok=True)
        inventory.append(entry)

    shutil.rmtree(trial, ignore_errors=True)
    freed_pt = sum(x["bytes"] for x in inventory if x.get("action") == "deleted_invalid_checkpoint_after_hash")
    base.write_json_atomic(
        target / "QUARANTINE_REASON.json",
        {
            "version": VERSION,
            "identity": {"model_key": model, "seed": seed, "outer_fold": fold0 + 1},
            "reason": reason,
            "storage_policy": "invalid .pt checkpoints hashed then deleted; small files renamed when possible",
            "invalid_checkpoint_bytes_released": int(freed_pt),
            "inventory": inventory,
            "quarantined_at_ns": time.time_ns(),
        },
    )
    print(
        f"Storage-aware quarantine: {model} seed={seed} fold={fold0+1} | "
        f"released_invalid_pt={freed_pt/1024**2:.1f} MiB"
    )
    return target


def find_or_upload_good38_bundle() -> dict[str, Any]:
    roots = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
        Path("/content/drive/.shortcut-targets-by-id"),
    ]
    roots = [r for r in roots if r.exists()]
    direct_candidates = [
        Path("/content/drive/MyDrive") / GOOD38_NOTEBOOK_NAME,
        Path("/content/drive/MyDrive/DFU-ImageGuard") / GOOD38_NOTEBOOK_NAME,
        Path("/content/drive/MyDrive/DFU-ImageGuard") / "evidence" / GOOD38_NOTEBOOK_NAME,
    ]
    seen: set[str] = set()
    for candidate in direct_candidates:
        if not candidate.is_file():
            continue
        seen.add(str(candidate))
        try:
            bundle = parse_good38_notebook(candidate.read_bytes(), str(candidate))
            print("GOOD38 package source: PASS", candidate)
            return bundle
        except Exception as exc:
            print("Rejected direct GOOD38 candidate:", candidate, type(exc).__name__, exc)

    print("Searching mounted Drive for exact GOOD38 recovery notebook...")
    scanned = 0
    for root in roots:
        for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
            scanned += 1
            dirs[:] = [
                d for d in dirs
                if d not in {".Trash", ".cache", "__pycache__", ".ipynb_checkpoints", "node_modules"}
            ]
            if GOOD38_NOTEBOOK_NAME not in names:
                continue
            candidate = Path(current) / GOOD38_NOTEBOOK_NAME
            if str(candidate) in seen:
                continue
            seen.add(str(candidate))
            try:
                bundle = parse_good38_notebook(candidate.read_bytes(), str(candidate))
                print("GOOD38 package source: PASS", candidate)
                return bundle
            except Exception as exc:
                print("Rejected GOOD38 candidate:", candidate, type(exc).__name__, exc)
    print(f"Drive search complete; directories scanned={scanned}. Valid GOOD38 package not found.")
    print("Upload the exact self-contained GOOD38 recovery notebook when the chooser opens.")
    from google.colab import files
    uploaded = files.upload()
    failures: list[str] = []
    for name, data in uploaded.items():
        try:
            bundle = parse_good38_notebook(data, f"manual_upload:{name}")
            print("GOOD38 manual upload: PASS", name)
            return bundle
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print("Rejected upload:", failures[-1])
    raise RuntimeError(
        "No valid GOOD38 evidence package was supplied. GPU remains blocked. "
        + " | ".join(failures[:3])
    )


def evidence_lock_from_good38(bundle: dict[str, Any]) -> pd.DataFrame:
    pred = normalize_identity_frame(bundle["pred"])
    pred["image_id"] = pred["image_id"].astype(str)
    pred["group_id"] = pred["group_id"].astype(str)
    pred["label"] = pd.to_numeric(pred["label"], errors="raise").astype(int)
    columns = ["image_id", "group_id", "label", "outer_fold"]
    if "label_name" in pred.columns:
        columns.append("label_name")
    if "relative_path" in pred.columns:
        columns.append("relative_path")
    locked = pred[columns].drop_duplicates("image_id").copy()
    if len(locked) != EXPECTED_IMAGE_COUNT or locked["image_id"].nunique() != EXPECTED_IMAGE_COUNT:
        raise RuntimeError("GOOD38 evidence cannot produce a complete 1055-image lock")
    locked["outer_fold"] = locked["outer_fold"].astype(int) - 1
    if "label_name" not in locked.columns:
        locked["label_name"] = np.where(locked["label"].astype(int) == 1, "DFU", "Normal")
    sizes = {
        int(f) + 1: int(n)
        for f, n in locked.groupby("outer_fold").size().sort_index().items()
    }
    if sizes != EXPECTED_FOLD_SIZES_ONE:
        raise RuntimeError(f"GOOD38 evidence fold sizes changed: {sizes}")
    if locked.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("GOOD38 evidence has a duplicate group crossing outer folds")
    return locked


def recover_original_order_and_prove(bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover the locked split without ever regenerating folds.

    Fold/group assignments come only from the cryptographically pinned GOOD38
    predictions. The pinned manifest builder is used only to recover the same
    dataset row/path order and to verify the current immutable Kaggle-v1 paths.
    """
    evidence_lock = evidence_lock_from_good38(bundle)
    shutil.rmtree(LOCAL_RECON_ROOT, ignore_errors=True)
    LOCAL_RECON_ROOT.mkdir(parents=True, exist_ok=True)

    from src import reliable_runner_v2 as rr
    from src.config_data import build_manifest, download_dataset, seed_everything

    settings = rr.ReliableSettingsV2(
        run_id="REPAIR45_V4_LOCAL_MANIFEST_ONLY",
        drive_root=str(LOCAL_RECON_ROOT),
        backup_root=str(LOCAL_RECON_ROOT / "backup"),
        source_commit="349143b4d8b16f885adce3559542f6c202a2bca1",
    )
    cfg = rr.build_config(settings)
    cfg.DRIVE_ROOT = str(LOCAL_RECON_ROOT)
    cfg.LOCAL_FALLBACK_ROOT = str(LOCAL_RECON_ROOT)
    cfg.MOUNT_DRIVE = False
    cfg.ALLOW_LOCAL_FALLBACK = True
    cfg.NUM_WORKERS = min(2, int(cfg.NUM_WORKERS))

    names = ("tables", "figures", "models", "xai", "predictions", "logs", "configs", "manifests", "cache")
    dirs = {"root": LOCAL_RECON_ROOT, **{name: LOCAL_RECON_ROOT / name for name in names}}
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    print("Recovering pinned dataset manifest order in /content (NO fold regeneration)...")
    seed_everything(cfg.SEED)
    dataset_root = download_dataset(cfg, dirs)
    manifest = build_manifest(dataset_root, cfg, dirs).copy()
    manifest["image_id"] = manifest["image_id"].astype(str)
    if manifest["image_id"].duplicated().any():
        raise RuntimeError("Pinned current manifest contains duplicate image_id values")

    evidence_ids = set(evidence_lock["image_id"].astype(str))
    current_ids = set(manifest["image_id"].astype(str))
    missing_now = sorted(evidence_ids - current_ids)
    if missing_now:
        raise RuntimeError(
            f"Current Kaggle-v1 manifest is missing {len(missing_now)} historical evidence images; "
            f"first={missing_now[:5]}"
        )

    current = manifest.loc[manifest["image_id"].isin(evidence_ids)].copy()
    if len(current) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Current manifest selection has {len(current)} historical images, expected {EXPECTED_IMAGE_COUNT}"
        )
    ev = evidence_lock[["image_id", "group_id", "label", "label_name", "relative_path", "outer_fold"]].copy()
    ev["image_id"] = ev["image_id"].astype(str)
    joined = current.merge(
        ev,
        on="image_id",
        how="left",
        suffixes=("_current", "_evidence"),
        validate="one_to_one",
        sort=False,
    )
    if joined["outer_fold"].isna().any():
        raise RuntimeError("Historical evidence mapping failed for current manifest rows")

    mismatch_counts: dict[str, int] = {}
    for col in ("label", "label_name", "relative_path"):
        a = joined[f"{col}_current"].astype(str)
        b = joined[f"{col}_evidence"].astype(str)
        n = int((a != b).sum())
        if n:
            mismatch_counts[col] = n
    if mismatch_counts:
        raise RuntimeError(
            "Current pinned dataset manifest disagrees with GOOD38 evidence: "
            + json.dumps(mismatch_counts, sort_keys=True)
        )

    joined["group_id"] = joined["group_id"].astype(str)
    joined["outer_fold"] = pd.to_numeric(joined["outer_fold"], errors="raise").astype(int)
    joined["label"] = pd.to_numeric(joined["label_evidence"], errors="raise").astype(int)
    joined["label_name"] = joined["label_name_evidence"].astype(str)
    joined["relative_path"] = joined["relative_path_evidence"].astype(str)
    drop_cols = [
        "label_current", "label_evidence",
        "label_name_current", "label_name_evidence",
        "relative_path_current", "relative_path_evidence",
    ]
    locked = joined.drop(columns=[c for c in drop_cols if c in joined.columns]).copy()

    sizes = {
        int(f) + 1: int(n)
        for f, n in locked.groupby("outer_fold").size().sort_index().items()
    }
    if sizes != EXPECTED_FOLD_SIZES_ONE:
        raise RuntimeError(f"Recovered historical fold sizes changed: {sizes}")
    if locked.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("Recovered historical duplicate group crosses outer folds")

    pred = normalize_identity_frame(bundle["pred"])
    pred["image_id"] = pred["image_id"].astype(str)
    order_checks: dict[str, str] = {}
    for f0 in FOLDS_ZERO:
        candidates = sorted(identity for identity in GOOD38 if identity[2] == f0)
        if not candidates:
            raise RuntimeError(f"GOOD38 has no representative identity for fold {f0+1}")
        m, s, _ = candidates[0]
        saved_order = pred.loc[
            (pred["model_key"] == m) & (pred["seed"] == s) & (pred["outer_fold"] == f0 + 1),
            "image_id",
        ].tolist()
        manifest_order = locked.loc[locked["outer_fold"] == f0, "image_id"].astype(str).tolist()
        if saved_order != manifest_order:
            mismatch_at = next(
                (i for i, (a, b) in enumerate(zip(saved_order, manifest_order)) if a != b),
                min(len(saved_order), len(manifest_order)),
            )
            raise RuntimeError(
                f"Pinned manifest row order does not match saved V4 Fold-{f0+1} prediction order; "
                f"first mismatch index={mismatch_at}"
            )
        order_checks[str(f0 + 1)] = hashlib.sha256("\n".join(saved_order).encode("utf-8")).hexdigest()

    hist_bytes = urllib.request.urlopen(HISTORICAL_FOLD1_URL, timeout=120).read()
    if sha256_bytes(hist_bytes) != HISTORICAL_FOLD1_SHA256:
        raise RuntimeError("Historical Fold-1 evidence SHA-256 mismatch")
    hist = pd.read_csv(io.BytesIO(hist_bytes))
    hist_ids = set(hist["image_id"].astype(str))
    fold1_ids = set(locked.loc[locked["outer_fold"] == 0, "image_id"].astype(str))
    if len(hist) != 209 or len(hist_ids) != 209 or hist_ids != fold1_ids:
        raise RuntimeError(
            f"Independent historical Fold-1 proof failed: hist={len(hist_ids)} "
            f"recovered={len(fold1_ids)} overlap={len(hist_ids & fold1_ids)}"
        )

    order_payload = "\n".join(
        f"{row.image_id},{int(row.outer_fold)},{row.group_id}"
        for row in locked[["image_id", "outer_fold", "group_id"]].itertuples(index=False)
    )
    proof = {
        "status": "PASS",
        "method": "GOOD38_fold_group_map_plus_pinned_manifest_order",
        "no_make_outer_folds_called": True,
        "no_assign_duplicate_groups_called": True,
        "exact_1055_image_match": True,
        "dataset_metadata_match_columns": ["label", "label_name", "relative_path"],
        "fold_sizes": sizes,
        "saved_prediction_order_verified_all_5_folds": True,
        "fold_order_sha256": order_checks,
        "historical_fold1_match": "209/209",
        "recovered_row_order_sha256": hashlib.sha256(order_payload.encode("utf-8")).hexdigest(),
    }
    print(
        "Locked split/order proof: PASS | 1055/1055 evidence images | "
        "all 5 saved fold orders match | historical Fold-1 209/209 | NO fold regeneration"
    )
    return locked, proof


def metric_record_from_row(row: pd.Series, base) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, np.generic):
            value = value.item()
        try:
            if pd.isna(value):
                value = None
        except Exception:
            pass
        out[key] = value
    threshold_rule = out.get("threshold_rule")
    if isinstance(threshold_rule, str) and threshold_rule.strip().startswith(("{", "[", "(")):
        try:
            out["threshold_rule"] = ast.literal_eval(threshold_rule)
        except Exception:
            pass
    return {key: base.jsonable(value) if not isinstance(value, (dict, list, tuple)) else value for key, value in out.items()}


def restore_good38(base, bundle: dict[str, Any], locked: pd.DataFrame, locked_sha: str) -> dict[str, Any]:
    pred = normalize_identity_frame(bundle["pred"])
    met = normalize_identity_frame(bundle["met"])
    pred["image_id"] = pred["image_id"].astype(str)
    already = 0
    restored = 0
    portable_present = 0
    restored_identities: list[dict[str, Any]] = []

    for m, s, f0 in sorted(GOOD38):
        ok, _, _, _ = base.direct_trial_candidate(locked, m, s, f0)
        trial = base.trial_path(m, s, f0)
        if ok:
            already += 1
            if (trial / "best_model_portable_fp16.pt").is_file():
                portable_present += 1
            continue

        pp = pred[
            (pred["model_key"] == m)
            & (pred["seed"] == s)
            & (pred["outer_fold"] == f0 + 1)
        ].copy()
        mm = met[
            (met["model_key"] == m)
            & (met["seed"] == s)
            & (met["outer_fold"] == f0 + 1)
        ]
        if len(mm) != 1:
            raise RuntimeError(f"GOOD38 metric row count invalid for {(m,s,f0+1)}: {len(mm)}")

        reference = locked[["image_id", "group_id", "label", "label_name", "relative_path"]].copy()
        reference["image_id"] = reference["image_id"].astype(str)
        for column in ("group_id", "label", "label_name", "relative_path"):
            if column not in pp.columns:
                pp = pp.merge(reference[["image_id", column]], on="image_id", how="left", validate="many_to_one")

        valid, reason, normalized = base.validate_predictions(pp, locked, m, s, f0)
        if not valid or normalized is None:
            raise RuntimeError(f"GOOD38 locked validation failed for {(m,s,f0+1)}: {reason}")
        metric = metric_record_from_row(mm.iloc[0], base)
        ok_metric, metric_reason = base.complete_identity_valid(metric, m, s, f0)
        if not ok_metric:
            raise RuntimeError(f"GOOD38 metric identity failed for {(m,s,f0+1)}: {metric_reason}")

        if trial.exists():
            quota_aware_quarantine(
                base,
                trial,
                (m, s, f0),
                "replaced_by_cryptographically_verified_GOOD38_evidence",
            )
        trial.mkdir(parents=True, exist_ok=True)
        base.write_json_atomic(trial / "COMPLETE.json", metric)
        base.write_csv_atomic(trial / "test_predictions.csv", normalized)
        verification = {
            "status": "PASS_EVIDENCE_ONLY_RECOVERY",
            "version": VERSION,
            "model_key": m,
            "seed": s,
            "outer_fold": f0 + 1,
            "locked_split_sha256": locked_sha,
            "good38_package_sha256": GOOD38_PACKAGE_SHA256,
            "predictions_sha256": sha256_file(trial / "test_predictions.csv"),
            "metrics_sha256": sha256_file(trial / "COMPLETE.json"),
            "portable_checkpoint_present": False,
            "training_performed": False,
            "recovered_at_ns": time.time_ns(),
        }
        base.write_json_atomic(trial / "TRIAL_VERIFICATION.json", verification)
        base.write_json_atomic(
            trial / "REPAIR45_RECOVERY.json",
            {
                **verification,
                "source": bundle["source"],
            },
        )
        restored += 1
        restored_identities.append({"model_key": m, "seed": s, "outer_fold": f0 + 1})

    failures = []
    for m, s, f0 in sorted(GOOD38):
        ok, reason, _, _ = base.direct_trial_candidate(locked, m, s, f0)
        if not ok:
            failures.append((m, s, f0 + 1, reason))
    if failures:
        raise RuntimeError(f"GOOD38 final evidence gate failed: {failures[:5]}")

    portable_present = sum(
        int((base.trial_path(m, s, f0) / "best_model_portable_fp16.pt").is_file())
        for m, s, f0 in GOOD38
    )
    summary = {
        "valid_good_trials": 38,
        "already_valid": already,
        "restored_without_training": restored,
        "portable_checkpoints_present_among_good38": portable_present,
        "evidence_only_without_portable": 38 - portable_present,
        "restored_identities": restored_identities,
        "source": bundle["source"],
    }
    print(
        "GOOD38 gate: PASS | valid=38 | "
        f"already={already} restored_without_training={restored} "
        f"portable_present={portable_present}/38"
    )
    return summary


def install_runtime_train_wrapper() -> None:
    from src import reliable_runner_v2 as rr

    if getattr(rr.train_trial, "_repair45_v4_wrapped", False):
        return
    original = rr.train_trial

    def wrapped_train_trial(*args, **kwargs):
        train_df = kwargs["train_df"]
        selection_df = kwargs["selection_df"]
        calibration_df = kwargs["calibration_df"]
        trial = Path(kwargs["trial"])
        model_key = str(kwargs["model_key"])
        seed = int(kwargs["seed"])
        fold = int(kwargs["fold"])

        frames = []
        for role, frame in (
            ("train", train_df),
            ("selection", selection_df),
            ("calibration", calibration_df),
        ):
            x = frame[["image_id", "group_id"]].copy()
            x["role"] = role
            frames.append(x)
        roles = pd.concat(frames, ignore_index=True)
        roles["image_id"] = roles["image_id"].astype(str)
        roles["group_id"] = roles["group_id"].astype(str)
        roles = roles.sort_values(["image_id", "role"]).reset_index(drop=True)
        payload = "\n".join(
            f"{r.image_id},{r.group_id},{r.role}" for r in roles.itertuples(index=False)
        )
        inner_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        context_path = trial / "REPAIR45_INNER_CONTEXT.json"
        if context_path.is_file():
            old = json.loads(context_path.read_text(encoding="utf-8"))
            if (
                old.get("inner_partition_sha256") != inner_sha
                or old.get("model_key") != model_key
                or int(old.get("seed")) != seed
                or int(old.get("outer_fold")) != fold + 1
            ):
                raise RuntimeError(
                    f"Inner-partition resume context mismatch for {(model_key,seed,fold+1)}"
                )
        context_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = context_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "model_key": model_key,
                    "seed": seed,
                    "outer_fold": fold + 1,
                    "inner_partition_sha256": inner_sha,
                    "train_n": int(len(train_df)),
                    "selection_n": int(len(selection_df)),
                    "calibration_n": int(len(calibration_df)),
                    "updated_at_ns": time.time_ns(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, context_path)
        print("Inner partition fingerprint:", inner_sha[:16], "for", model_key, seed, fold + 1)
        return original(*args, **kwargs)

    wrapped_train_trial._repair45_v4_wrapped = True
    rr.train_trial = wrapped_train_trial


def install_patches(base) -> None:
    original_ensure_drive = base.ensure_drive
    original_validate = base.validate_locked_manifest
    original_audit = base.audit_and_recover
    state = {"recovery_done": False}

    def ensure_drive_v4():
        original_ensure_drive()
        drive_capacity_probe(base)

    def validate_v4():
        if state["recovery_done"]:
            return original_validate()

        recovery_report = base.RUN_ROOT / "REPAIR45_V4_RECOVERY.json"
        if base.LOCKED_SPLIT.is_file() and recovery_report.is_file():
            try:
                report = json.loads(recovery_report.read_text(encoding="utf-8"))
                if (
                    report.get("version") == VERSION
                    and report.get("status") == "PASS"
                    and report.get("good38_package_sha256") == GOOD38_PACKAGE_SHA256
                    and report.get("evidence_manifest_exact_1055_match") is True
                ):
                    locked, locked_sha = original_validate()
                    failures = []
                    for m, s, f in sorted(GOOD38):
                        ok, reason, _, _ = base.direct_trial_candidate(locked, m, s, f)
                        if not ok:
                            failures.append((m, s, f + 1, reason))
                    if not failures:
                        state["recovery_done"] = True
                        install_runtime_train_wrapper()
                        print("V4 recovery state reuse: PASS | lock + GOOD38 already verified")
                        return locked, locked_sha
                    print("Existing V4 report found but GOOD38 is incomplete; rebuilding recovery state.")
            except Exception as exc:
                print("Existing V4 recovery state rejected:", type(exc).__name__, exc)

        bundle = find_or_upload_good38_bundle()
        reconstructed, proof = recover_original_order_and_prove(bundle)
        base.LOCKED_SPLIT.parent.mkdir(parents=True, exist_ok=True)
        base.write_csv_atomic(base.LOCKED_SPLIT, reconstructed)
        locked, locked_sha = original_validate()
        good_summary = restore_good38(base, bundle, locked, locked_sha)
        install_runtime_train_wrapper()
        base.write_json_atomic(
            recovery_report,
            {
                "version": VERSION,
                "status": "PASS",
                "good38_package_sha256": GOOD38_PACKAGE_SHA256,
                "good38_source": bundle["source"],
                "locked_split_sha256": locked_sha,
                "evidence_manifest_exact_1055_match": True,
                "split_proof": proof,
                "good38": good_summary,
                "training_performed_during_recovery": False,
                "created_at_ns": time.time_ns(),
            },
        )
        state["recovery_done"] = True
        print("V4 pre-GPU recovery/proof: PASS")
        return locked, locked_sha

    def audit_v4(locked, locked_sha):
        audit, protected, train_list = original_audit(locked, locked_sha)
        unauthorized = sorted(set(train_list) - BAD7)
        if unauthorized:
            raise RuntimeError(
                "SAFETY BLOCK: reconciliation attempted to authorize training outside the historical BAD7 set: "
                + repr(unauthorized)
            )
        if len(train_list) > 7:
            raise RuntimeError(f"SAFETY BLOCK: train list has {len(train_list)} identities, maximum is 7")
        print("Training authorization gate: PASS")
        print("Protected GOOD/valid trials:", len(protected))
        print("GPU-authorized identities:", len(train_list))
        for identity in train_list:
            print("  -", identity[0], "seed", identity[1], "fold", identity[2] + 1)
        return audit, protected, train_list

    base.ensure_drive = ensure_drive_v4
    base.validate_locked_manifest = validate_v4
    base.audit_and_recover = audit_v4
    base.quarantine_trial = lambda trial, identity, reason: quota_aware_quarantine(base, trial, identity, reason)
    base.VERSION = VERSION


def main() -> None:
    print("=" * 96)
    print(VERSION)
    print("Safety order: Drive quota probe -> exact GOOD38 package -> deterministic 1055/1055 split/order proof")
    print("             -> restore GOOD38 without training -> audit all 45 -> GPU only verified remaining BAD7")
    print("No GPU training can begin before every preflight gate above passes.")
    print("=" * 96)
    base = load_base_module()
    print("Pinned base Repair45 source SHA256: PASS")
    install_patches(base)
    base.main()


if __name__ == "__main__":
    main()
