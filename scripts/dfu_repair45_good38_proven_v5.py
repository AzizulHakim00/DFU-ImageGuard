from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

VERSION = "DFU_REPAIR45_GOOD38_PROVEN_V5_20260810"
V4_COMMIT = "d318d7aec68604e2e3f9a2c8d529910e41e1340d"
V4_GIT_BLOB_SHA1 = "739f9598ee3fbe6f1a340fc1f59bacd254063e1e"
V4_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/"
    + V4_COMMIT
    + "/scripts/dfu_repair45_good38_proven_v4.py"
)
ZIP_HINT_NAMES = {
    "DFU_GOOD38_RECOVERY_EVIDENCE.zip",
    "GOOD38_RECOVERY_EVIDENCE.zip",
}


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1((f"blob {len(raw)}\0").encode("ascii") + raw).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_v4_module():
    raw = urllib.request.urlopen(V4_URL, timeout=120).read()
    actual = git_blob_sha1(raw)
    if actual != V4_GIT_BLOB_SHA1:
        raise RuntimeError(
            f"Pinned V4 Git blob mismatch: expected={V4_GIT_BLOB_SHA1} actual={actual}"
        )
    path = Path("/content/dfu_repair45_good38_proven_v4_for_v5.py")
    path.write_bytes(raw)
    compile(raw.decode("utf-8"), str(path), "exec")
    spec = importlib.util.spec_from_file_location("dfu_repair45_good38_proven_v4_for_v5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create V4 module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v4 = load_v4_module()
v4.VERSION = VERSION


def parse_direct_good38_zip(raw: bytes, source: str) -> dict[str, Any]:
    """Accept the exact embedded GOOD38 evidence ZIP or an equivalent wrapper ZIP."""
    candidates: list[tuple[str, bytes]] = [(source, raw)]
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            names = set(zf.namelist())
            required = {"MANIFEST.json", "good38_predictions.csv", "good38_metrics.csv"}
            if not required.issubset(names):
                for name in zf.namelist():
                    low = name.lower()
                    if low.endswith(".zip") and "good38" in low:
                        candidates.append((f"{source}::{name}", zf.read(name)))
                    elif low.endswith(".ipynb") and "good38" in low:
                        try:
                            return v4.parse_good38_notebook(zf.read(name), f"{source}::{name}")
                        except Exception:
                            pass
    except zipfile.BadZipFile as exc:
        raise ValueError(f"uploaded .zip is unreadable: {exc}") from exc

    errors: list[str] = []
    for candidate_source, package in candidates:
        try:
            if sha256_bytes(package) != v4.GOOD38_PACKAGE_SHA256:
                raise ValueError(
                    "GOOD38 ZIP SHA-256 does not match the preserved evidence package"
                )
            with zipfile.ZipFile(io.BytesIO(package), "r") as zf:
                required = {"MANIFEST.json", "good38_predictions.csv", "good38_metrics.csv"}
                names = set(zf.namelist())
                if not required.issubset(names):
                    raise ValueError(f"GOOD38 ZIP missing members: {sorted(required-names)}")
                manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))
                pred_bytes = zf.read("good38_predictions.csv")
                metric_bytes = zf.read("good38_metrics.csv")
            if sha256_bytes(pred_bytes) != str(manifest.get("predictions_sha256")):
                raise ValueError("GOOD38 predictions inner SHA-256 mismatch")
            if sha256_bytes(metric_bytes) != str(manifest.get("metrics_sha256")):
                raise ValueError("GOOD38 metrics inner SHA-256 mismatch")
            pred = pd.read_csv(io.BytesIO(pred_bytes))
            met = pd.read_csv(io.BytesIO(metric_bytes))
            v4.validate_good38_tables(pred, met)
            return {
                "source": candidate_source,
                "package_sha256": v4.GOOD38_PACKAGE_SHA256,
                "pred": pred,
                "met": met,
                "manifest": manifest,
            }
        except Exception as exc:
            errors.append(f"{candidate_source}: {type(exc).__name__}: {exc}")
    raise ValueError("No valid preserved GOOD38 package found in ZIP. " + " | ".join(errors[:3]))


def parse_good38_any(raw: bytes, source: str, name: str = "") -> dict[str, Any]:
    low = (name or source).lower()
    if low.endswith(".ipynb"):
        return v4.parse_good38_notebook(raw, source)
    if low.endswith(".zip") or raw[:4] == b"PK\x03\x04":
        return parse_direct_good38_zip(raw, source)
    nb_error = None
    try:
        return v4.parse_good38_notebook(raw, source)
    except Exception as exc:
        nb_error = exc
    try:
        return parse_direct_good38_zip(raw, source)
    except Exception as zip_exc:
        raise ValueError(
            f"unsupported GOOD38 evidence format; notebook={type(nb_error).__name__}: {nb_error}; "
            f"zip={type(zip_exc).__name__}: {zip_exc}"
        ) from zip_exc


def find_or_upload_good38_bundle_v5() -> dict[str, Any]:
    direct = [
        Path("/content") / v4.GOOD38_NOTEBOOK_NAME,
        Path("/content") / "DFU_GOOD38_RECOVERY_EVIDENCE.zip",
        Path("/content/drive/MyDrive") / v4.GOOD38_NOTEBOOK_NAME,
        Path("/content/drive/MyDrive") / "DFU_GOOD38_RECOVERY_EVIDENCE.zip",
        Path("/content/drive/MyDrive/DFU-ImageGuard") / v4.GOOD38_NOTEBOOK_NAME,
        Path("/content/drive/MyDrive/DFU-ImageGuard") / "DFU_GOOD38_RECOVERY_EVIDENCE.zip",
        Path("/content/drive/MyDrive/DFU-ImageGuard/evidence") / v4.GOOD38_NOTEBOOK_NAME,
        Path("/content/drive/MyDrive/DFU-ImageGuard/evidence") / "DFU_GOOD38_RECOVERY_EVIDENCE.zip",
    ]
    seen: set[str] = set()
    for path in direct:
        if not path.is_file() or str(path) in seen:
            continue
        seen.add(str(path))
        try:
            bundle = parse_good38_any(path.read_bytes(), str(path), path.name)
            print("GOOD38 evidence source: PASS", path)
            return bundle
        except Exception as exc:
            print("Rejected direct GOOD38 candidate:", path, type(exc).__name__, exc)

    roots = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
        Path("/content/drive/.shortcut-targets-by-id"),
    ]
    roots = [r for r in roots if r.exists()]
    print("Searching mounted Drive for GOOD38 notebook or evidence ZIP...")
    scanned = 0
    for root in roots:
        for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
            scanned += 1
            dirs[:] = [d for d in dirs if d not in {".Trash", ".cache", "__pycache__", ".ipynb_checkpoints", "node_modules"}]
            for name in names:
                low = name.lower()
                eligible = (
                    name == v4.GOOD38_NOTEBOOK_NAME
                    or name in ZIP_HINT_NAMES
                    or ("good38" in low and low.endswith((".ipynb", ".zip")))
                )
                if not eligible:
                    continue
                path = Path(current) / name
                if str(path) in seen:
                    continue
                seen.add(str(path))
                try:
                    bundle = parse_good38_any(path.read_bytes(), str(path), name)
                    print("GOOD38 evidence source: PASS", path)
                    return bundle
                except Exception as exc:
                    print("Rejected GOOD38 candidate:", path, type(exc).__name__, exc)
    print(f"Drive search complete; directories scanned={scanned}. No valid GOOD38 source found.")
    print("Upload either the exact GOOD38 self-contained notebook OR DFU_GOOD38_RECOVERY_EVIDENCE.zip.")
    from google.colab import files
    uploaded = files.upload()
    failures: list[str] = []
    for name, data in uploaded.items():
        try:
            bundle = parse_good38_any(data, f"manual_upload:{name}", name)
            print("GOOD38 manual upload: PASS", name)
            return bundle
        except Exception as exc:
            msg = f"{name}: {type(exc).__name__}: {exc}"
            failures.append(msg)
            print("Rejected upload:", msg)
    raise RuntimeError(
        "No valid GOOD38 evidence package was supplied. GPU remains blocked. "
        + " | ".join(failures[:3])
    )


def recover_original_cleaned_order_and_prove_v5(bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prove the historical split without regenerating outer folds.

    Regenerate only the pinned duplicate-group preprocessing locally. The group_id
    mapping must match the saved GOOD38 evidence for all 1,055 images. Outer-fold
    membership is copied only from the saved GOOD38 evidence.
    """
    evidence_lock = v4.evidence_lock_from_good38(bundle)
    root = Path("/content/dfu_repair45_v5_group_proof")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    from src import reliable_runner_v2 as rr
    from src.config_data import assign_duplicate_groups, build_manifest, download_dataset, seed_everything

    settings = rr.ReliableSettingsV2(
        run_id="REPAIR45_V5_LOCAL_GROUP_PROOF",
        drive_root=str(root),
        backup_root=str(root / "backup"),
        source_commit="349143b4d8b16f885adce3559542f6c202a2bca1",
    )
    cfg = rr.build_config(settings)
    cfg.DRIVE_ROOT = str(root)
    cfg.LOCAL_FALLBACK_ROOT = str(root)
    cfg.MOUNT_DRIVE = False
    cfg.ALLOW_LOCAL_FALLBACK = True
    cfg.NUM_WORKERS = min(2, int(cfg.NUM_WORKERS))

    names = ("tables", "figures", "models", "xai", "predictions", "logs", "configs", "manifests", "cache")
    dirs = {"root": root, **{name: root / name for name in names}}
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    print("Rebuilding pinned manifest + duplicate groups in /content for proof only...")
    print("IMPORTANT: outer folds are NOT regenerated.")
    seed_everything(cfg.SEED)
    dataset_root = download_dataset(cfg, dirs)
    manifest = build_manifest(dataset_root, cfg, dirs)
    cleaned = assign_duplicate_groups(manifest, cfg, dirs).copy()
    cleaned["image_id"] = cleaned["image_id"].astype(str)
    cleaned["group_id"] = cleaned["group_id"].astype(str)

    if len(cleaned) != v4.EXPECTED_IMAGE_COUNT or cleaned["image_id"].nunique() != v4.EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Pinned duplicate-group reconstruction has {len(cleaned)} rows / "
            f"{cleaned['image_id'].nunique()} unique IDs; expected {v4.EXPECTED_IMAGE_COUNT}"
        )

    ev = evidence_lock.copy()
    ev["image_id"] = ev["image_id"].astype(str)
    ev["group_id"] = ev["group_id"].astype(str)
    current_ids = set(cleaned["image_id"])
    evidence_ids = set(ev["image_id"])
    if current_ids != evidence_ids:
        raise RuntimeError(
            f"Pinned cleaned image set mismatch: missing={len(evidence_ids-current_ids)} "
            f"unexpected={len(current_ids-evidence_ids)}"
        )

    check = cleaned[["image_id", "group_id", "label", "label_name", "relative_path"]].merge(
        ev[["image_id", "group_id", "label", "label_name", "relative_path", "outer_fold"]],
        on="image_id",
        how="left",
        suffixes=("_current", "_evidence"),
        validate="one_to_one",
        sort=False,
    )
    mismatch: dict[str, int] = {}
    for col in ("group_id", "label", "label_name", "relative_path"):
        a = check[f"{col}_current"].astype(str)
        b = check[f"{col}_evidence"].astype(str)
        n = int((a != b).sum())
        if n:
            mismatch[col] = n
    if mismatch:
        raise RuntimeError(
            "Pinned duplicate-group reconstruction disagrees with GOOD38 evidence: "
            + json.dumps(mismatch, sort_keys=True)
        )

    fold_map = ev[["image_id", "outer_fold"]].copy()
    fold_map["outer_fold"] = pd.to_numeric(fold_map["outer_fold"], errors="raise").astype(int)
    locked = cleaned.merge(fold_map, on="image_id", how="left", validate="one_to_one", sort=False)
    if locked["outer_fold"].isna().any():
        raise RuntimeError("GOOD38 outer-fold map did not cover all cleaned rows")
    locked["outer_fold"] = locked["outer_fold"].astype(int)
    locked = locked.drop(columns=["image_path"], errors="ignore")

    sizes = {int(f) + 1: int(n) for f, n in locked.groupby("outer_fold").size().sort_index().items()}
    if sizes != v4.EXPECTED_FOLD_SIZES_ONE:
        raise RuntimeError(f"Recovered historical fold sizes changed: {sizes}")
    if locked.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("Recovered historical duplicate group crosses outer folds")

    pred = v4.normalize_identity_frame(bundle["pred"])
    pred["image_id"] = pred["image_id"].astype(str)
    fold_set_checks = 0
    for m, s, f0 in sorted(v4.GOOD38):
        got = set(
            pred.loc[
                (pred["model_key"] == m)
                & (pred["seed"] == s)
                & (pred["outer_fold"] == f0 + 1),
                "image_id",
            ]
        )
        expected = set(locked.loc[locked["outer_fold"] == f0, "image_id"].astype(str))
        if got != expected:
            raise RuntimeError(
                f"GOOD38 trial set proof failed for {(m,s,f0+1)}: "
                f"missing={len(expected-got)} unexpected={len(got-expected)}"
            )
        fold_set_checks += 1

    hist_bytes = urllib.request.urlopen(v4.HISTORICAL_FOLD1_URL, timeout=120).read()
    if sha256_bytes(hist_bytes) != v4.HISTORICAL_FOLD1_SHA256:
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
        f"{r.image_id},{r.group_id},{int(r.outer_fold)}"
        for r in locked[["image_id", "group_id", "outer_fold"]].itertuples(index=False)
    )
    group_payload = "\n".join(
        f"{r.image_id},{r.group_id}"
        for r in locked[["image_id", "group_id"]].itertuples(index=False)
    )
    proof = {
        "status": "PASS",
        "method": "GOOD38_outer_fold_map_plus_pinned_duplicate_group_reconstruction",
        "outer_folds_regenerated": False,
        "duplicate_groups_regenerated_for_validation_only": True,
        "exact_1055_image_match": True,
        "exact_1055_group_id_match": True,
        "exact_dataset_metadata_match": True,
        "good38_trial_fold_set_checks": fold_set_checks,
        "fold_sizes": sizes,
        "historical_fold1_match": "209/209",
        "cleaned_order_sha256": hashlib.sha256(order_payload.encode("utf-8")).hexdigest(),
        "group_map_sha256": hashlib.sha256(group_payload.encode("utf-8")).hexdigest(),
    }
    print(
        "Locked split/group/order proof: PASS | 1055/1055 image+group mapping | "
        "38/38 saved trial fold sets | historical Fold-1 209/209 | NO outer-fold regeneration"
    )
    return locked, proof


v4.find_or_upload_good38_bundle = find_or_upload_good38_bundle_v5
v4.recover_original_order_and_prove = recover_original_cleaned_order_and_prove_v5
v4.VERSION = VERSION

print(VERSION)
print("Pinned V4 base module Git blob: PASS", V4_GIT_BLOB_SHA1)
print("Fixes: accepts GOOD38 .ipynb or .zip; removes false prediction-row-order gate;")
print("       proves 1055/1055 duplicate-group mapping locally; outer folds remain evidence-locked.")
v4.main()
