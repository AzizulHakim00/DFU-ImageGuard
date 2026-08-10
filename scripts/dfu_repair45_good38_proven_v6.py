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

VERSION = "DFU_REPAIR45_GOOD38_PROVEN_V6_20260810"
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
    path = Path("/content/dfu_repair45_good38_proven_v4_for_v6.py")
    path.write_bytes(raw)
    compile(raw.decode("utf-8"), str(path), "exec")
    spec = importlib.util.spec_from_file_location("dfu_repair45_good38_proven_v4_for_v6", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create V4 module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v4 = load_v4_module()
v4.VERSION = VERSION


def parse_direct_good38_zip(raw: bytes, source: str) -> dict[str, Any]:
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
                raise ValueError("GOOD38 ZIP SHA-256 does not match the preserved evidence package")
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
    try:
        return v4.parse_good38_notebook(raw, source)
    except Exception as nb_exc:
        try:
            return parse_direct_good38_zip(raw, source)
        except Exception as zip_exc:
            raise ValueError(
                f"unsupported GOOD38 evidence format; notebook={type(nb_exc).__name__}: {nb_exc}; "
                f"zip={type(zip_exc).__name__}: {zip_exc}"
            ) from zip_exc


def find_or_upload_good38_bundle_v6() -> dict[str, Any]:
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


def group_partition_signatures(frame: pd.DataFrame) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Canonicalize duplicate groups by their member image sets, not unstable DG labels."""
    x = frame[["image_id", "group_id"]].copy()
    x["image_id"] = x["image_id"].astype(str)
    x["group_id"] = x["group_id"].astype(str)
    groups: dict[str, list[str]] = {}
    image_to_signature: dict[str, str] = {}
    for group_id, part in x.groupby("group_id", sort=False):
        members = sorted(part["image_id"].tolist())
        signature = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()
        groups[signature] = members
        for image_id in members:
            image_to_signature[image_id] = signature
    return image_to_signature, groups


def recover_original_cleaned_order_and_prove_v6(bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    evidence_lock = v4.evidence_lock_from_good38(bundle)
    root = Path("/content/dfu_repair45_v6_group_partition_proof")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    from src import reliable_runner_v2 as rr
    from src.config_data import assign_duplicate_groups, build_manifest, download_dataset, seed_everything

    settings = rr.ReliableSettingsV2(
        run_id="REPAIR45_V6_LOCAL_GROUP_PARTITION_PROOF",
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

    print("Rebuilding pinned manifest + duplicate grouping structure in /content for proof only...")
    print("IMPORTANT: DGxxxx names are ignored because they are enumeration labels, not scientific IDs.")
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

    current_sig, current_groups = group_partition_signatures(cleaned)
    evidence_sig, evidence_groups = group_partition_signatures(ev)
    partition_mismatch = sorted(
        image_id for image_id in evidence_ids
        if current_sig.get(image_id) != evidence_sig.get(image_id)
    )
    if partition_mismatch:
        raise RuntimeError(
            "Duplicate-group PARTITION really changed (not just DG labels): "
            f"mismatched_images={len(partition_mismatch)} first={partition_mismatch[:5]} "
            f"current_groups={len(current_groups)} evidence_groups={len(evidence_groups)}"
        )
    if set(current_groups) != set(evidence_groups):
        raise RuntimeError(
            f"Duplicate-group member-set signatures differ despite per-image check: "
            f"current_groups={len(current_groups)} evidence_groups={len(evidence_groups)}"
        )

    check = cleaned[["image_id", "label", "label_name", "relative_path"]].merge(
        ev[["image_id", "label", "label_name", "relative_path", "outer_fold", "group_id"]],
        on="image_id",
        how="left",
        suffixes=("_current", "_evidence"),
        validate="one_to_one",
        sort=False,
    )
    mismatch: dict[str, int] = {}
    for col in ("label", "label_name", "relative_path"):
        a = check[f"{col}_current"].astype(str)
        b = check[f"{col}_evidence"].astype(str)
        n = int((a != b).sum())
        if n:
            mismatch[col] = n
    if mismatch:
        raise RuntimeError(
            "Pinned dataset metadata disagrees with GOOD38 evidence: "
            + json.dumps(mismatch, sort_keys=True)
        )

    # Preserve the deterministic cleaned row order, but restore the HISTORICAL group labels and outer-fold map.
    historical_map = ev[["image_id", "group_id", "outer_fold"]].copy()
    historical_map["outer_fold"] = pd.to_numeric(historical_map["outer_fold"], errors="raise").astype(int)
    locked = cleaned.drop(columns=["group_id", "image_path"], errors="ignore").merge(
        historical_map,
        on="image_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if locked[["group_id", "outer_fold"]].isna().any().any():
        raise RuntimeError("Historical GOOD38 group/fold map did not cover all 1055 cleaned rows")
    locked["group_id"] = locked["group_id"].astype(str)
    locked["outer_fold"] = locked["outer_fold"].astype(int)

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

    row_payload = "\n".join(
        f"{r.image_id},{r.group_id},{int(r.outer_fold)}"
        for r in locked[["image_id", "group_id", "outer_fold"]].itertuples(index=False)
    )
    partition_payload = "\n".join(sorted(evidence_groups))
    proof = {
        "status": "PASS",
        "method": "GOOD38_historical_group_labels_and_outer_folds_plus_partition_equivalence_proof",
        "outer_folds_regenerated": False,
        "duplicate_group_partition_regenerated_for_validation_only": True,
        "literal_regenerated_group_labels_compared": False,
        "historical_good38_group_labels_restored": True,
        "exact_1055_image_match": True,
        "exact_duplicate_partition_match": True,
        "duplicate_group_count": int(len(evidence_groups)),
        "exact_dataset_metadata_match": True,
        "good38_trial_fold_set_checks": int(fold_set_checks),
        "fold_sizes": sizes,
        "historical_fold1_match": "209/209",
        "locked_row_order_sha256": hashlib.sha256(row_payload.encode("utf-8")).hexdigest(),
        "duplicate_partition_sha256": hashlib.sha256(partition_payload.encode("utf-8")).hexdigest(),
    }
    print(
        "Locked split/group proof: PASS | 1055/1055 images | duplicate member partition identical | "
        "historical GOOD38 group labels restored | 38/38 trial fold sets | Fold-1 209/209 | NO outer-fold regeneration"
    )
    return locked, proof


v4.find_or_upload_good38_bundle = find_or_upload_good38_bundle_v6
v4.recover_original_order_and_prove = recover_original_cleaned_order_and_prove_v6
v4.VERSION = VERSION

print(VERSION)
print("Pinned V4 base module Git blob: PASS", V4_GIT_BLOB_SHA1)
print("Fixes: accepts GOOD38 .ipynb/.zip; compares duplicate-group MEMBER SETS, not unstable DG labels;")
print("       restores historical GOOD38 group_id values; outer folds remain evidence-locked.")
v4.main()
