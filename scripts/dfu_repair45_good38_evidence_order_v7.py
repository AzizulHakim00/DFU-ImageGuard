from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

VERSION = "DFU_REPAIR45_GOOD38_EVIDENCE_ORDER_V7_20260811"
V4_COMMIT = "d318d7aec68604e2e3f9a2c8d529910e41e1340d"
V4_GIT_BLOB_SHA1 = "739f9598ee3fbe6f1a340fc1f59bacd254063e1e"
V4_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/"
    + V4_COMMIT
    + "/scripts/dfu_repair45_good38_proven_v4.py"
)

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
    path = Path("/content/dfu_repair45_good38_proven_v4_for_v7.py")
    path.write_bytes(raw)
    compile(raw.decode("utf-8"), str(path), "exec")
    spec = importlib.util.spec_from_file_location("dfu_repair45_good38_proven_v4_for_v7", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create V4 module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

v4 = load_v4_module()
v4.VERSION = VERSION

def parse_direct_good38_zip(raw: bytes, source: str) -> dict[str, Any]:
    if sha256_bytes(raw) != v4.GOOD38_PACKAGE_SHA256:
        raise ValueError(
            "GOOD38 ZIP SHA-256 does not match the preserved evidence package"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            required = {"MANIFEST.json", "good38_predictions.csv", "good38_metrics.csv"}
            names = set(zf.namelist())
            if not required.issubset(names):
                raise ValueError(f"GOOD38 ZIP missing members: {sorted(required-names)}")
            manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))
            pred_bytes = zf.read("good38_predictions.csv")
            metric_bytes = zf.read("good38_metrics.csv")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"GOOD38 ZIP unreadable: {exc}") from exc
    if sha256_bytes(pred_bytes) != str(manifest.get("predictions_sha256")):
        raise ValueError("GOOD38 predictions inner SHA-256 mismatch")
    if sha256_bytes(metric_bytes) != str(manifest.get("metrics_sha256")):
        raise ValueError("GOOD38 metrics inner SHA-256 mismatch")
    pred = pd.read_csv(io.BytesIO(pred_bytes))
    met = pd.read_csv(io.BytesIO(metric_bytes))
    v4.validate_good38_tables(pred, met)
    return {
        "source": source,
        "package_sha256": v4.GOOD38_PACKAGE_SHA256,
        "pred": pred,
        "met": met,
        "manifest": manifest,
    }

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
                f"unsupported GOOD38 evidence format; "
                f"notebook={type(nb_exc).__name__}: {nb_exc}; "
                f"zip={type(zip_exc).__name__}: {zip_exc}"
            ) from zip_exc

def find_or_upload_good38_bundle_v7() -> dict[str, Any]:
    names = (
        v4.GOOD38_NOTEBOOK_NAME,
        "DFU_GOOD38_RECOVERY_EVIDENCE.zip",
        "GOOD38_RECOVERY_EVIDENCE.zip",
    )
    roots = [
        Path("/content"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
        Path("/content/drive/.shortcut-targets-by-id"),
    ]
    seen: set[str] = set()

    direct_dirs = [
        Path("/content"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/MyDrive/DFU-ImageGuard"),
        Path("/content/drive/MyDrive/DFU-ImageGuard/evidence"),
    ]
    for root in direct_dirs:
        for name in names:
            path = root / name
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))
            try:
                bundle = parse_good38_any(path.read_bytes(), str(path), path.name)
                print("GOOD38 evidence source: PASS", path)
                return bundle
            except Exception as exc:
                print("Rejected direct GOOD38 candidate:", path, type(exc).__name__, exc)

    print("Searching mounted Drive for GOOD38 notebook or evidence ZIP...")
    scanned = 0
    for root in roots[1:]:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
            scanned += 1
            dirs[:] = [
                d for d in dirs
                if d not in {".Trash", ".cache", "__pycache__", ".ipynb_checkpoints", "node_modules"}
            ]
            for name in files:
                low = name.lower()
                if not (
                    name == v4.GOOD38_NOTEBOOK_NAME
                    or ("good38" in low and low.endswith((".ipynb", ".zip")))
                ):
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

def _identity_sequence(pred: pd.DataFrame, identity: tuple[str, int, int]) -> list[str]:
    model, seed, fold0 = identity
    part = pred.loc[
        (pred["model_key"] == model)
        & (pred["seed"] == seed)
        & (pred["outer_fold"] == fold0 + 1)
    ]
    return part["image_id"].astype(str).tolist()

def _fold_sequences_from_evidence(pred: pd.DataFrame) -> dict[int, list[str]]:
    sequences: dict[int, list[str]] = {}
    for fold0 in range(5):
        identities = sorted(x for x in v4.GOOD38 if x[2] == fold0)
        if not identities:
            raise RuntimeError(f"GOOD38 has no preserved representative for Fold-{fold0+1}")
        reference = _identity_sequence(pred, identities[0])
        if len(reference) != v4.EXPECTED_FOLD_SIZES_ONE[fold0 + 1]:
            raise RuntimeError(
                f"GOOD38 Fold-{fold0+1} sequence length {len(reference)} "
                f"!= {v4.EXPECTED_FOLD_SIZES_ONE[fold0+1]}"
            )
        if len(reference) != len(set(reference)):
            raise RuntimeError(f"GOOD38 Fold-{fold0+1} sequence contains duplicate image IDs")
        for identity in identities[1:]:
            other = _identity_sequence(pred, identity)
            if other != reference:
                first = next(
                    (i for i, (a, b) in enumerate(zip(reference, other)) if a != b),
                    min(len(reference), len(other)),
                )
                raise RuntimeError(
                    f"GOOD38 preserved row order disagrees within Fold-{fold0+1}; "
                    f"{identities[0]} vs {identity}, first mismatch index={first}"
                )
        sequences[fold0] = reference
    return sequences

def _infer_historical_class_order(
    fold_sequences: dict[int, list[str]],
    evidence_lock: pd.DataFrame,
) -> tuple[int, int]:
    meta = evidence_lock.set_index("image_id")[["label", "relative_path"]].copy()
    meta.index = meta.index.astype(str)
    orders: list[tuple[int, int]] = []
    for fold0, seq in fold_sequences.items():
        part = meta.loc[seq]
        labels = pd.to_numeric(part["label"], errors="raise").astype(int).tolist()
        compressed: list[int] = []
        for label in labels:
            if not compressed or compressed[-1] != label:
                compressed.append(label)
        if len(compressed) != 2 or set(compressed) != {0, 1}:
            raise RuntimeError(
                f"Fold-{fold0+1} preserved prediction order is not compatible with "
                f"the pinned two-class block manifest order; label blocks={compressed}"
            )
        for label in compressed:
            paths = part.loc[
                pd.to_numeric(part["label"], errors="raise").astype(int) == label,
                "relative_path",
            ].astype(str).tolist()
            if paths != sorted(paths):
                raise RuntimeError(
                    f"Fold-{fold0+1} preserved prediction order is not path-sorted "
                    f"inside historical class {label}; exact original row order cannot be proven"
                )
        orders.append((int(compressed[0]), int(compressed[1])))
    if len(set(orders)) != 1:
        raise RuntimeError(f"Historical class-block order disagrees across folds: {orders}")
    return orders[0]

def recover_original_order_from_good38_v7(
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evidence_lock = v4.evidence_lock_from_good38(bundle).copy()
    evidence_lock["image_id"] = evidence_lock["image_id"].astype(str)
    evidence_lock["group_id"] = evidence_lock["group_id"].astype(str)
    evidence_lock["label"] = pd.to_numeric(
        evidence_lock["label"], errors="raise"
    ).astype(int)
    evidence_lock["outer_fold"] = pd.to_numeric(
        evidence_lock["outer_fold"], errors="raise"
    ).astype(int)
    if "relative_path" not in evidence_lock.columns:
        raise RuntimeError(
            "GOOD38 evidence lacks relative_path; exact historical row order cannot be proven"
        )
    if evidence_lock["relative_path"].isna().any():
        raise RuntimeError(
            "GOOD38 evidence contains missing relative_path; exact historical row order cannot be proven"
        )
    evidence_lock["relative_path"] = evidence_lock["relative_path"].astype(str)

    pred = v4.normalize_identity_frame(bundle["pred"])
    pred["image_id"] = pred["image_id"].astype(str)
    fold_sequences = _fold_sequences_from_evidence(pred)
    class_order = _infer_historical_class_order(fold_sequences, evidence_lock)

    blocks = []
    for label in class_order:
        block = evidence_lock.loc[evidence_lock["label"] == int(label)].copy()
        block = block.sort_values("relative_path", kind="stable")
        blocks.append(block)
    locked = pd.concat(blocks, ignore_index=True)

    if len(locked) != v4.EXPECTED_IMAGE_COUNT or locked["image_id"].nunique() != v4.EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Recovered lock has {len(locked)} rows / {locked['image_id'].nunique()} unique IDs; "
            f"expected {v4.EXPECTED_IMAGE_COUNT}"
        )

    exact_fold_order_checks = 0
    fold_order_sha: dict[str, str] = {}
    for fold0 in range(5):
        reconstructed = locked.loc[
            locked["outer_fold"] == fold0, "image_id"
        ].astype(str).tolist()
        preserved = fold_sequences[fold0]
        if reconstructed != preserved:
            first = next(
                (i for i, (a, b) in enumerate(zip(reconstructed, preserved)) if a != b),
                min(len(reconstructed), len(preserved)),
            )
            raise RuntimeError(
                f"Historical global-order reconstruction failed Fold-{fold0+1}; "
                f"first mismatch index={first}. GPU remains blocked."
            )
        exact_fold_order_checks += 1
        fold_order_sha[str(fold0 + 1)] = hashlib.sha256(
            "\n".join(preserved).encode("utf-8")
        ).hexdigest()

    sizes = {
        int(f) + 1: int(n)
        for f, n in locked.groupby("outer_fold").size().sort_index().items()
    }
    if sizes != v4.EXPECTED_FOLD_SIZES_ONE:
        raise RuntimeError(f"Recovered historical fold sizes changed: {sizes}")
    if locked.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("Historical GOOD38 duplicate group crosses outer folds")

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
        f"{r.image_id},{r.group_id},{int(r.outer_fold)},{r.relative_path}"
        for r in locked[["image_id", "group_id", "outer_fold", "relative_path"]].itertuples(index=False)
    )
    proof = {
        "status": "PASS",
        "method": "GOOD38_preserved_fold_row_sequences_plus_pinned_class_block_path_order",
        "outer_folds_regenerated": False,
        "duplicate_groups_regenerated": False,
        "resnet_or_embedding_proof_used": False,
        "exact_1055_image_match": True,
        "historical_group_labels_used_directly": True,
        "historical_class_block_order": list(class_order),
        "exact_saved_fold_order_checks": exact_fold_order_checks,
        "fold_order_sha256": fold_order_sha,
        "fold_sizes": sizes,
        "historical_fold1_match": "209/209",
        "locked_row_order_sha256": hashlib.sha256(
            row_payload.encode("utf-8")
        ).hexdigest(),
    }
    print(
        "Locked split/order proof: PASS | 1055/1055 images | "
        "5/5 preserved fold row sequences | historical GOOD38 group labels | "
        "Fold-1 209/209 | NO duplicate-group regeneration | NO outer-fold regeneration"
    )
    return locked, proof

v4.find_or_upload_good38_bundle = find_or_upload_good38_bundle_v7
v4.recover_original_order_and_prove = recover_original_order_from_good38_v7
v4.VERSION = VERSION

print(VERSION)
print("Pinned V4 base module Git blob: PASS", V4_GIT_BLOB_SHA1)
print("Recovery proof uses only preserved GOOD38 fold row sequences + historical paths/groups/folds.")
print("No ResNet18 duplicate reconstruction. No make_outer_folds(). No assign_duplicate_groups().")
v4.main()
