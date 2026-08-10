from __future__ import annotations

import hashlib
import importlib.util
import inspect
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

VERSION = "DFU_REPAIR45_GOOD38_RUNTIME_CERT_V8_20260811"
V4_COMMIT = "d318d7aec68604e2e3f9a2c8d529910e41e1340d"
V4_GIT_BLOB_SHA1 = "739f9598ee3fbe6f1a340fc1f59bacd254063e1e"
V4_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/"
    + V4_COMMIT
    + "/scripts/dfu_repair45_good38_proven_v4.py"
)
EVIDENCE_ZIP_NAME = "DFU_GOOD38_RECOVERY_EVIDENCE.zip"
LOCAL_DATASET_PROOF_ROOT = Path("/content/dfu_repair45_v8_dataset_proof")
ROW_ORDER_RANDOM_SEEDS = (8101, 8102, 8103, 8104, 8105, 8106)


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
    path = Path("/content/dfu_repair45_good38_proven_v4_for_v8.py")
    path.write_bytes(raw)
    compile(raw.decode("utf-8"), str(path), "exec")
    spec = importlib.util.spec_from_file_location(
        "dfu_repair45_good38_proven_v4_for_v8", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create V4 module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v4 = load_v4_module()
v4.VERSION = VERSION


def parse_exact_good38_zip(raw: bytes, source: str) -> dict[str, Any]:
    if sha256_bytes(raw) != v4.GOOD38_PACKAGE_SHA256:
        raise ValueError(
            "GOOD38 ZIP SHA-256 does not match the preserved evidence package"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            required = {"MANIFEST.json", "good38_predictions.csv", "good38_metrics.csv"}
            names = set(zf.namelist())
            if not required.issubset(names):
                raise ValueError(f"GOOD38 ZIP missing members: {sorted(required - names)}")
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


def find_or_upload_good38_bundle_v8() -> dict[str, Any]:
    candidates = [
        Path("/content") / EVIDENCE_ZIP_NAME,
        Path("/content/drive/MyDrive") / EVIDENCE_ZIP_NAME,
        Path("/content/drive/MyDrive/DFU-ImageGuard") / EVIDENCE_ZIP_NAME,
        Path("/content/drive/MyDrive/DFU-ImageGuard/evidence") / EVIDENCE_ZIP_NAME,
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            bundle = parse_exact_good38_zip(path.read_bytes(), str(path))
            print("GOOD38 evidence ZIP: PASS", path)
            return bundle
        except Exception as exc:
            print("Rejected GOOD38 ZIP:", path, type(exc).__name__, exc)

    print("Upload the exact DFU_GOOD38_RECOVERY_EVIDENCE.zip file.")
    from google.colab import files

    uploaded = files.upload()
    failures = []
    for name, data in uploaded.items():
        try:
            bundle = parse_exact_good38_zip(data, f"manual_upload:{name}")
            print("GOOD38 manual upload: PASS", name)
            return bundle
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "No valid GOOD38 evidence ZIP was supplied. GPU remains blocked. "
        + " | ".join(failures[:3])
    )


def normalized_historical_lock(bundle: dict[str, Any]) -> pd.DataFrame:
    locked = v4.evidence_lock_from_good38(bundle).copy()
    required = {
        "image_id", "group_id", "label", "label_name", "relative_path", "outer_fold"
    }
    missing = sorted(required - set(locked.columns))
    if missing:
        raise RuntimeError(f"GOOD38 historical lock missing columns: {missing}")

    locked["image_id"] = locked["image_id"].astype(str)
    locked["group_id"] = locked["group_id"].astype(str)
    locked["label"] = pd.to_numeric(locked["label"], errors="raise").astype(int)
    locked["label_name"] = locked["label_name"].astype(str)
    locked["relative_path"] = locked["relative_path"].astype(str)
    locked["outer_fold"] = pd.to_numeric(
        locked["outer_fold"], errors="raise"
    ).astype(int)

    if len(locked) != v4.EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Historical lock rows={len(locked)} != {v4.EXPECTED_IMAGE_COUNT}"
        )
    if locked["image_id"].nunique() != v4.EXPECTED_IMAGE_COUNT:
        raise RuntimeError("Historical lock has duplicate/missing image IDs")

    sizes = {
        int(f) + 1: int(n)
        for f, n in locked.groupby("outer_fold").size().sort_index().items()
    }
    if sizes != v4.EXPECTED_FOLD_SIZES_ONE:
        raise RuntimeError(f"Historical fold sizes changed: {sizes}")
    if locked.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("Historical duplicate group crosses outer folds")
    if locked.groupby("outer_fold")["label"].nunique().min() != 2:
        raise RuntimeError("A historical outer fold is missing a class")
    if (locked.groupby("group_id")["label"].nunique() > 1).any():
        raise RuntimeError("A historical duplicate group has conflicting labels")

    pred = v4.normalize_identity_frame(bundle["pred"])
    pred["image_id"] = pred["image_id"].astype(str)
    for model, seed, fold0 in sorted(v4.GOOD38):
        got = set(
            pred.loc[
                (pred["model_key"] == model)
                & (pred["seed"] == seed)
                & (pred["outer_fold"] == fold0 + 1),
                "image_id",
            ]
        )
        expected = set(
            locked.loc[locked["outer_fold"] == fold0, "image_id"].astype(str)
        )
        if got != expected:
            raise RuntimeError(
                f"GOOD38 membership mismatch for {(model, seed, fold0 + 1)}: "
                f"missing={len(expected - got)} unexpected={len(got - expected)}"
            )

    return locked.sort_values("image_id", kind="stable").reset_index(drop=True)


def validate_current_pinned_dataset(locked: pd.DataFrame) -> dict[str, Any]:
    shutil.rmtree(LOCAL_DATASET_PROOF_ROOT, ignore_errors=True)
    LOCAL_DATASET_PROOF_ROOT.mkdir(parents=True, exist_ok=True)

    from src import reliable_runner_v2 as rr
    from src.config_data import build_manifest, download_dataset, seed_everything

    settings = rr.ReliableSettingsV2(
        run_id="REPAIR45_V8_LOCAL_DATASET_PROOF",
        drive_root=str(LOCAL_DATASET_PROOF_ROOT),
        backup_root=str(LOCAL_DATASET_PROOF_ROOT / "backup"),
        source_commit="349143b4d8b16f885adce3559542f6c202a2bca1",
    )
    cfg = rr.build_config(settings)
    cfg.DRIVE_ROOT = str(LOCAL_DATASET_PROOF_ROOT)
    cfg.LOCAL_FALLBACK_ROOT = str(LOCAL_DATASET_PROOF_ROOT)
    cfg.MOUNT_DRIVE = False
    cfg.ALLOW_LOCAL_FALLBACK = True
    cfg.NUM_WORKERS = min(2, int(cfg.NUM_WORKERS))

    dirs = {"root": LOCAL_DATASET_PROOF_ROOT}
    for name in (
        "tables", "figures", "models", "xai", "predictions",
        "logs", "configs", "manifests", "cache"
    ):
        dirs[name] = LOCAL_DATASET_PROOF_ROOT / name
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    print("Validating pinned Kaggle dataset against historical GOOD38 membership...")
    seed_everything(cfg.SEED)
    dataset_root = download_dataset(cfg, dirs)
    manifest = build_manifest(dataset_root, cfg, dirs).copy()
    manifest["image_id"] = manifest["image_id"].astype(str)

    if manifest["image_id"].duplicated().any():
        raise RuntimeError("Current pinned manifest contains duplicate image IDs")
    historical_ids = set(locked["image_id"])
    current_ids = set(manifest["image_id"])
    if current_ids != historical_ids:
        raise RuntimeError(
            "Current pinned dataset image set differs from GOOD38 evidence: "
            f"missing={len(historical_ids - current_ids)} "
            f"unexpected={len(current_ids - historical_ids)}"
        )

    check = locked[
        ["image_id", "label", "label_name", "relative_path"]
    ].merge(
        manifest[
            ["image_id", "label", "label_name", "relative_path", "file_sha256"]
        ],
        on="image_id",
        how="left",
        suffixes=("_historical", "_current"),
        validate="one_to_one",
    )
    mismatch = {}
    for col in ("label", "label_name", "relative_path"):
        n = int(
            (
                check[f"{col}_historical"].astype(str)
                != check[f"{col}_current"].astype(str)
            ).sum()
        )
        if n:
            mismatch[col] = n
    if mismatch:
        raise RuntimeError(
            "Current pinned dataset metadata differs from GOOD38 evidence: "
            + json.dumps(mismatch, sort_keys=True)
        )
    if check["file_sha256"].isna().any():
        raise RuntimeError("Current pinned dataset contains an unreadable/unhashed image")

    fingerprint = hashlib.sha256(
        "\n".join(
            f"{r.image_id},{r.file_sha256}"
            for r in check[["image_id", "file_sha256"]]
            .sort_values("image_id")
            .itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()

    print("Pinned dataset verification: PASS | 1055/1055 IDs + labels + paths")
    return {
        "status": "PASS",
        "image_count": int(len(check)),
        "current_dataset_file_sha256_fingerprint": fingerprint,
    }


def inner_map(rr, cfg, data: pd.DataFrame, fold0: int) -> dict[str, tuple[int, str]]:
    outer_train = data.loc[data["outer_fold"].astype(int) != int(fold0)].copy()
    inner = rr.make_inner_partition(outer_train, cfg, int(fold0))
    if inner["image_id"].duplicated().any():
        raise RuntimeError(f"Fold-{fold0 + 1} inner partition duplicated image IDs")
    mapping = {
        str(r.image_id): (int(r.inner_fold), str(r.inner_role))
        for r in inner[["image_id", "inner_fold", "inner_role"]].itertuples(index=False)
    }
    if set(mapping) != set(outer_train["image_id"].astype(str)):
        raise RuntimeError(f"Fold-{fold0 + 1} inner partition lost/added images")
    for role in ("train", "selection", "calibration"):
        part = inner.loc[inner["inner_role"] == role]
        if part.empty or part["label"].nunique() != 2:
            raise RuntimeError(
                f"Fold-{fold0 + 1} role {role} is empty or class-incomplete"
            )
    roles = {
        role: set(inner.loc[inner["inner_role"] == role, "group_id"].astype(str))
        for role in ("train", "selection", "calibration")
    }
    if (
        roles["train"] & roles["selection"]
        or roles["train"] & roles["calibration"]
        or roles["selection"] & roles["calibration"]
    ):
        raise RuntimeError(f"Fold-{fold0 + 1} inner duplicate-group leakage")
    return mapping


def row_orders(locked: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    x = locked.copy().reset_index(drop=True)
    out = [
        ("image_id", x.sort_values("image_id", kind="stable").reset_index(drop=True)),
        (
            "image_id_reverse",
            x.sort_values("image_id", ascending=False, kind="stable").reset_index(drop=True),
        ),
        (
            "group_image",
            x.sort_values(["group_id", "image_id"], kind="stable").reset_index(drop=True),
        ),
        (
            "label_group_image",
            x.sort_values(["label", "group_id", "image_id"], kind="stable").reset_index(drop=True),
        ),
        (
            "relative_path",
            x.sort_values("relative_path", kind="stable").reset_index(drop=True),
        ),
    ]
    for seed in ROW_ORDER_RANDOM_SEEDS:
        out.append(
            (
                f"random_{seed}",
                x.sample(frac=1.0, random_state=seed).reset_index(drop=True),
            )
        )
    return out


def runtime_inner_invariance_certificate(locked: pd.DataFrame) -> dict[str, Any]:
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold
    from src import reliable_runner_v2 as rr

    settings = rr.ReliableSettingsV2(
        run_id="REPAIR45_V8_SPLIT_CERT",
        drive_root="/content/dfu_repair45_v8_split_cert",
        backup_root="/content/dfu_repair45_v8_split_cert_backup",
        source_commit="349143b4d8b16f885adce3559542f6c202a2bca1",
    )
    cfg = rr.build_config(settings)
    variants = row_orders(locked)
    comparisons = 0
    fold_hashes = {}

    for fold0 in range(5):
        reference = inner_map(rr, cfg, variants[0][1], fold0)
        fold_hashes[str(fold0 + 1)] = hashlib.sha256(
            "\n".join(
                f"{image_id},{inner_fold},{role}"
                for image_id, (inner_fold, role) in sorted(reference.items())
            ).encode("utf-8")
        ).hexdigest()

        for name, variant in variants[1:]:
            candidate = inner_map(rr, cfg, variant, fold0)
            comparisons += 1
            if candidate != reference:
                changed = sorted(
                    image_id
                    for image_id in set(reference) | set(candidate)
                    if reference.get(image_id) != candidate.get(image_id)
                )
                raise RuntimeError(
                    "ROW-ORDER INVARIANCE FAILED. GPU remains blocked. "
                    f"fold={fold0 + 1} variant={name} "
                    f"changed_images={len(changed)} first={changed[:5]}"
                )

    try:
        source = inspect.getsource(StratifiedGroupKFold._iter_test_indices)
        source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    except Exception:
        source_sha = None

    print(
        "Runtime inner-split certificate: PASS | "
        f"sklearn={sklearn.__version__} | "
        f"{comparisons}/{comparisons} exact image->inner_fold/role comparisons"
    )
    return {
        "status": "PASS",
        "sklearn_version": str(sklearn.__version__),
        "stratified_group_kfold_source_sha256": source_sha,
        "comparisons": int(comparisons),
        "exact_inner_assignment_invariant": True,
        "fold_assignment_sha256": fold_hashes,
    }


def recover_locked_v8(bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    locked = normalized_historical_lock(bundle)

    hist_bytes = urllib.request.urlopen(v4.HISTORICAL_FOLD1_URL, timeout=120).read()
    if sha256_bytes(hist_bytes) != v4.HISTORICAL_FOLD1_SHA256:
        raise RuntimeError("Historical Fold-1 evidence SHA-256 mismatch")
    hist = pd.read_csv(io.BytesIO(hist_bytes))
    hist_ids = set(hist["image_id"].astype(str))
    fold1_ids = set(locked.loc[locked["outer_fold"] == 0, "image_id"].astype(str))
    if len(hist_ids) != 209 or hist_ids != fold1_ids:
        raise RuntimeError(
            f"Historical Fold-1 proof failed: historical={len(hist_ids)} "
            f"recovered={len(fold1_ids)} overlap={len(hist_ids & fold1_ids)}"
        )

    dataset_proof = validate_current_pinned_dataset(locked)
    inner_certificate = runtime_inner_invariance_certificate(locked)
    canonical = locked.sort_values("image_id", kind="stable").reset_index(drop=True)

    payload = "\n".join(
        f"{r.image_id},{r.group_id},{int(r.outer_fold)},{r.relative_path}"
        for r in canonical[
            ["image_id", "group_id", "outer_fold", "relative_path"]
        ].itertuples(index=False)
    )
    proof = {
        "status": "PASS",
        "method": "GOOD38_membership_plus_dataset_match_plus_runtime_inner_invariance",
        "outer_folds_regenerated": False,
        "duplicate_groups_regenerated": False,
        "good38_prediction_row_order_used": False,
        "historical_group_ids_used": True,
        "historical_outer_folds_used": True,
        "canonical_repair_row_order": "image_id_ascending",
        "historical_minibatch_order_claimed": False,
        "historical_fold1_match": "209/209",
        "dataset_proof": dataset_proof,
        "inner_split_certificate": inner_certificate,
        "canonical_lock_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }

    print(
        "Locked split recovery: PASS | historical 1055 image/group/fold membership | "
        "current dataset match | Fold-1 209/209"
    )
    print(
        "Exact outer and inner assignments: PROVEN. "
        "BAD7 training uses deterministic canonical image_id row order."
    )
    return canonical, proof


v4.find_or_upload_good38_bundle = find_or_upload_good38_bundle_v8
v4.recover_original_order_and_prove = recover_locked_v8
v4.VERSION = VERSION

print(VERSION)
print("Pinned V4 base module Git blob: PASS", V4_GIT_BLOB_SHA1)
print("No GOOD38 prediction row-order assumption.")
print("No duplicate-group regeneration. No outer-fold regeneration.")
print("GPU remains blocked until dataset and runtime inner-split certificates pass.")
v4.main()
