from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "DFU_REPAIR45_RECONCILE_LOCKED_V1_20260810"
PINNED_CODE_COMMIT = "4472d66a4f918d42ce9cfdfb57ed6dd95bdb0f11"
ALGORITHM_SOURCE_COMMIT = "349143b4d8b16f885adce3559542f6c202a2bca1"
RUN_ID = "RELIABLE_DFU_CV_V3_MISSING38"
MODELS = ("convnextv2_tiny", "mobilenetv3_large", "densenet121")
SEEDS = (2026, 2027, 2028)
FOLDS_ZERO = (0, 1, 2, 3, 4)
EXPECTED = tuple((m, int(s), int(f)) for f in FOLDS_ZERO for s in SEEDS for m in MODELS)

DRIVE_ROOT = Path("/content/drive/MyDrive/DFU-ImageGuard")
BACKUP_ROOT = Path("/content/drive/MyDrive/DFU-ImageGuard-Backup")
RUN_ROOT = DRIVE_ROOT / "runs" / RUN_ID
LOCKED_SPLIT = RUN_ROOT / "manifests" / "locked_outer_fold_assignments.csv"
REPO_ROOT = Path("/content/DFU-ImageGuard-repair45-pinned")
ARCHIVE = Path("/content/DFU-ImageGuard-repair45-pinned.tar.gz")
EXTRACT_ROOT = Path("/content/DFU-ImageGuard-repair45-extract")
QUARANTINE_ROOT = RUN_ROOT / "quarantine" / "repair45"
RECON_CSV = RUN_ROOT / "tables" / "REPAIR45_RECONCILIATION.csv"
RECON_JSON = RUN_ROOT / "REPAIR45_RECONCILIATION.json"
PROGRESS_JSON = RUN_ROOT / "REPAIR45_PROGRESS.json"
FINAL_JSON = RUN_ROOT / "REPAIR45_FINAL_VERIFICATION.json"
PKL_PATH = RUN_ROOT / "reliable_dfu_reproducibility_repair45.pkl"
EXPORT_ZIP = RUN_ROOT / "DFU_REPAIR45_FINAL_EXPORT.zip"

if len(EXPECTED) != 45:
    raise RuntimeError(f"Protocol identity count changed: {len(EXPECTED)}")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=jsonable, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def ensure_drive() -> None:
    try:
        from google.colab import drive
    except Exception as exc:
        raise RuntimeError(f"This runner is intended for Google Colab: {exc}") from exc

    if not Path("/content/drive/MyDrive").is_dir():
        first_error = None
        try:
            drive.mount("/content/drive", force_remount=False)
        except Exception as exc:
            first_error = exc
            try:
                drive.flush_and_unmount()
            except Exception:
                pass
            try:
                drive.mount("/content/drive", force_remount=True)
            except Exception as exc2:
                raise RuntimeError(
                    f"Google Drive mount failed. initial={first_error!r}; retry={exc2!r}"
                ) from exc2

    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Google Drive mount is not visible at /content/drive/MyDrive")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    sentinel = RUN_ROOT / "REPAIR45_DRIVE_SENTINEL.txt"
    token = f"{VERSION}|{time.time_ns()}"
    sentinel.write_text(token, encoding="utf-8")
    if sentinel.read_text(encoding="utf-8") != token:
        raise RuntimeError("Google Drive write/read verification failed")
    print("Google Drive mount + write verification: PASS")


def ensure_dependencies() -> None:
    required = {
        "timm": "timm",
        "kagglehub": "kagglehub",
        "imagehash": "ImageHash",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
    }
    for module, package in required.items():
        if importlib.util.find_spec(module) is None:
            print(f"Installing missing dependency only: {package}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", package]
            )
    print("Dependency verification: PASS")


def prepare_pinned_source() -> None:
    shutil.rmtree(REPO_ROOT, ignore_errors=True)
    shutil.rmtree(EXTRACT_ROOT, ignore_errors=True)
    ARCHIVE.unlink(missing_ok=True)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    url = (
        "https://codeload.github.com/AzizulHakim00/DFU-ImageGuard/tar.gz/"
        + PINNED_CODE_COMMIT
    )
    print("Downloading pinned training source...")
    urllib.request.urlretrieve(url, ARCHIVE)

    with tarfile.open(ARCHIVE, "r:gz") as tf:
        root = EXTRACT_ROOT.resolve()
        for member in tf.getmembers():
            target = (EXTRACT_ROOT / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        tf.extractall(EXTRACT_ROOT, filter="data")

    folders = [p for p in EXTRACT_ROOT.iterdir() if p.is_dir()]
    if len(folders) != 1:
        raise RuntimeError(f"Unexpected pinned archive layout: {folders}")
    shutil.move(str(folders[0]), str(REPO_ROOT))
    ARCHIVE.unlink(missing_ok=True)
    shutil.rmtree(EXTRACT_ROOT, ignore_errors=True)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    print("Pinned training source preparation: PASS")


def trial_path(model: str, seed: int, fold_zero: int) -> Path:
    return RUN_ROOT / "trials" / model / f"seed_{seed}" / f"fold_{fold_zero + 1}"


def expected_fold_frame(locked: pd.DataFrame, fold_zero: int) -> pd.DataFrame:
    return locked.loc[locked["outer_fold"].astype(int) == int(fold_zero)].copy()


def validate_locked_manifest() -> tuple[pd.DataFrame, str]:
    if not LOCKED_SPLIT.is_file():
        raise RuntimeError(
            "LOCKED SPLIT IS MISSING. Repair is blocked because folds must never be regenerated: "
            + str(LOCKED_SPLIT)
        )
    locked_sha = sha256_file(LOCKED_SPLIT)
    locked = pd.read_csv(LOCKED_SPLIT)
    need = {
        "image_id", "group_id", "label", "label_name",
        "relative_path", "outer_fold",
    }
    missing = sorted(need - set(locked.columns))
    if missing:
        raise RuntimeError(f"Locked split is missing columns: {missing}")
    if locked.empty or locked["image_id"].duplicated().any():
        raise RuntimeError("Locked split has zero rows or duplicate image_id values")
    folds = sorted(pd.to_numeric(locked["outer_fold"], errors="raise").astype(int).unique())
    if folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Locked split fold values changed: {folds}")
    if locked.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("Locked split contains duplicate-group leakage across outer folds")
    if locked.groupby("outer_fold")["label"].nunique().min() != 2:
        raise RuntimeError("At least one locked fold is missing a class")
    label_map = locked[["label", "label_name"]].drop_duplicates()
    if len(label_map) != 2:
        raise RuntimeError(f"Unexpected label/label_name mapping in locked split: {label_map}")
    sizes = {
        int(f) + 1: int(n)
        for f, n in locked.groupby("outer_fold").size().sort_index().items()
    }
    print("Locked split: PASS")
    print("Locked split SHA256:", locked_sha)
    print("Locked fold sizes:", sizes)
    print("Locked unique images:", int(locked["image_id"].nunique()))
    return locked, locked_sha


def complete_identity_valid(metrics: dict[str, Any], model: str, seed: int, fold_zero: int) -> tuple[bool, str]:
    try:
        if str(metrics.get("model_key")) != model:
            return False, "COMPLETE model_key mismatch"
        if int(metrics.get("seed")) != int(seed):
            return False, "COMPLETE seed mismatch"
        if int(metrics.get("outer_fold")) != int(fold_zero) + 1:
            return False, "COMPLETE outer_fold mismatch"
    except Exception as exc:
        return False, f"COMPLETE identity unreadable: {exc}"
    return True, "ok"


def validate_predictions(
    frame: pd.DataFrame,
    locked: pd.DataFrame,
    model: str,
    seed: int,
    fold_zero: int,
) -> tuple[bool, str, pd.DataFrame | None]:
    if frame is None or frame.empty:
        return False, "prediction frame missing/empty", None
    pred = frame.copy()
    need = {
        "image_id", "group_id", "label", "model_key", "seed", "outer_fold",
        "prob_calibrated", "pred",
    }
    missing = sorted(need - set(pred.columns))
    if missing:
        return False, f"prediction columns missing: {missing}", None

    try:
        identity = pred[["model_key", "seed", "outer_fold"]].drop_duplicates()
        if len(identity) != 1:
            return False, f"prediction identity rows={len(identity)}", None
        row = identity.iloc[0]
        if str(row["model_key"]) != model:
            return False, "prediction model_key mismatch", None
        if int(row["seed"]) != int(seed):
            return False, "prediction seed mismatch", None
        if int(row["outer_fold"]) != int(fold_zero) + 1:
            return False, "prediction outer_fold mismatch", None
    except Exception as exc:
        return False, f"prediction identity unreadable: {exc}", None

    expected = expected_fold_frame(locked, fold_zero)
    if pred["image_id"].duplicated().any():
        return False, "duplicate image_id inside trial predictions", None
    if len(pred) != len(expected):
        return False, f"row count {len(pred)} != locked fold size {len(expected)}", None

    got_ids = set(pred["image_id"].astype(str))
    exp_ids = set(expected["image_id"].astype(str))
    if got_ids != exp_ids:
        return False, (
            f"image set mismatch: missing={len(exp_ids-got_ids)} "
            f"unexpected={len(got_ids-exp_ids)}"
        ), None

    exp = expected[["image_id", "group_id", "label", "label_name"]].copy()
    exp["image_id"] = exp["image_id"].astype(str)
    pred["image_id"] = pred["image_id"].astype(str)
    merged = pred.merge(exp, on="image_id", how="left", suffixes=("", "_locked"), validate="one_to_one")
    if merged["label_locked"].isna().any():
        return False, "prediction image not found in locked manifest", None
    if not np.array_equal(
        pd.to_numeric(merged["label"], errors="raise").astype(int).to_numpy(),
        pd.to_numeric(merged["label_locked"], errors="raise").astype(int).to_numpy(),
    ):
        return False, "label mismatch against locked manifest", None
    if not np.array_equal(
        merged["group_id"].astype(str).to_numpy(),
        merged["group_id_locked"].astype(str).to_numpy(),
    ):
        return False, "group_id mismatch against locked manifest", None
    if "label_name" in merged.columns:
        if not np.array_equal(
            merged["label_name"].astype(str).to_numpy(),
            merged["label_name_locked"].astype(str).to_numpy(),
        ):
            return False, "label_name mismatch against locked manifest", None

    probs = pd.to_numeric(merged["prob_calibrated"], errors="coerce").to_numpy(float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        return False, "prob_calibrated contains invalid values", None
    preds = pd.to_numeric(merged["pred"], errors="coerce")
    if preds.isna().any() or not set(preds.astype(int).unique()).issubset({0, 1}):
        return False, "pred contains values outside {0,1}", None

    return True, "ok", pred.sort_values("image_id").reset_index(drop=True)


def load_complete(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, "COMPLETE.json missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None, "COMPLETE.json is not an object"
        return data, "ok"
    except Exception as exc:
        return None, f"COMPLETE.json unreadable: {type(exc).__name__}: {exc}"


def direct_trial_candidate(
    locked: pd.DataFrame, model: str, seed: int, fold_zero: int
) -> tuple[bool, str, dict[str, Any] | None, pd.DataFrame | None]:
    t = trial_path(model, seed, fold_zero)
    metrics, m_reason = load_complete(t / "COMPLETE.json")
    if metrics is None:
        return False, m_reason, None, None
    ok_id, id_reason = complete_identity_valid(metrics, model, seed, fold_zero)
    if not ok_id:
        return False, id_reason, metrics, None
    pp = t / "test_predictions.csv"
    if not pp.is_file():
        return False, "test_predictions.csv missing", metrics, None
    try:
        pred = pd.read_csv(pp)
    except Exception as exc:
        return False, f"test_predictions.csv unreadable: {type(exc).__name__}: {exc}", metrics, None
    ok, reason, normalized = validate_predictions(pred, locked, model, seed, fold_zero)
    return ok, reason, metrics, normalized


def load_aggregate_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = RUN_ROOT / "tables" / "fold_seed_metrics.csv"
    preds_path = RUN_ROOT / "tables" / "all_oof_predictions.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    preds = pd.read_csv(preds_path) if preds_path.is_file() else pd.DataFrame()
    print(
        "Existing aggregate evidence:",
        f"metrics_rows={len(metrics)}",
        f"prediction_rows={len(preds)}",
    )
    return metrics, preds


def aggregate_candidate(
    metrics_all: pd.DataFrame,
    preds_all: pd.DataFrame,
    locked: pd.DataFrame,
    model: str,
    seed: int,
    fold_zero: int,
) -> tuple[bool, str, dict[str, Any] | None, pd.DataFrame | None]:
    if metrics_all.empty or preds_all.empty:
        return False, "aggregate tables unavailable", None, None
    required_m = {"model_key", "seed", "outer_fold"}
    required_p = {"model_key", "seed", "outer_fold"}
    if not required_m.issubset(metrics_all.columns) or not required_p.issubset(preds_all.columns):
        return False, "aggregate identity columns missing", None, None

    try:
        mm = metrics_all[
            (metrics_all["model_key"].astype(str) == model)
            & (pd.to_numeric(metrics_all["seed"], errors="coerce") == int(seed))
            & (pd.to_numeric(metrics_all["outer_fold"], errors="coerce") == int(fold_zero) + 1)
        ]
        pp = preds_all[
            (preds_all["model_key"].astype(str) == model)
            & (pd.to_numeric(preds_all["seed"], errors="coerce") == int(seed))
            & (pd.to_numeric(preds_all["outer_fold"], errors="coerce") == int(fold_zero) + 1)
        ]
    except Exception as exc:
        return False, f"aggregate filtering failed: {exc}", None, None

    if len(mm) != 1:
        return False, f"aggregate metric rows for identity={len(mm)}", None, None
    metrics = {k: jsonable(v) for k, v in mm.iloc[0].to_dict().items()}
    ok_id, id_reason = complete_identity_valid(metrics, model, seed, fold_zero)
    if not ok_id:
        return False, "aggregate " + id_reason, None, None
    ok, reason, normalized = validate_predictions(pp, locked, model, seed, fold_zero)
    if not ok:
        return False, "aggregate " + reason, metrics, None
    return True, "ok", metrics, normalized


def quarantine_trial(t: Path, identity: tuple[str, int, int], reason: str) -> Path | None:
    if not t.exists():
        return None
    model, seed, fold_zero = identity
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target = (
        QUARANTINE_ROOT
        / stamp
        / model
        / f"seed_{seed}"
        / f"fold_{fold_zero + 1}"
    )
    counter = 1
    while target.exists():
        counter += 1
        target = target.with_name(target.name + f"_{counter}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(t), str(target))
    write_json_atomic(
        target / "QUARANTINE_REASON.json",
        {
            "version": VERSION,
            "identity": {
                "model_key": model,
                "seed": seed,
                "outer_fold": fold_zero + 1,
            },
            "reason": reason,
            "quarantined_at_ns": time.time_ns(),
        },
    )
    return target


def restore_from_aggregate(
    identity: tuple[str, int, int],
    metrics: dict[str, Any],
    pred: pd.DataFrame,
) -> None:
    model, seed, fold_zero = identity
    t = trial_path(model, seed, fold_zero)
    if t.exists():
        quarantine_trial(t, identity, "replaced_by_locked-valid_aggregate_evidence")
    t.mkdir(parents=True, exist_ok=True)
    write_json_atomic(t / "COMPLETE.json", metrics)
    write_csv_atomic(t / "test_predictions.csv", pred)
    write_json_atomic(
        t / "REPAIR45_RECOVERY.json",
        {
            "status": "PASS",
            "source": "run_aggregate_tables",
            "version": VERSION,
            "model_key": model,
            "seed": seed,
            "outer_fold": fold_zero + 1,
            "recovered_at_ns": time.time_ns(),
        },
    )


def repair_context_valid(t: Path, identity: tuple[str, int, int], locked_sha: str) -> bool:
    context = t / "REPAIR45_TRIAL_CONTEXT.json"
    resume = t / "last_resume.pt"
    if not (context.is_file() and resume.is_file()):
        return False
    try:
        data = json.loads(context.read_text(encoding="utf-8"))
    except Exception:
        return False
    model, seed, fold_zero = identity
    return (
        data.get("version") == VERSION
        and data.get("locked_split_sha256") == locked_sha
        and data.get("source_commit") == ALGORITHM_SOURCE_COMMIT
        and data.get("model_key") == model
        and int(data.get("seed")) == seed
        and int(data.get("outer_fold")) == fold_zero + 1
    )


def critical_snapshot(identities: set[tuple[str, int, int]]) -> dict[str, str]:
    names = (
        "COMPLETE.json",
        "test_predictions.csv",
        "best_model_portable_fp16.pt",
        "TRIAL_VERIFICATION.json",
        "REPAIR45_RECOVERY.json",
    )
    out: dict[str, str] = {}
    for identity in sorted(identities):
        t = trial_path(*identity)
        for name in names:
            p = t / name
            if p.is_file():
                out[str(p.relative_to(RUN_ROOT))] = sha256_file(p)
    return out


def audit_and_recover(
    locked: pd.DataFrame, locked_sha: str
) -> tuple[pd.DataFrame, set[tuple[str, int, int]], list[tuple[str, int, int]]]:
    metrics_all, preds_all = load_aggregate_tables()
    rows: list[dict[str, Any]] = []
    protected: set[tuple[str, int, int]] = set()
    train_list: list[tuple[str, int, int]] = []

    for identity in EXPECTED:
        model, seed, fold_zero = identity
        t = trial_path(model, seed, fold_zero)
        direct_ok, direct_reason, _, _ = direct_trial_candidate(
            locked, model, seed, fold_zero
        )
        if direct_ok:
            status = "GOOD"
            source = "per_trial"
            action = "PROTECT_READ_ONLY"
            reason = "locked-fold validation passed"
            protected.add(identity)
        else:
            agg_ok, agg_reason, agg_metrics, agg_pred = aggregate_candidate(
                metrics_all, preds_all, locked, model, seed, fold_zero
            )
            if agg_ok and agg_metrics is not None and agg_pred is not None:
                restore_from_aggregate(identity, agg_metrics, agg_pred)
                re_ok, re_reason, _, _ = direct_trial_candidate(
                    locked, model, seed, fold_zero
                )
                if not re_ok:
                    raise RuntimeError(
                        f"Aggregate recovery failed validation for {identity}: {re_reason}"
                    )
                status = "GOOD_RECOVERED"
                source = "aggregate_tables"
                action = "PROTECT_READ_ONLY"
                reason = f"direct={direct_reason}; recovered from locked-valid aggregate"
                protected.add(identity)
            else:
                has_final = (t / "COMPLETE.json").exists() or (t / "test_predictions.csv").exists()
                if repair_context_valid(t, identity, locked_sha):
                    status = "PARTIAL_RESUMABLE"
                    source = "repair45_checkpoint"
                    action = "RESUME"
                    reason = "repair45 context + locked SHA match"
                elif has_final:
                    status = "BAD"
                    source = "per_trial_invalid"
                    action = "QUARANTINE_RETRAIN"
                    reason = f"direct={direct_reason}; aggregate={agg_reason}"
                elif (t / "last_resume.pt").is_file() or (t / "best_model.pt").is_file():
                    status = "PARTIAL_UNBOUND"
                    source = "old_checkpoint"
                    action = "QUARANTINE_RETRAIN"
                    reason = (
                        "old partial checkpoint is not cryptographically bound to this "
                        f"Repair45 locked split; aggregate={agg_reason}"
                    )
                else:
                    status = "MISSING"
                    source = "none"
                    action = "TRAIN"
                    reason = f"direct={direct_reason}; aggregate={agg_reason}"
                train_list.append(identity)

        rows.append(
            {
                "model_key": model,
                "seed": seed,
                "outer_fold": fold_zero + 1,
                "status": status,
                "source": source,
                "action": action,
                "reason": reason,
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        ["outer_fold", "seed", "model_key"]
    ).reset_index(drop=True)
    write_csv_atomic(RECON_CSV, audit)
    write_json_atomic(
        RECON_JSON,
        {
            "version": VERSION,
            "status": "PASS",
            "locked_split_sha256": locked_sha,
            "counts": {str(k): int(v) for k, v in audit["status"].value_counts().items()},
            "protected_count": len(protected),
            "train_or_resume_count": len(train_list),
            "rows": audit.to_dict("records"),
            "created_at_ns": time.time_ns(),
        },
    )
    return audit, protected, train_list


def remap_locked_dataset(rr, cfg, locked: pd.DataFrame) -> pd.DataFrame:
    dirs = {
        "root": RUN_ROOT,
        **{
            name: RUN_ROOT / name
            for name in (
                "tables", "figures", "models", "xai", "predictions",
                "logs", "configs", "manifests", "cache",
            )
        },
    }
    for p in dirs.values():
        Path(p).mkdir(parents=True, exist_ok=True)

    dataset_root = rr.download_dataset(cfg, dirs)
    data = locked.copy()
    data["image_path"] = [
        str(dataset_root / str(rel)) for rel in data["relative_path"].astype(str)
    ]
    missing = [p for p in data["image_path"] if not Path(p).is_file()]
    if missing:
        raise RuntimeError(
            f"Current dataset is missing {len(missing)} locked image paths; first={missing[:3]}"
        )

    if "file_sha256" in data.columns and data["file_sha256"].notna().all():
        mismatches = []
        for row in data[["image_path", "file_sha256", "image_id"]].itertuples(index=False):
            actual = sha256_file(Path(row.image_path))
            if actual != str(row.file_sha256):
                mismatches.append(str(row.image_id))
                if len(mismatches) >= 10:
                    break
        if mismatches:
            raise RuntimeError(
                "Dataset bytes do not match the locked manifest; mismatched image_id(s): "
                + ", ".join(mismatches)
            )
        print("Locked dataset file-hash verification: PASS")
    else:
        print("Locked dataset file-hash verification: SKIPPED (hash column incomplete)")

    print("Dataset runtime path remap: PASS (outer_fold values unchanged)")
    return data


def rebuild_from_trials(
    locked: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for identity in EXPECTED:
        model, seed, fold_zero = identity
        ok, reason, metrics, pred = direct_trial_candidate(
            locked, model, seed, fold_zero
        )
        if not ok or metrics is None or pred is None:
            errors.append(f"{model}/{seed}/fold{fold_zero+1}: {reason}")
            continue
        metric_rows.append(metrics)
        pred_frames.append(pred)

    if errors:
        raise RuntimeError(
            "Per-trial rebuild is incomplete:\n" + "\n".join(errors[:20])
        )
    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_frames, ignore_index=True)
    metrics_df = metrics_df.sort_values(
        ["outer_fold", "seed", "model_key"]
    ).reset_index(drop=True)
    preds_df = preds_df.sort_values(
        ["outer_fold", "seed", "model_key", "image_id"]
    ).reset_index(drop=True)
    write_csv_atomic(RUN_ROOT / "tables" / "fold_seed_metrics.csv", metrics_df)
    write_csv_atomic(RUN_ROOT / "tables" / "all_oof_predictions.csv", preds_df)
    return metrics_df, preds_df


def final_integrity(
    locked: pd.DataFrame,
    locked_sha_before: str,
    protected_snapshot_before: dict[str, str],
    protected: set[tuple[str, int, int]],
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
) -> dict[str, Any]:
    if sha256_file(LOCKED_SPLIT) != locked_sha_before:
        raise RuntimeError("LOCKED SPLIT SHA CHANGED DURING REPAIR")

    if len(metrics_df) != 45:
        raise RuntimeError(f"Expected 45 metrics rows; got {len(metrics_df)}")
    identity_count = metrics_df[
        ["model_key", "seed", "outer_fold"]
    ].drop_duplicates().shape[0]
    if identity_count != 45:
        raise RuntimeError(f"Expected 45 unique metric identities; got {identity_count}")

    expected_rows = len(locked) * len(MODELS) * len(SEEDS)
    if len(preds_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} OOF prediction rows; got {len(preds_df)}"
        )

    combo = (
        preds_df.groupby(["model_key", "seed"])
        .agg(rows=("image_id", "size"), unique_images=("image_id", "nunique"))
        .reset_index()
    )
    if len(combo) != len(MODELS) * len(SEEDS):
        raise RuntimeError(f"Expected 9 model×seed groups; got {len(combo)}")
    if not (combo["rows"] == len(locked)).all():
        raise RuntimeError("At least one model×seed does not have exactly all locked images")
    if not (combo["unique_images"] == len(locked)).all():
        raise RuntimeError("At least one model×seed has duplicate/missing image_id values")

    fold_map = locked[["image_id", "outer_fold"]].copy()
    fold_map["image_id"] = fold_map["image_id"].astype(str)
    fold_map["locked_outer_fold_one"] = fold_map["outer_fold"].astype(int) + 1
    check = preds_df.copy()
    check["image_id"] = check["image_id"].astype(str)
    check = check.merge(
        fold_map[["image_id", "locked_outer_fold_one"]],
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    wrong = int(
        (
            pd.to_numeric(check["outer_fold"], errors="raise").astype(int)
            != check["locked_outer_fold_one"].astype(int)
        ).sum()
    )
    if wrong:
        raise RuntimeError(f"Wrong image→fold assignments remain: {wrong}")

    protected_after = critical_snapshot(protected)
    if protected_after != protected_snapshot_before:
        before_keys = set(protected_snapshot_before)
        after_keys = set(protected_after)
        changed = sorted(
            k for k in before_keys & after_keys
            if protected_snapshot_before[k] != protected_after[k]
        )
        raise RuntimeError(
            "Protected GOOD trial artifacts changed. "
            f"added={sorted(after_keys-before_keys)[:5]} "
            f"removed={sorted(before_keys-after_keys)[:5]} "
            f"changed={changed[:5]}"
        )

    return {
        "completed_unique_trials": 45,
        "prediction_rows": int(len(preds_df)),
        "expected_prediction_rows": int(expected_rows),
        "unique_images_per_model_seed": int(len(locked)),
        "wrong_image_to_fold_assignments": 0,
        "locked_split_sha256": locked_sha_before,
        "protected_trials_preserved": int(len(protected)),
        "model_seed_counts": combo.to_dict("records"),
    }


def main() -> None:
    print("=" * 88)
    print(VERSION)
    print("Authoritative rule: actual evidence + existing locked split decide what trains.")
    print("No fold regeneration. No hard-coded Repair-7 assumption.")
    print("=" * 88)

    ensure_drive()
    ensure_dependencies()
    prepare_pinned_source()

    from src import reliable_runner_v2 as rr
    from src.reliable_analysis import build_reports
    from src.reliable_storage_rescue import (
        metadata_only_active_backup,
        storage_bounded_torch_save,
    )

    rr.atomic_torch = storage_bounded_torch_save
    rr.backup_active_trial = metadata_only_active_backup

    settings = rr.ReliableSettingsV2(
        run_id=RUN_ID,
        drive_root=str(DRIVE_ROOT),
        backup_root=str(BACKUP_ROOT),
        source_commit=ALGORITHM_SOURCE_COMMIT,
    )
    cfg = rr.build_config(settings)

    locked, locked_sha = validate_locked_manifest()
    audit, protected, train_list = audit_and_recover(locked, locked_sha)

    print("\n" + "=" * 88)
    print("CANONICAL 45-TRIAL RECONCILIATION")
    print("=" * 88)
    print(audit.to_string(index=False))
    print("\nStatus counts:")
    print(audit["status"].value_counts().to_string())
    print(f"\nProtected/read-only trials: {len(protected)}")
    print(f"Trials allowed to train/resume: {len(train_list)}")
    print("Training has not touched any trial yet: PASS")

    protected_snapshot_before = critical_snapshot(protected)

    if train_list:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"PyTorch unavailable: {exc}") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Non-GOOD trials exist but no CUDA GPU is available. "
                "Reconciliation is saved; training was not started."
            )
        print("GPU:", torch.cuda.get_device_name(0))
        data = remap_locked_dataset(rr, cfg, locked)

        protected_lookup = set(protected)
        train_lookup = set(train_list)
        if protected_lookup & train_lookup:
            raise RuntimeError("Internal safety error: protected and train sets overlap")

        for index, identity in enumerate(train_list, 1):
            model, seed, fold_zero = identity
            if identity in protected_lookup:
                raise RuntimeError(f"REFUSING to train protected identity: {identity}")

            t = trial_path(model, seed, fold_zero)
            resumable = repair_context_valid(t, identity, locked_sha)
            if not resumable and t.exists():
                q = quarantine_trial(
                    t,
                    identity,
                    "non-GOOD evidence/old unbound partial before Repair45 training",
                )
                print("Quarantined:", q)
            t.mkdir(parents=True, exist_ok=True)

            write_json_atomic(
                t / "REPAIR45_TRIAL_CONTEXT.json",
                {
                    "version": VERSION,
                    "locked_split_sha256": locked_sha,
                    "source_commit": ALGORITHM_SOURCE_COMMIT,
                    "model_key": model,
                    "seed": seed,
                    "outer_fold": fold_zero + 1,
                    "created_at_ns": time.time_ns(),
                },
            )

            outer_train = data.loc[data["outer_fold"].astype(int) != fold_zero].copy()
            inner = rr.make_inner_partition(outer_train, cfg, fold_zero)
            train_df = inner.loc[inner["inner_role"] == "train"].copy()
            selection_df = inner.loc[inner["inner_role"] == "selection"].copy()
            calibration_df = inner.loc[inner["inner_role"] == "calibration"].copy()
            test_df = data.loc[data["outer_fold"].astype(int) == fold_zero].copy()

            required_runtime = {
                "image_id", "image_path", "relative_path", "group_id",
                "label", "label_name", "outer_fold",
            }
            for name, frame in {
                "train": train_df,
                "selection": selection_df,
                "calibration": calibration_df,
                "test": test_df,
            }.items():
                missing = sorted(required_runtime - set(frame.columns))
                if missing:
                    raise RuntimeError(f"{name} dataframe missing required columns: {missing}")

            mode = "RESUME REPAIR45" if resumable else "TRAIN REPAIR45"
            print("\n" + "-" * 88)
            print(
                f"{mode} [{index}/{len(train_list)}]: "
                f"{model} seed={seed} fold={fold_zero+1}"
            )
            print("-" * 88)

            rr.train_trial(
                train_df=train_df,
                selection_df=selection_df,
                calibration_df=calibration_df,
                test_df=test_df,
                model_key=model,
                seed=seed,
                fold=fold_zero,
                cfg=cfg,
                settings=settings,
                trial=t,
                run=RUN_ROOT,
            )

            ok, reason, _, _ = direct_trial_candidate(
                locked, model, seed, fold_zero
            )
            if not ok:
                raise RuntimeError(
                    f"Post-training locked-fold validation failed for {identity}: {reason}"
                )

            completed_now = 0
            prediction_rows_now = 0
            for check_identity in EXPECTED:
                cm, cs, cf = check_identity
                c_ok, _, _, c_pred = direct_trial_candidate(
                    locked, cm, cs, cf
                )
                if c_ok and c_pred is not None:
                    completed_now += 1
                    prediction_rows_now += len(c_pred)
            write_json_atomic(
                PROGRESS_JSON,
                {
                    "version": VERSION,
                    "status": "RUNNING",
                    "completed_unique_trials": int(completed_now),
                    "prediction_rows": int(prediction_rows_now),
                    "last_completed": {
                        "model_key": model,
                        "seed": seed,
                        "outer_fold": fold_zero + 1,
                    },
                    "locked_split_sha256": locked_sha,
                    "protected_trials": len(protected),
                    "updated_at_ns": time.time_ns(),
                },
            )
    else:
        print("All 45 identities are already locked-valid. GPU training is unnecessary.")

    metrics_df, preds_df = rebuild_from_trials(locked)

    summary_metrics = [
        c for c in (
            "accuracy", "balanced_accuracy", "sensitivity", "specificity",
            "precision", "f1", "mcc", "roc_auc", "pr_auc",
            "brier", "log_loss", "ece",
        ) if c in metrics_df.columns
    ]
    summary = (
        metrics_df.groupby("model_key")[summary_metrics]
        .agg(["mean", "std"])
        .sort_index()
    )
    summary.to_csv(RUN_ROOT / "tables" / "model_summary.csv")

    reports = build_reports(RUN_ROOT)

    integrity = final_integrity(
        locked=locked,
        locked_sha_before=locked_sha,
        protected_snapshot_before=protected_snapshot_before,
        protected=protected,
        metrics_df=metrics_df,
        preds_df=preds_df,
    )

    with PKL_PATH.open("wb") as handle:
        pickle.dump(
            {
                "version": VERSION,
                "run_id": RUN_ID,
                "pinned_code_commit": PINNED_CODE_COMMIT,
                "algorithm_source_commit": ALGORITHM_SOURCE_COMMIT,
                "locked_split_sha256": locked_sha,
                "settings": settings.__dict__,
                "reconciliation": audit.to_dict("records"),
                "metrics": metrics_df.to_dict("records"),
                "predictions": preds_df.to_dict("records"),
                "reports": reports,
                "integrity": integrity,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    final_payload = {
        "version": VERSION,
        "status": "PASS",
        "run_id": RUN_ID,
        "pinned_code_commit": PINNED_CODE_COMMIT,
        "algorithm_source_commit": ALGORITHM_SOURCE_COMMIT,
        "reconciliation_status_counts": {
            str(k): int(v) for k, v in audit["status"].value_counts().items()
        },
        "initial_protected_trials": len(protected),
        "initial_train_or_resume_trials": len(train_list),
        **integrity,
        "reports": reports,
        "pkl": str(PKL_PATH),
        "updated_at_ns": time.time_ns(),
    }
    write_json_atomic(FINAL_JSON, final_payload)

    export_files = [
        RECON_CSV,
        RECON_JSON,
        PROGRESS_JSON,
        FINAL_JSON,
        RUN_ROOT / "tables" / "fold_seed_metrics.csv",
        RUN_ROOT / "tables" / "all_oof_predictions.csv",
        RUN_ROOT / "tables" / "model_summary.csv",
        RUN_ROOT / "tables" / "selective_prediction.csv",
        RUN_ROOT / "tables" / "error_audit.csv",
        RUN_ROOT / "tables" / "paired_bootstrap.json",
        PKL_PATH,
    ]
    EXPORT_ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(EXPORT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in export_files:
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(RUN_ROOT)))

    print("\n" + "=" * 88)
    print("REPAIR45 FINAL VERIFICATION: PASS")
    print("=" * 88)
    print("completed_unique_trials:", integrity["completed_unique_trials"])
    print("prediction_rows:", integrity["prediction_rows"])
    print("unique_images_per_model_seed:", integrity["unique_images_per_model_seed"])
    print("wrong_image_to_fold_assignments:", integrity["wrong_image_to_fold_assignments"])
    print("protected_trials_preserved:", integrity["protected_trials_preserved"])
    print("locked_split_sha256:", locked_sha)
    print("Final export:", EXPORT_ZIP)
    print("No locked fold assignment was regenerated or modified.")


if __name__ == "__main__":
    main()
