from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "DFU_REPAIR45_LOCKED_RECOVERY_V2_20260810"
BASE_COMMIT = "0413eec1acea851664f52e8af7cc7934182aa24b"
BASE_SHA256 = "be3de60220b677da3d9bad5d8a06dcbb3e67f498a2fe97972a475a74666cab99"
BASE_URL = (
    "https://raw.githubusercontent.com/AzizulHakim00/DFU-ImageGuard/"
    + BASE_COMMIT
    + "/scripts/dfu_repair45_reconcile_locked_v1.py"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_base_module():
    raw = urllib.request.urlopen(BASE_URL, timeout=120).read()
    actual = sha256_bytes(raw)
    if actual != BASE_SHA256:
        raise RuntimeError(
            f"Base Repair45 checksum mismatch: expected={BASE_SHA256} actual={actual}"
        )
    path = Path("/content/dfu_repair45_base_v1.py")
    path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location("dfu_repair45_base_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create module spec for base Repair45 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()
base.VERSION = VERSION

RECOVERY_REPORT = base.RUN_ROOT / "LOCKED_SPLIT_RECOVERY.json"
RECOVERY_TEMP = Path("/content/dfu_locked_split_recovery")


def normalize_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    required = {"image_id", "group_id", "label", "outer_fold"}
    missing = sorted(required - set(x.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    x["image_id"] = x["image_id"].astype(str)
    x["group_id"] = x["group_id"].astype(str)
    x["label"] = pd.to_numeric(x["label"], errors="raise").astype(int)
    x["outer_fold"] = pd.to_numeric(x["outer_fold"], errors="raise").astype(int)

    if "label_name" not in x.columns:
        x["label_name"] = np.where(x["label"] == 1, "DFU", "Normal")
    if "relative_path" not in x.columns:
        raise ValueError("relative_path is required for safe runtime path remapping")

    if x.empty or x["image_id"].duplicated().any():
        raise ValueError("zero rows or duplicate image_id")
    if sorted(x["outer_fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError(f"unexpected fold values: {sorted(x['outer_fold'].unique().tolist())}")
    if x.groupby("group_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("duplicate-group leakage across outer folds")
    if x.groupby("outer_fold")["label"].nunique().min() != 2:
        raise ValueError("at least one fold is missing a class")
    if x["label"].nunique() != 2:
        raise ValueError("candidate does not contain exactly two labels")
    return x.reset_index(drop=True)


def iter_trial_prediction_files():
    trial_root = base.RUN_ROOT / "trials"
    if not trial_root.is_dir():
        return
    for model, seed, fold_zero in base.EXPECTED:
        path = base.trial_path(model, seed, fold_zero) / "test_predictions.csv"
        if path.is_file():
            yield model, seed, fold_zero, path


def candidate_evidence_score(candidate: pd.DataFrame) -> dict[str, Any]:
    matches = 0
    mismatches = 0
    unreadable = 0
    details: list[dict[str, Any]] = []
    expected_sets = {
        fold_zero: set(
            candidate.loc[candidate["outer_fold"] == fold_zero, "image_id"].astype(str)
        )
        for fold_zero in base.FOLDS_ZERO
    }

    for model, seed, fold_zero, path in iter_trial_prediction_files() or []:
        try:
            pred = pd.read_csv(path)
            need = {"image_id", "model_key", "seed", "outer_fold"}
            if pred.empty or not need.issubset(pred.columns):
                raise ValueError("missing identity/image columns")
            ids = pred[["model_key", "seed", "outer_fold"]].drop_duplicates()
            if len(ids) != 1:
                raise ValueError("multiple identities in prediction file")
            row = ids.iloc[0]
            if (
                str(row["model_key"]) != model
                or int(row["seed"]) != int(seed)
                or int(row["outer_fold"]) != int(fold_zero) + 1
            ):
                raise ValueError("identity mismatch")
            got = set(pred["image_id"].astype(str))
            ok = len(got) == len(pred) and got == expected_sets[fold_zero]
            if ok:
                matches += 1
            else:
                mismatches += 1
            details.append(
                {
                    "model_key": model,
                    "seed": seed,
                    "outer_fold": fold_zero + 1,
                    "match": bool(ok),
                    "rows": int(len(pred)),
                }
            )
        except Exception as exc:
            unreadable += 1
            details.append(
                {
                    "model_key": model,
                    "seed": seed,
                    "outer_fold": fold_zero + 1,
                    "match": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    aggregate_path = base.RUN_ROOT / "tables" / "all_oof_predictions.csv"
    aggregate_combo_matches = 0
    aggregate_combo_mismatches = 0
    if aggregate_path.is_file():
        try:
            agg = pd.read_csv(aggregate_path)
            need = {"image_id", "model_key", "seed", "outer_fold"}
            if need.issubset(agg.columns):
                for model, seed, fold_zero in base.EXPECTED:
                    part = agg[
                        (agg["model_key"].astype(str) == model)
                        & (pd.to_numeric(agg["seed"], errors="coerce") == int(seed))
                        & (pd.to_numeric(agg["outer_fold"], errors="coerce") == int(fold_zero) + 1)
                    ]
                    if part.empty:
                        continue
                    got = set(part["image_id"].astype(str))
                    ok = (
                        not part["image_id"].astype(str).duplicated().any()
                        and got == expected_sets[fold_zero]
                    )
                    if ok:
                        aggregate_combo_matches += 1
                    else:
                        aggregate_combo_mismatches += 1
        except Exception:
            pass

    evaluable = matches + mismatches
    return {
        "trial_matches": int(matches),
        "trial_mismatches": int(mismatches),
        "trial_unreadable": int(unreadable),
        "trial_evaluable": int(evaluable),
        "aggregate_combo_matches": int(aggregate_combo_matches),
        "aggregate_combo_mismatches": int(aggregate_combo_mismatches),
        "details": details,
    }


def candidate_is_strong(score: dict[str, Any]) -> bool:
    evaluable = int(score["trial_evaluable"])
    matches = int(score["trial_matches"])
    mismatches = int(score["trial_mismatches"])
    if evaluable >= 20:
        return matches >= 20 and matches > mismatches
    agg_m = int(score["aggregate_combo_matches"])
    agg_x = int(score["aggregate_combo_mismatches"])
    return matches >= 5 and agg_m >= 20 and (matches + agg_m) > (mismatches + agg_x)


def fold_signature(candidate: pd.DataFrame) -> str:
    x = candidate[["image_id", "outer_fold"]].copy()
    x = x.sort_values("image_id")
    payload = "\n".join(f"{r.image_id},{int(r.outer_fold)}" for r in x.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover_saved_candidates() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    roots = [base.DRIVE_ROOT, base.BACKUP_ROOT]
    seen_files: set[str] = set()

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("locked_outer_fold_assignments.csv"):
            try:
                if path.resolve() == base.LOCKED_SPLIT.resolve():
                    continue
            except Exception:
                pass
            key = str(path)
            if key in seen_files:
                continue
            seen_files.add(key)
            try:
                raw = path.read_bytes()
                candidate = normalize_candidate(pd.read_csv(io.BytesIO(raw)))
                score = candidate_evidence_score(candidate)
                found.append(
                    {
                        "method": "saved_csv",
                        "source": str(path),
                        "source_sha256": sha256_bytes(raw),
                        "candidate": candidate,
                        "score": score,
                        "signature": fold_signature(candidate),
                    }
                )
            except Exception as exc:
                print(f"Ignoring unusable saved locked CSV {path}: {type(exc).__name__}: {exc}")

    zip_seen = 0
    for root in roots:
        if not root.exists():
            continue
        for zpath in root.rglob("*.zip"):
            zip_seen += 1
            if zip_seen > 300:
                break
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    members = [
                        name for name in zf.namelist()
                        if Path(name).name == "locked_outer_fold_assignments.csv"
                    ]
                    for member in members[:3]:
                        raw = zf.read(member)
                        candidate = normalize_candidate(pd.read_csv(io.BytesIO(raw)))
                        score = candidate_evidence_score(candidate)
                        found.append(
                            {
                                "method": "zip_member",
                                "source": f"{zpath}::{member}",
                                "source_sha256": sha256_bytes(raw),
                                "raw": raw,
                                "candidate": candidate,
                                "score": score,
                                "signature": fold_signature(candidate),
                            }
                        )
            except Exception:
                continue
    return found


def reconstruct_deterministically() -> dict[str, Any]:
    print("No authoritative saved locked CSV copy was found; starting deterministic reconstruction in /content only.")
    shutil.rmtree(RECOVERY_TEMP, ignore_errors=True)
    RECOVERY_TEMP.mkdir(parents=True, exist_ok=True)

    from src.config_data import (
        Config,
        assign_duplicate_groups,
        build_manifest,
        download_dataset,
        make_outer_folds,
        seed_everything,
    )

    cfg = Config()
    cfg.DRIVE_ROOT = str(RECOVERY_TEMP)
    cfg.LOCAL_FALLBACK_ROOT = str(RECOVERY_TEMP)
    cfg.MOUNT_DRIVE = False
    cfg.ALLOW_LOCAL_FALLBACK = True
    cfg.SEED = 2026
    cfg.N_FOLDS = 5
    cfg.NUM_WORKERS = 2

    dirs = {"root": RECOVERY_TEMP}
    for name in ("tables", "figures", "models", "xai", "predictions", "logs", "configs", "manifests", "cache"):
        dirs[name] = RECOVERY_TEMP / name
        dirs[name].mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.SEED)
    dataset_root = download_dataset(cfg, dirs)
    manifest = build_manifest(dataset_root, cfg, dirs)
    cleaned = assign_duplicate_groups(manifest, cfg, dirs)
    candidate = make_outer_folds(cleaned, cfg, dirs)
    candidate = normalize_candidate(candidate)
    score = candidate_evidence_score(candidate)
    return {
        "method": "deterministic_pinned_reconstruction",
        "source": str(RECOVERY_TEMP / "manifests" / "locked_outer_fold_assignments.csv"),
        "source_sha256": hashlib.sha256(
            (RECOVERY_TEMP / "manifests" / "locked_outer_fold_assignments.csv").read_bytes()
        ).hexdigest(),
        "candidate": candidate,
        "score": score,
        "signature": fold_signature(candidate),
    }


def install_candidate(entry: dict[str, Any]) -> None:
    candidate = normalize_candidate(entry["candidate"])
    base.LOCKED_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    tmp = base.LOCKED_SPLIT.with_suffix(".csv.recovery_tmp")
    candidate.to_csv(tmp, index=False)
    os.replace(tmp, base.LOCKED_SPLIT)
    report = {
        "version": VERSION,
        "status": "PASS",
        "method": entry["method"],
        "source": entry["source"],
        "source_sha256": entry["source_sha256"],
        "installed_sha256": base.sha256_file(base.LOCKED_SPLIT),
        "fold_signature_sha256": entry["signature"],
        "evidence_score": entry["score"],
        "fold_sizes": {
            str(int(f) + 1): int(n)
            for f, n in candidate.groupby("outer_fold").size().sort_index().items()
        },
        "unique_images": int(candidate["image_id"].nunique()),
        "recovered_at_ns": time.time_ns(),
    }
    base.write_json_atomic(RECOVERY_REPORT, report)
    print("LOCKED SPLIT RECOVERY: PASS")
    print("Recovery method:", entry["method"])
    print("Recovery source:", entry["source"])
    print("Evidence matches/mismatches:", entry["score"]["trial_matches"], "/", entry["score"]["trial_mismatches"])
    print("Recovered fold sizes:", report["fold_sizes"])
    print("Recovered locked split SHA256:", report["installed_sha256"])


def validate_locked_manifest_with_recovery():
    if base.LOCKED_SPLIT.is_file():
        return original_validate_locked_manifest()

    print("Locked split missing at expected V4 path. Recovery mode engaged; GPU training remains BLOCKED.")
    candidates = discover_saved_candidates()
    strong = [c for c in candidates if candidate_is_strong(c["score"])]

    if strong:
        strong.sort(
            key=lambda c: (
                int(c["score"]["trial_matches"]),
                int(c["score"]["aggregate_combo_matches"]),
                -int(c["score"]["trial_mismatches"]),
            ),
            reverse=True,
        )
        best = strong[0]
        best_key = (
            int(best["score"]["trial_matches"]),
            int(best["score"]["aggregate_combo_matches"]),
            int(best["score"]["trial_mismatches"]),
        )
        tied = [
            c for c in strong
            if (
                int(c["score"]["trial_matches"]),
                int(c["score"]["aggregate_combo_matches"]),
                int(c["score"]["trial_mismatches"]),
            ) == best_key
        ]
        signatures = {c["signature"] for c in tied}
        if len(signatures) > 1:
            raise RuntimeError(
                "LOCKED SPLIT RECOVERY AMBIGUOUS: equally supported saved copies disagree. "
                "Training remains blocked."
            )
        install_candidate(best)
        return original_validate_locked_manifest()

    reconstructed = reconstruct_deterministically()
    if not candidate_is_strong(reconstructed["score"]):
        s = reconstructed["score"]
        raise RuntimeError(
            "Deterministic locked-split reconstruction could not be proven against saved evidence. "
            f"trial_matches={s['trial_matches']} trial_mismatches={s['trial_mismatches']} "
            f"aggregate_matches={s['aggregate_combo_matches']} aggregate_mismatches={s['aggregate_combo_mismatches']}. "
            "GPU training remains blocked."
        )
    install_candidate(reconstructed)
    return original_validate_locked_manifest()


original_validate_locked_manifest = base.validate_locked_manifest
base.validate_locked_manifest = validate_locked_manifest_with_recovery

print(VERSION)
print("Base Repair45 SHA256: PASS")
print("Locked-split policy: saved-copy recovery first; deterministic reconstruction only with evidence proof")
print("GPU training cannot begin until recovered split validation passes")
base.main()
