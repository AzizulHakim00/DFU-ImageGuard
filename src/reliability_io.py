from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config_data import (
    Config,
    assign_duplicate_groups,
    build_manifest,
    download_dataset,
    make_outer_folds,
    sha256_file,
)
from .reliability_models import MODEL_SPECS

@dataclass
class ReliabilitySettings:
    run_id: str = "DFU_RELIABILITY_FINAL_V1"
    drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard"
    secondary_drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard-Backup"
    seeds: tuple[int, ...] = (2026, 2027, 2028)
    model_keys: tuple[str, ...] = (
        "convnextv2_tiny",
        "mobilenetv3_large",
        "densenet121",
    )
    primary_model_key: str = "convnextv2_tiny"
    n_folds: int = 5
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    max_epochs: int = 30
    patience: int = 7
    freeze_epochs: int = 2
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.20
    target_sensitivity: float = 0.95
    coverage_levels: tuple[float, ...] = (1.0, 0.95, 0.90, 0.80)
    bootstrap_reps: int = 2000
    run_robustness: bool = True
    run_gradcam: bool = True
    gradcam_cases: int = 12
    require_secondary_backup: bool = True
    github_sync_branch: str = "reliability-results"
    force_retrain: bool = False
    robustness_levels: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "brightness": (0.80, 0.65),
            "contrast": (0.80, 0.65),
            "gaussian_blur": (1.0, 2.0),
            "jpeg_quality": (70.0, 40.0),
            "gaussian_noise": (0.03, 0.07),
            "rotation": (7.0, 15.0),
            "occlusion": (0.10, 0.20),
        }
    )

    def validate(self) -> None:
        if self.n_folds != 5:
            raise ValueError("The primary protocol is locked to five outer folds")
        if len(self.seeds) < 3:
            raise ValueError("At least three predefined seeds are required")
        if self.primary_model_key not in self.model_keys:
            raise ValueError("Primary model must be included in model_keys")
        unknown = set(self.model_keys) - set(MODEL_SPECS)
        if unknown:
            raise KeyError(f"Unknown model keys: {sorted(unknown)}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique")
        if not 0 < self.target_sensitivity <= 1:
            raise ValueError("target_sensitivity must be in (0,1]")
        if self.bootstrap_reps < 500:
            raise ValueError("At least 500 bootstrap replicates are required")


DIRECTORIES = (
    "tables",
    "figures",
    "models",
    "predictions",
    "logs",
    "configs",
    "manifests",
    "cache",
    "xai",
    "robustness",
    "external",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _file_manifest(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        rel = str(path.relative_to(root))
        result[rel] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return result


def _copy_tree_verified(source: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    source_manifest = _file_manifest(source)
    destination_manifest = _file_manifest(destination)
    if source_manifest != destination_manifest:
        raise RuntimeError(f"Backup verification failed: {source} -> {destination}")
    return {
        "verified": True,
        "source": str(source),
        "destination": str(destination),
        "files": len(source_manifest),
    }


def _sync_trial_verified(trial_dir: Path, run_root: Path, settings: ReliabilitySettings) -> dict[str, Any]:
    relative = trial_dir.relative_to(run_root)
    destination = (
        Path(settings.secondary_drive_root)
        / "runs"
        / settings.run_id
        / relative
    )
    status = _copy_tree_verified(trial_dir, destination)
    atomic_json(trial_dir / "SECONDARY_TRIAL_BACKUP.json", status)
    shutil.copy2(
        trial_dir / "SECONDARY_TRIAL_BACKUP.json",
        destination / "SECONDARY_TRIAL_BACKUP.json",
    )
    return status


def mirror_full_run(run_root: Path, settings: ReliabilitySettings) -> dict[str, Any]:
    destination = Path(settings.secondary_drive_root) / "runs" / settings.run_id
    status = _copy_tree_verified(run_root, destination)
    atomic_json(run_root / "SECONDARY_BACKUP_STATUS.json", status)
    shutil.copy2(
        run_root / "SECONDARY_BACKUP_STATUS.json",
        destination / "SECONDARY_BACKUP_STATUS.json",
    )
    return status


def persistence_preflight(settings: ReliabilitySettings) -> tuple[Path, dict[str, Path]]:
    settings.validate()
    drive_root = Path(settings.drive_root)
    secondary_root = Path(settings.secondary_drive_root)
    allowed = [Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")]
    if not any(root.exists() and _inside(drive_root, root) for root in allowed):
        raise RuntimeError(f"Primary persistent Drive path is invalid: {drive_root}")
    if settings.require_secondary_backup and not any(
        root.exists() and _inside(secondary_root, root) for root in allowed
    ):
        raise RuntimeError(f"Secondary persistent Drive path is invalid: {secondary_root}")

    run_root = drive_root / "runs" / settings.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    dirs = {"root": run_root}
    for name in DIRECTORIES:
        dirs[name] = run_root / name
        dirs[name].mkdir(parents=True, exist_ok=True)

    nonce = hashlib.sha256(f"{time.time_ns()}-{settings.run_id}".encode()).hexdigest()
    primary_marker = run_root / "PRIMARY_DRIVE_SENTINEL.json"
    atomic_json(primary_marker, {"nonce": nonce, "run_root": str(run_root)})
    if json.loads(primary_marker.read_text())["nonce"] != nonce:
        raise RuntimeError("Primary Drive write/read verification failed")

    secondary_run = secondary_root / "runs" / settings.run_id
    secondary_run.mkdir(parents=True, exist_ok=True)
    secondary_marker = secondary_run / "SECONDARY_DRIVE_SENTINEL.json"
    atomic_json(secondary_marker, {"nonce": nonce, "run_root": str(secondary_run)})
    if json.loads(secondary_marker.read_text())["nonce"] != nonce:
        raise RuntimeError("Secondary Drive write/read verification failed")

    atomic_json(run_root / "resolved_settings.json", asdict(settings))
    atomic_json(
        run_root / "PROTOCOL_LOCK.json",
        {
            "architecture_search_closed": True,
            "primary_model": settings.primary_model_key,
            "models": list(settings.model_keys),
            "seeds": list(settings.seeds),
            "outer_folds": settings.n_folds,
            "expected_training_trials": (
                len(settings.model_keys) * len(settings.seeds) * settings.n_folds
            ),
            "test_set_role": "final inference only",
            "selection_role": "early stopping only",
            "calibration_role": "temperature and threshold only",
            "external_tuning_allowed": False,
        },
    )
    print("PERSISTENCE PREFLIGHT: PASS", run_root)
    return run_root, dirs


def _resolved_config(settings: ReliabilitySettings) -> Config:
    cfg = Config()
    cfg.DRIVE_ROOT = settings.drive_root
    cfg.ALLOW_LOCAL_FALLBACK = False
    cfg.SEED = int(settings.seeds[0])
    cfg.N_FOLDS = settings.n_folds
    cfg.IMAGE_SIZE = settings.image_size
    cfg.BATCH_SIZE = settings.batch_size
    cfg.NUM_WORKERS = settings.num_workers
    cfg.MAX_EPOCHS = settings.max_epochs
    cfg.PATIENCE = settings.patience
    cfg.WEIGHT_DECAY = settings.weight_decay
    cfg.DROPOUT = settings.dropout
    cfg.TARGET_SENSITIVITY = settings.target_sensitivity
    cfg.FORCE_RETRAIN = settings.force_retrain
    return cfg


def _prepare_locked_manifest(
    cfg: Config,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    dataset_root = download_dataset(cfg, dirs)
    current = assign_duplicate_groups(build_manifest(dataset_root, cfg, dirs), cfg, dirs)
    locked_path = dirs["manifests"] / "locked_outer_fold_assignments.csv"
    if locked_path.exists():
        old = pd.read_csv(locked_path)
        required = {"image_id", "outer_fold", "label", "group_id"}
        if not required.issubset(old.columns):
            raise RuntimeError("Existing locked fold file is invalid")
        merged = current.merge(
            old[["image_id", "outer_fold", "label", "group_id"]].rename(
                columns={"label": "locked_label", "group_id": "locked_group"}
            ),
            on="image_id",
            how="left",
            validate="one_to_one",
        )
        if merged.outer_fold.isna().any() or len(merged) != len(old):
            raise RuntimeError(
                "Current cleaned dataset no longer matches the locked fold assignment"
            )
        if not (merged.label == merged.locked_label).all():
            raise RuntimeError("Label changed relative to locked assignment")
        if not (merged.group_id == merged.locked_group).all():
            raise RuntimeError("Duplicate grouping changed relative to locked assignment")
        manifest = merged.drop(columns=["locked_label", "locked_group"])
        manifest.outer_fold = manifest.outer_fold.astype(int)
        atomic_csv(manifest, locked_path)
    else:
        manifest = make_outer_folds(current, cfg, dirs)

    for fold in range(cfg.N_FOLDS):
        train_groups = set(manifest.loc[manifest.outer_fold != fold, "group_id"])
        test_groups = set(manifest.loc[manifest.outer_fold == fold, "group_id"])
        if train_groups & test_groups:
            raise AssertionError(f"Duplicate-group leakage in fold {fold + 1}")
    atomic_json(
        dirs["root"] / "LOCKED_SPLIT_HASH.json",
        {
            "sha256": sha256_file(locked_path),
            "n_images": int(len(manifest)),
            "n_groups": int(manifest.group_id.nunique()),
            "patient_ids_available": False,
            "split_unit": "duplicate_group",
        },
    )
    return manifest
