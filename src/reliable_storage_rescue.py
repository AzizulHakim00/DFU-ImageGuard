from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .config_data import sha256_file

ORIGINAL_TRAINING_COMMIT = "349143b4d8b16f885adce3559542f6c202a2bca1"
RESCUE_POLICY_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _stream_replace_from_local(local_source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with local_source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        try:
            os.fsync(dst.fileno())
        except OSError:
            pass


def storage_bounded_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Stage locally, then truncate/replace the Drive checkpoint.

    The old Drive-side atomic strategy retained the previous checkpoint while
    writing a second full `.tmp` checkpoint, temporarily requiring roughly
    twice the checkpoint size inside the same Drive quota.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    local_root = Path("/content/dfu_reliable_checkpoint_staging")
    local_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    local_new = local_root / f"{path.name}.{token}.new.pt"
    local_old = local_root / f"{path.name}.{token}.old.pt"
    path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)

    try:
        torch.save(payload, local_new)
        torch.load(local_new, map_location="cpu", weights_only=False)
        expected_size = int(local_new.stat().st_size)
        expected_sha = sha256_file(local_new)

        if path.is_file():
            shutil.copy2(path, local_old)

        _stream_replace_from_local(local_new, path)
        if int(path.stat().st_size) != expected_size:
            raise IOError(
                f"Checkpoint size mismatch after Drive write: "
                f"{path.stat().st_size} != {expected_size}"
            )
        if sha256_file(path) != expected_sha:
            raise IOError(f"Checkpoint SHA-256 mismatch after Drive write: {path}")
    except Exception as exc:
        if local_old.is_file():
            try:
                _stream_replace_from_local(local_old, path)
                if sha256_file(path) != sha256_file(local_old):
                    raise IOError("Previous checkpoint restoration hash mismatch")
            except Exception as restore_exc:
                raise RuntimeError(
                    f"Drive checkpoint write failed and restoration also failed for {path}: "
                    f"{type(exc).__name__}: {exc}; restore={type(restore_exc).__name__}: "
                    f"{restore_exc}"
                ) from exc
        raise RuntimeError(
            f"Drive checkpoint write failed for {path}. "
            "The previous checkpoint was preserved when available. "
            f"Cause: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        local_new.unlink(missing_ok=True)
        local_old.unlink(missing_ok=True)


def metadata_only_active_backup(
    run: str | Path,
    backup_root: str | Path,
    run_id: str,
    trial: str | Path,
) -> dict[str, Any]:
    """Do not duplicate active `.pt` files inside the same Drive quota."""
    run = Path(run)
    trial = Path(trial)
    backup_target = Path(backup_root) / "runs" / run_id
    active = backup_target / "_active_trial"
    skipped: list[dict[str, Any]] = []
    copied: list[dict[str, Any]] = []
    try:
        if active.exists():
            shutil.rmtree(active, ignore_errors=True)
        active.mkdir(parents=True, exist_ok=True)
        history = trial / "history.csv"
        if history.is_file():
            history_dst = active / "history.csv"
            shutil.copy2(history, history_dst)
            copied.append(
                {
                    "path": "_active_trial/history.csv",
                    "bytes": int(history_dst.stat().st_size),
                    "sha256": sha256_file(history_dst),
                }
            )
        _atomic_json(
            active / "ACTIVE_TRIAL.json",
            {
                "policy_version": RESCUE_POLICY_VERSION,
                "mode": "metadata_only_same_account_backup",
                "source_trial": str(trial.resolve().relative_to(run.resolve())),
                "primary_last_resume": str(trial / "last_resume.pt"),
                "primary_best_model": str(trial / "best_model.pt"),
                "reason": (
                    "The previous implementation duplicated hundreds of megabytes "
                    "of active checkpoint state within the same Google Drive quota."
                ),
                "updated_at_ns": time.time_ns(),
            },
        )
        copied.append(
            {
                "path": "_active_trial/ACTIVE_TRIAL.json",
                "bytes": int((active / "ACTIVE_TRIAL.json").stat().st_size),
                "sha256": sha256_file(active / "ACTIVE_TRIAL.json"),
            }
        )
    except Exception as exc:
        skipped.append(
            {
                "path": "_active_trial",
                "reason": type(exc).__name__,
                "detail": str(exc)[-500:],
            }
        )

    status = {
        "policy_version": RESCUE_POLICY_VERSION,
        "mode": "storage_rescue_metadata_only_active_backup",
        "phase": "active_trial_metadata_only",
        "verified": not skipped,
        "degraded": bool(skipped),
        "primary_authoritative": True,
        "training_may_continue": True,
        "backup_target": str(backup_target),
        "copied_files": len(copied),
        "copied_bytes": int(sum(item.get("bytes", 0) for item in copied)),
        "copied": copied,
        "skipped": skipped,
        "updated_at_ns": time.time_ns(),
    }
    try:
        _atomic_json(run / "STORAGE_STATUS.json", status)
    except Exception:
        pass
    try:
        _atomic_json(backup_target / "STORAGE_STATUS.json", status)
    except Exception:
        pass
    return status


def _validate_resume(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_key",
        "seed",
        "fold",
        "epoch",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"Missing resume keys {missing}: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "model_key": payload["model_key"],
        "seed": int(payload["seed"]),
        "fold": int(payload["fold"]),
        "epoch": int(payload["epoch"]),
        "source_commit": payload.get("source_commit"),
    }


def prepare_existing_v2_run(
    drive_root: str | Path = "/content/drive/MyDrive/DFU-ImageGuard",
    backup_root: str | Path = "/content/drive/MyDrive/DFU-ImageGuard-Backup",
    run_id: str = "RELIABLE_DFU_CV_V2",
) -> dict[str, Any]:
    """Remove failed temporary writes and preserve a valid V2 resume state."""
    run = Path(drive_root) / "runs" / run_id
    backup_target = Path(backup_root) / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    removed: list[dict[str, Any]] = []

    for root in (run, backup_target):
        if not root.exists():
            continue
        for path in list(root.rglob("*.tmp")) + list(root.rglob("*.partial")):
            if path.is_file():
                size = int(path.stat().st_size)
                path.unlink(missing_ok=True)
                removed.append({"path": str(path), "bytes": size})

    completed = []
    for complete in sorted(run.rglob("COMPLETE.json")):
        trial = complete.parent
        if (trial / "test_predictions.csv").is_file():
            completed.append(str(trial.relative_to(run)))

    valid_primary: list[dict[str, Any]] = []
    invalid_primary: list[dict[str, Any]] = []
    for resume in sorted(run.rglob("last_resume.pt")):
        if (resume.parent / "COMPLETE.json").exists():
            continue
        try:
            valid_primary.append(_validate_resume(resume))
        except Exception as exc:
            invalid_primary.append(
                {"path": str(resume), "error": f"{type(exc).__name__}: {exc}"}
            )

    restored_from_secondary = None
    secondary_active = backup_target / "_active_trial"
    secondary_resume = secondary_active / "last_resume.pt"
    active_meta = secondary_active / "ACTIVE_TRIAL.json"
    if not valid_primary and secondary_resume.is_file() and active_meta.is_file():
        try:
            meta = json.loads(active_meta.read_text(encoding="utf-8"))
            destination = run / Path(meta["source_trial"]) / "last_resume.pt"
            secondary_info = _validate_resume(secondary_resume)
            _stream_replace_from_local(secondary_resume, destination)
            restored = _validate_resume(destination)
            if restored["sha256"] != secondary_info["sha256"]:
                raise RuntimeError("Secondary resume restoration SHA mismatch")
            restored_from_secondary = restored
            valid_primary.append(restored)
        except Exception as exc:
            invalid_primary.append(
                {
                    "path": str(secondary_resume),
                    "error": f"secondary_restore_failed: {type(exc).__name__}: {exc}",
                }
            )

    secondary_bytes_released = 0
    if secondary_active.exists() and valid_primary:
        for path in secondary_active.rglob("*"):
            if path.is_file():
                try:
                    secondary_bytes_released += int(path.stat().st_size)
                except OSError:
                    pass
        shutil.rmtree(secondary_active, ignore_errors=True)

    restart_cleanup: list[dict[str, Any]] = []
    if not valid_primary:
        for history in sorted(run.rglob("history.csv")):
            trial = history.parent
            if (trial / "COMPLETE.json").exists():
                continue
            for name in ("last_resume.pt", "best_model.pt"):
                candidate = trial / name
                if candidate.is_file():
                    restart_cleanup.append(
                        {
                            "path": str(candidate),
                            "bytes": int(candidate.stat().st_size),
                            "sha256": sha256_file(candidate),
                        }
                    )
                    candidate.unlink(missing_ok=True)
            _atomic_json(
                trial / "INTERRUPTED_TRIAL_RESTART.json",
                {
                    "reason": "No valid exact-resume checkpoint survived the failed Drive write.",
                    "completed_trials_preserved": len(completed),
                    "restart_scope": "this incomplete trial only",
                    "updated_at_ns": time.time_ns(),
                },
            )

    report = {
        "policy_version": RESCUE_POLICY_VERSION,
        "run_id": run_id,
        "completed_trials": len(completed),
        "expected_trials": 45,
        "valid_incomplete_resume_count": len(valid_primary),
        "valid_incomplete_resumes": valid_primary,
        "invalid_resumes": invalid_primary,
        "restored_from_secondary": restored_from_secondary,
        "resume_action": (
            "resume_valid_incomplete_trial"
            if valid_primary
            else "restart_incomplete_trial_only"
        ),
        "restart_cleanup": restart_cleanup,
        "removed_temporary_files": removed,
        "removed_temporary_bytes": int(sum(x["bytes"] for x in removed)),
        "secondary_active_bytes_released": int(secondary_bytes_released),
        "original_training_commit": ORIGINAL_TRAINING_COMMIT,
        "prepared_at_ns": time.time_ns(),
    }
    _atomic_json(run / "STORAGE_RESCUE_AUDIT.json", report)
    print(json.dumps(report, indent=2, default=str))
    if len(completed) < 1:
        raise RuntimeError("No completed V2 trials were found; refusing blind continuation.")
    return report


def install_rescue_patches(runner_module: Any) -> None:
    runner_module.atomic_torch = storage_bounded_torch_save
    runner_module.backup_active_trial = metadata_only_active_backup
