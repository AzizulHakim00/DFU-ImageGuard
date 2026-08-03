from __future__ import annotations

import errno
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from .config_data import sha256_file


STATUS_FILE = "STORAGE_STATUS.json"
POLICY_VERSION = 2


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _is_quota_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EDQUOT}:
        return True
    text = str(exc).lower()
    return any(token in text for token in ("no space left", "quota", "storage full"))


def _verified_copy(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.unlink(missing_ok=True)
    shutil.copy2(src, tmp)
    expected_size = src.stat().st_size
    if tmp.stat().st_size != expected_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Byte-count mismatch while backing up {src}")
    expected_sha = sha256_file(src)
    observed_sha = sha256_file(tmp)
    if observed_sha != expected_sha:
        tmp.unlink(missing_ok=True)
        raise IOError(f"SHA-256 mismatch while backing up {src}")
    os.replace(tmp, dst)
    return {"bytes": int(expected_size), "sha256": expected_sha}


def _status(
    run: Path,
    backup_target: Path,
    phase: str,
    copied: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "policy_version": POLICY_VERSION,
        "mode": "bounded_primary_plus_metadata_backup",
        "phase": phase,
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
    atomic_json(run / STATUS_FILE, payload)
    try:
        atomic_json(backup_target / STATUS_FILE, payload)
    except Exception:
        pass
    return payload


def _copy_many(
    run: Path,
    backup_target: Path,
    sources: Iterable[Path],
    destination_prefix: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for src in sources:
        if not src.is_file():
            continue
        try:
            relative = src.resolve().relative_to(run.resolve())
        except Exception:
            relative = Path(src.name)
        destination = backup_target / (destination_prefix or Path()) / relative
        try:
            meta = _verified_copy(src, destination)
            copied.append({"path": relative.as_posix(), **meta})
        except Exception as exc:
            destination.with_name(destination.name + ".partial").unlink(missing_ok=True)
            skipped.append(
                {
                    "path": relative.as_posix(),
                    "reason": "NO_SPACE_OR_QUOTA" if _is_quota_error(exc) else type(exc).__name__,
                    "detail": str(exc)[-400:],
                }
            )
    return copied, skipped


def backup_active_trial(
    run: str | Path,
    backup_root: str | Path,
    run_id: str,
    trial: str | Path,
) -> dict[str, Any]:
    """Keep one rolling secondary resume copy for only the active trial."""
    run = Path(run)
    trial = Path(trial)
    backup_target = Path(backup_root) / "runs" / run_id
    active = backup_target / "_active_trial"
    try:
        if active.exists():
            shutil.rmtree(active, ignore_errors=True)
        active.mkdir(parents=True, exist_ok=True)
        candidates = [trial / "last_resume.pt", trial / "best_model.pt", trial / "history.csv"]
        copied: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for src in candidates:
            if not src.is_file():
                continue
            dst = active / src.name
            try:
                meta = _verified_copy(src, dst)
                copied.append({"path": f"_active_trial/{src.name}", **meta})
            except Exception as exc:
                dst.with_name(dst.name + ".partial").unlink(missing_ok=True)
                skipped.append(
                    {
                        "path": f"_active_trial/{src.name}",
                        "reason": "NO_SPACE_OR_QUOTA" if _is_quota_error(exc) else type(exc).__name__,
                        "detail": str(exc)[-400:],
                    }
                )
        atomic_json(
            active / "ACTIVE_TRIAL.json",
            {
                "source_trial": str(trial.resolve().relative_to(run.resolve())),
                "updated_at_ns": time.time_ns(),
            },
        )
        return _status(run, backup_target, "active_trial", copied, skipped)
    except Exception as exc:
        skipped = [
            {
                "path": "_active_trial",
                "reason": "NO_SPACE_OR_QUOTA" if _is_quota_error(exc) else type(exc).__name__,
                "detail": str(exc)[-400:],
            }
        ]
        return _status(run, backup_target, "active_trial", [], skipped)


def clear_active_trial_backup(
    run: str | Path,
    backup_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    run = Path(run)
    backup_target = Path(backup_root) / "runs" / run_id
    active = backup_target / "_active_trial"
    try:
        if active.exists():
            shutil.rmtree(active, ignore_errors=True)
        return _status(run, backup_target, "active_trial_cleared", [], [])
    except Exception as exc:
        return _status(
            run,
            backup_target,
            "active_trial_cleared",
            [],
            [{"path": "_active_trial", "reason": type(exc).__name__, "detail": str(exc)[-400:]}],
        )


def metadata_files(run: Path) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        if "cache" in relative.parts or path.name.endswith((".tmp", ".partial")):
            continue
        if path.suffix.lower() == ".pt":
            continue
        if path.name == STATUS_FILE:
            continue
        selected.append(path)
    return selected


def backup_metadata(
    run: str | Path,
    backup_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Back up reports and provenance, never duplicate completed model weights."""
    run = Path(run)
    backup_target = Path(backup_root) / "runs" / run_id
    backup_target.mkdir(parents=True, exist_ok=True)
    copied, skipped = _copy_many(run, backup_target, metadata_files(run))
    return _status(run, backup_target, "metadata", copied, skipped)


def verify_portable_checkpoint(path: Path, expected_model_key: str, expected_seed: int, expected_fold: int) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_key") != expected_model_key:
        raise RuntimeError(f"Portable checkpoint model mismatch: {path}")
    if int(payload.get("seed")) != int(expected_seed):
        raise RuntimeError(f"Portable checkpoint seed mismatch: {path}")
    if int(payload.get("fold")) != int(expected_fold):
        raise RuntimeError(f"Portable checkpoint fold mismatch: {path}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Portable checkpoint has no state_dict: {path}")
    return {
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "tensor_keys": len(state_dict),
    }


def retire_completed_full_checkpoints(
    trial: str | Path,
    model_key: str,
    seed: int,
    fold: int,
) -> dict[str, Any]:
    """Delete bulky training-state files only after portable evidence is verified."""
    trial = Path(trial)
    portable = trial / "best_model_portable_fp16.pt"
    predictions = trial / "test_predictions.csv"
    complete = trial / "COMPLETE.json"
    if not (portable.is_file() and predictions.is_file() and complete.is_file()):
        raise RuntimeError(f"Cannot retire incomplete trial: {trial}")

    portable_meta = verify_portable_checkpoint(portable, model_key, seed, fold)
    retired: list[dict[str, Any]] = []
    for name in ("last_resume.pt", "best_model.pt"):
        path = trial / name
        if path.is_file():
            retired.append(
                {
                    "name": name,
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
            path.unlink()

    payload = {
        "policy_version": POLICY_VERSION,
        "status": "COMPLETED_PORTABLE_RETAINED",
        "portable_checkpoint": portable_meta,
        "predictions": {
            "bytes": int(predictions.stat().st_size),
            "sha256": sha256_file(predictions),
        },
        "retired_full_checkpoints": retired,
        "retired_at_ns": time.time_ns(),
        "reason": "Completed trials retain verified FP16 inference weights; optimizer resume state is needed only while a trial is incomplete.",
    }
    atomic_json(trial / "CHECKPOINT_RETENTION.json", payload)
    return payload
