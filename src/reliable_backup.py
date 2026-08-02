from __future__ import annotations

import errno
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from .config_data import sha256_file


BACKUP_LAYOUT_VERSION = 2
_LAYOUT_FILE = "BACKUP_LAYOUT.json"
_STATUS_FILE = "SECONDARY_BACKUP_STATUS.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _is_no_space(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EDQUOT}:
        return True
    text = str(exc).lower()
    return "no space left" in text or "quota" in text or "storage quota" in text


def _failure_status(
    run: Path,
    backup_root: str | Path,
    run_id: str,
    phase: str,
    exc: BaseException,
) -> dict[str, Any]:
    target = Path(backup_root) / "runs" / run_id
    status = {
        "layout_version": BACKUP_LAYOUT_VERSION,
        "mode": "compact_storage_aware",
        "phase": phase,
        "verified": False,
        "degraded": True,
        "target": str(target),
        "copied_files": 0,
        "copied_bytes": 0,
        "skipped": [
            {
                "path": "secondary_backup_operation",
                "reason": "NO_SPACE_OR_QUOTA" if _is_no_space(exc) else type(exc).__name__,
                "detail": str(exc)[-500:],
            }
        ],
        "updated_at_ns": time.time_ns(),
        "primary_drive_remains_authoritative": True,
        "training_may_continue": True,
    }
    try:
        _atomic_json(run / _STATUS_FILE, status)
    except Exception:
        pass
    return status


def _copy_verified(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.unlink(missing_ok=True)
    shutil.copy2(src, tmp)
    src_size = src.stat().st_size
    if tmp.stat().st_size != src_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Backup byte-count mismatch for {src}")
    src_sha = sha256_file(src)
    tmp_sha = sha256_file(tmp)
    if src_sha != tmp_sha:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Backup SHA-256 mismatch for {src}")
    os.replace(tmp, dst)
    return {"bytes": src_size, "sha256": src_sha}


def _prepare_target(backup_root: str | Path, run_id: str) -> Path:
    target = Path(backup_root) / "runs" / run_id
    layout = target / _LAYOUT_FILE
    current: dict[str, Any] = {}
    if layout.exists():
        try:
            current = json.loads(layout.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    if current.get("layout_version") != BACKUP_LAYOUT_VERSION:
        # Legacy layout copied the whole run after every epoch. Delete it once to
        # release quota, then rebuild using the compact layout below.
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            layout,
            {
                "layout_version": BACKUP_LAYOUT_VERSION,
                "mode": "compact_storage_aware",
                "created_at_ns": time.time_ns(),
                "policy": {
                    "primary_drive": "all full checkpoints and artifacts",
                    "secondary_epoch": "one rolling active last_resume.pt, best_model.pt and history",
                    "secondary_completed": "portable FP16 checkpoints and non-checkpoint artifacts",
                    "failure_policy": "record DEGRADED and continue; primary artifacts are never deleted",
                },
            },
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def _relative_to_run(path: Path, run: Path) -> Path:
    return path.resolve().relative_to(run.resolve())


def _remove_partial_files(target: Path) -> None:
    for path in target.rglob("*.partial"):
        path.unlink(missing_ok=True)


def _write_status(
    run: Path,
    target: Path,
    phase: str,
    copied: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    total_bytes = int(sum(int(item.get("bytes", 0)) for item in copied))
    status = {
        "layout_version": BACKUP_LAYOUT_VERSION,
        "mode": "compact_storage_aware",
        "phase": phase,
        "verified": len(skipped) == 0,
        "degraded": len(skipped) > 0,
        "target": str(target),
        "copied_files": len(copied),
        "copied_bytes": total_bytes,
        "skipped": skipped,
        "updated_at_ns": time.time_ns(),
        "primary_drive_remains_authoritative": True,
        "training_may_continue": True,
    }
    _atomic_json(run / _STATUS_FILE, status)
    try:
        _atomic_json(target / _STATUS_FILE, status)
    except Exception:
        pass
    return status


def _copy_many(
    run: Path,
    target: Path,
    files: Iterable[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for src in files:
        if not src.is_file():
            continue
        rel = _relative_to_run(src, run)
        dst = target / rel
        try:
            meta = _copy_verified(src, dst)
            copied.append({"path": rel.as_posix(), **meta})
        except Exception as exc:
            dst.with_name(dst.name + ".partial").unlink(missing_ok=True)
            skipped.append(
                {
                    "path": rel.as_posix(),
                    "reason": "NO_SPACE_OR_QUOTA" if _is_no_space(exc) else type(exc).__name__,
                    "detail": str(exc)[-300:],
                }
            )
    return copied, skipped


def _backup_epoch_state_impl(
    run: Path,
    backup_root: str | Path,
    run_id: str,
    trial: Path,
) -> dict[str, Any]:
    target = _prepare_target(backup_root, run_id)
    _remove_partial_files(target)

    active_root = target / "_active_trial"
    if active_root.exists():
        shutil.rmtree(active_root, ignore_errors=True)
    active_root.mkdir(parents=True, exist_ok=True)

    candidates = [
        trial / "last_resume.pt",
        trial / "best_model.pt",
        trial / "history.csv",
    ]
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for src in candidates:
        if not src.is_file():
            continue
        dst = active_root / src.name
        try:
            meta = _copy_verified(src, dst)
            copied.append(
                {
                    "path": f"_active_trial/{src.name}",
                    "source": _relative_to_run(src, run).as_posix(),
                    **meta,
                }
            )
        except Exception as exc:
            dst.with_name(dst.name + ".partial").unlink(missing_ok=True)
            skipped.append(
                {
                    "path": f"_active_trial/{src.name}",
                    "source": _relative_to_run(src, run).as_posix(),
                    "reason": "NO_SPACE_OR_QUOTA" if _is_no_space(exc) else type(exc).__name__,
                    "detail": str(exc)[-300:],
                }
            )
    try:
        _atomic_json(
            active_root / "ACTIVE_TRIAL.json",
            {
                "trial": _relative_to_run(trial, run).as_posix(),
                "updated_at_ns": time.time_ns(),
            },
        )
    except Exception as exc:
        skipped.append(
            {
                "path": "_active_trial/ACTIVE_TRIAL.json",
                "reason": "NO_SPACE_OR_QUOTA" if _is_no_space(exc) else type(exc).__name__,
                "detail": str(exc)[-300:],
            }
        )
    return _write_status(run, target, "epoch", copied, skipped)


def backup_epoch_state(
    run: str | Path,
    backup_root: str | Path,
    run_id: str,
    trial: str | Path,
) -> dict[str, Any]:
    run_path = Path(run)
    try:
        return _backup_epoch_state_impl(
            run_path,
            backup_root,
            run_id,
            Path(trial),
        )
    except Exception as exc:
        return _failure_status(run_path, backup_root, run_id, "epoch", exc)


def _completed_artifact_files(run: Path) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run)
        if (
            "cache" in rel.parts
            or path.name.endswith(".tmp")
            or path.name == _STATUS_FILE
        ):
            continue
        if path.suffix == ".pt":
            if (
                path.name == "best_model_portable_fp16.pt"
                and (path.parent / "COMPLETE.json").exists()
            ):
                selected.append(path)
            continue
        selected.append(path)
    return selected


def _backup_completed_artifacts_impl(
    run: Path,
    backup_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    target = _prepare_target(backup_root, run_id)
    _remove_partial_files(target)

    active_root = target / "_active_trial"
    if active_root.exists():
        shutil.rmtree(active_root, ignore_errors=True)

    files = _completed_artifact_files(run)
    copied, skipped = _copy_many(run, target, files)
    return _write_status(
        run,
        target,
        "completed_trial_or_final",
        copied,
        skipped,
    )


def backup_completed_artifacts(
    run: str | Path,
    backup_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    run_path = Path(run)
    try:
        return _backup_completed_artifacts_impl(
            run_path,
            backup_root,
            run_id,
        )
    except Exception as exc:
        return _failure_status(
            run_path,
            backup_root,
            run_id,
            "completed_trial_or_final",
            exc,
        )
