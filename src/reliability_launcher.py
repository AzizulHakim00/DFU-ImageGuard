from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


def _is_mountpoint(path: Path) -> bool:
    try:
        return subprocess.run(
            ["mountpoint", "-q", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except Exception:
        return False


def _verify_drive(my_drive: Path) -> Path:
    if not my_drive.is_dir():
        raise RuntimeError(f"{my_drive} is unavailable")
    project_root = my_drive / "DFU-ImageGuard"
    marker_dir = project_root / "_mount_verification"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"reliability_{uuid.uuid4().hex}.txt"
    expected = f"verified-{time.time_ns()}"
    marker.write_text(expected, encoding="utf-8")
    if marker.read_text(encoding="utf-8") != expected:
        raise RuntimeError("Drive write/read verification failed")
    marker.unlink(missing_ok=True)
    return project_root


def _clean_stale_mount(drive_module: Any, mount_point: Path) -> None:
    try:
        drive_module.flush_and_unmount()
    except Exception:
        pass
    time.sleep(2)
    if _is_mountpoint(mount_point):
        subprocess.run(
            ["fusermount", "-uz", str(mount_point)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
    if not _is_mountpoint(mount_point):
        shutil.rmtree(mount_point, ignore_errors=True)
        mount_point.mkdir(parents=True, exist_ok=True)


def mount_drive_verified(max_attempts: int = 3) -> Path:
    from google.colab import drive

    mount_point = Path("/content/drive")
    my_drive = mount_point / "MyDrive"
    if my_drive.is_dir():
        try:
            root = _verify_drive(my_drive)
            print(f"Google Drive already mounted and verified: {root}")
            return root
        except Exception:
            pass

    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(
                    f"Repairing stale Drive mount before attempt "
                    f"{attempt}/{max_attempts}..."
                )
                _clean_stale_mount(drive, mount_point)
            print(f"Mounting Google Drive — attempt {attempt}/{max_attempts}...")
            drive.mount(str(mount_point), force_remount=attempt > 1)
            root = _verify_drive(my_drive)
            print(f"Google Drive mounted and write-verified: {root}")
            return root
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            time.sleep(2)
    raise RuntimeError(
        "Google Drive could not be mounted. Training was NOT started.\n"
        + "\n".join(errors)
    )


def launch_reliability_experiment(
    repository_dir: str | Path = "/content/DFU-ImageGuard-reliability",
):
    project_root = mount_drive_verified()
    backup_root = Path("/content/drive/MyDrive/DFU-ImageGuard-Backup")
    backup_root.mkdir(parents=True, exist_ok=True)

    from .reliability_io import ReliabilitySettings
    from .reliability_runner import run_reliability_experiment

    settings = ReliabilitySettings(
        run_id="DFU_RELIABILITY_FINAL_V1",
        drive_root=str(project_root),
        secondary_drive_root=str(backup_root),
        seeds=(2026, 2027, 2028),
        model_keys=(
            "convnextv2_tiny",
            "mobilenetv3_large",
            "densenet121",
        ),
        primary_model_key="convnextv2_tiny",
        n_folds=5,
        max_epochs=30,
        patience=7,
        batch_size=16,
        num_workers=2,
        run_robustness=True,
        run_gradcam=True,
        require_secondary_backup=True,
        force_retrain=False,
    )
    print(
        "FINAL PROTOCOL LOCKED: "
        "3 models × 3 seeds × 5 folds = 45 resumable trials"
    )
    print(
        "Architecture search is closed; "
        "outer tests are used once for final inference."
    )
    return run_reliability_experiment(
        settings=settings,
        repository_dir=Path(repository_dir),
    )
