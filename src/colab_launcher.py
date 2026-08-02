from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from getpass import getpass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


def _verify_drive_read_write(my_drive: Path) -> Path:
    if not my_drive.is_dir():
        raise RuntimeError(f"{my_drive} is unavailable after mounting.")

    project_root = my_drive / "DFU-ImageGuard"
    check_dir = project_root / "_mount_verification"
    check_dir.mkdir(parents=True, exist_ok=True)
    check_file = check_dir / f"verified_{uuid.uuid4().hex}.txt"
    expected = f"DFU-ImageGuard verified at {time.time_ns()}"
    check_file.write_text(expected, encoding="utf-8")
    observed = check_file.read_text(encoding="utf-8")
    if observed != expected:
        raise RuntimeError("Google Drive write/read verification did not match.")
    check_file.unlink(missing_ok=True)
    return project_root


def _clean_stale_mount(drive_module: Any, mount_point: Path) -> None:
    try:
        drive_module.flush_and_unmount()
    except Exception:
        pass
    time.sleep(2)

    if _is_mountpoint(mount_point):
        try:
            subprocess.run(
                ["fusermount", "-uz", str(mount_point)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
        time.sleep(1)

    if not _is_mountpoint(mount_point):
        shutil.rmtree(mount_point, ignore_errors=True)
        mount_point.mkdir(parents=True, exist_ok=True)


def mount_drive_verified(max_attempts: int = 3) -> Path:
    from google.colab import drive

    mount_point = Path("/content/drive")
    my_drive = mount_point / "MyDrive"
    errors: list[str] = []

    if my_drive.is_dir():
        try:
            root = _verify_drive_read_write(my_drive)
            print(f"Google Drive already mounted and write-verified: {root}")
            return root
        except Exception as exc:
            errors.append(f"existing mount: {type(exc).__name__}: {exc}")

    for attempt in range(1, max_attempts + 1):
        force = attempt > 1
        try:
            if force:
                print(f"Repairing stale Drive mount before attempt {attempt}/{max_attempts}...")
                _clean_stale_mount(drive, mount_point)
            print(f"Mounting Google Drive — attempt {attempt}/{max_attempts}...")
            drive.mount(str(mount_point), force_remount=force)
            root = _verify_drive_read_write(my_drive)
            print(f"Google Drive mounted and write-verified: {root}")
            return root
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            print(f"Drive attempt {attempt} failed: {exc}")
            time.sleep(2)

    details = "\n".join(f"- {item}" for item in errors)
    raise RuntimeError(
        "Google Drive could not be mounted after verified retries. Training was NOT started.\n\n"
        "Use Runtime → Disconnect and delete runtime, reopen the notebook, approve Drive access, and run again.\n\n"
        f"Recorded errors:\n{details}"
    )


def _read_secret_safely() -> str:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        return token

    try:
        from google.colab import userdata

        token = userdata.get("GITHUB_TOKEN")
        if token:
            return str(token).strip()
    except Exception as exc:
        print(f"Colab secret GITHUB_TOKEN is unavailable: {type(exc).__name__}")

    print("\nGitHub token is required only for automatic backup pushes.")
    print("Paste it in the hidden prompt below. It will stay only in this runtime and will not be written to Drive or GitHub.")
    token = getpass("GitHub token (hidden): ").strip()
    if not token:
        raise RuntimeError(
            "No GitHub token was provided. Training was NOT started. "
            "Create a token with repository write access, then rerun."
        )
    return token


def _validate_token(token: str, repository: str = "AzizulHakim00/DFU-ImageGuard") -> str:
    request = Request(
        f"https://api.github.com/repos/{repository}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "DFU-RadialAdapter-Colab",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"GitHub token validation failed with HTTP {exc.code}. Training was NOT started.") from exc
    except URLError as exc:
        raise RuntimeError(f"GitHub token validation could not reach GitHub: {exc}. Training was NOT started.") from exc

    permissions = payload.get("permissions") or {}
    if not permissions.get("push", False):
        raise RuntimeError(
            "The GitHub token can read the repository but does not have push permission. "
            "Grant repository Contents: Read and write, then rerun. Training was NOT started."
        )
    owner = (payload.get("owner") or {}).get("login", "authenticated user")
    print(f"GitHub token verified with push access for {repository} ({owner}).")
    return token


def launch_radial_pilot(repository_dir: str | Path = "/content/DFU-ImageGuard-radial") -> dict[str, Any]:
    project_root = mount_drive_verified()
    token = _validate_token(_read_secret_safely())
    os.environ["GITHUB_TOKEN"] = token

    from .radial_pilot_runner import PilotSettings, run_radial_adapter_pilot

    settings = PilotSettings(
        run_id="RADIAL_ADAPTER_PILOT_V1",
        drive_root=str(project_root),
        secondary_drive_root="/content/drive/MyDrive/DFU-ImageGuard-Backup",
        seeds=(2026, 2027),
        outer_fold=0,
        max_epochs=25,
        patience=7,
        batch_size=16,
        num_workers=2,
        github_export=True,
        github_export_required=True,
        github_branch="radial-pilot-results",
        github_chunk_full_checkpoints=True,
        github_chunk_bytes=48 * 1024 * 1024,
        github_export_after_each_trial=True,
        require_secondary_drive_backup=True,
    )
    return run_radial_adapter_pilot(settings=settings, repository_dir=Path(repository_dir))
