from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


def _run(args: list[str], *, cwd: Path | None = None, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _is_mountpoint(path: Path) -> bool:
    try:
        return _run(["mountpoint", "-q", str(path)]).returncode == 0
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
            _run(["fusermount", "-uz", str(mount_point)])
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


def _secret_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata

        value = userdata.get(name)
        return str(value).strip() if value else ""
    except Exception:
        return ""


def _token_from_gh_cli() -> str:
    if shutil.which("gh") is None:
        return ""
    result = _run(["gh", "auth", "token"])
    return result.stdout.strip() if result.returncode == 0 else ""


def _token_from_git_credential() -> str:
    result = _run(
        ["git", "credential", "fill"],
        input_text="protocol=https\nhost=github.com\n\n",
    )
    if result.returncode != 0:
        return ""
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields.get("password", "")


def discover_github_auth() -> tuple[str, str]:
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
        token = _secret_value(name)
        if token:
            return token, name

    token = _token_from_gh_cli()
    if token:
        return token, "gh auth"

    token = _token_from_git_credential()
    if token:
        return token, "git credential helper"

    return "", "none"


def _push_dry_run(repository_dir: Path, token: str = "") -> tuple[bool, str]:
    branch = f"radial-auth-check-{uuid.uuid4().hex[:12]}"
    command = ["git", "push", "--dry-run", "origin", f"HEAD:refs/heads/{branch}"]
    if token:
        import base64

        raw = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        command = [
            "git",
            "-c",
            f"http.extraHeader=Authorization: Basic {raw}",
            "push",
            "--dry-run",
            "origin",
            f"HEAD:refs/heads/{branch}",
        ]
    result = _run(command, cwd=repository_dir)
    message = (result.stderr or result.stdout).strip()
    return result.returncode == 0, message[-600:]


def configure_github_backup(repository_dir: Path) -> tuple[bool, str]:
    token, source = discover_github_auth()

    # First test the already configured terminal/git authentication exactly as the user has it.
    ok, message = _push_dry_run(repository_dir, token="")
    if ok:
        os.environ.pop("GITHUB_TOKEN", None)
        print("GitHub push access verified using existing terminal/Git credentials.")
        return True, "existing git credentials"

    # Then test any token discovered from environment, Colab Secrets, gh, or credential helper.
    if token:
        ok, token_message = _push_dry_run(repository_dir, token=token)
        if ok:
            os.environ["GITHUB_TOKEN"] = token
            print(f"GitHub push access verified using {source}.")
            return True, source
        message = token_message or message

    print("GitHub push authentication is not currently usable.")
    print("Training WILL continue because primary and SHA-256-verified secondary Drive backups are active.")
    print("GitHub export will be marked PENDING and can be synced later without retraining.")
    if message:
        print("Git authentication detail:", message.replace(token, "***") if token else message)
    os.environ.pop("GITHUB_TOKEN", None)
    return False, "pending authentication"


def launch_radial_pilot(repository_dir: str | Path = "/content/DFU-ImageGuard-radial") -> dict[str, Any]:
    project_root = mount_drive_verified()
    repository_dir = Path(repository_dir)
    github_ready, auth_source = configure_github_backup(repository_dir)

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
        github_export=github_ready,
        github_export_required=False,
        github_branch="radial-pilot-results",
        github_chunk_full_checkpoints=True,
        github_chunk_bytes=48 * 1024 * 1024,
        github_export_after_each_trial=True,
        require_secondary_drive_backup=True,
        github_auth_source=auth_source,
    )
    return run_radial_adapter_pilot(settings=settings, repository_dir=repository_dir)
