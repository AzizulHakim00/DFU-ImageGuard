from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .reliability_io import atomic_json


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunk_file_for_git(
    source: Path,
    destination: Path,
    chunk_bytes: int,
) -> dict[str, Any]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    parts = []
    with source.open("rb") as handle:
        index = 0
        while True:
            payload = handle.read(chunk_bytes)
            if not payload:
                break
            part = destination / f"part-{index:04d}.bin"
            part.write_bytes(payload)
            parts.append(
                {
                    "name": part.name,
                    "bytes": len(payload),
                    "sha256": _sha256(part),
                }
            )
            index += 1
    manifest = {
        "original_name": source.name,
        "original_bytes": int(source.stat().st_size),
        "original_sha256": _sha256(source),
        "chunk_bytes": int(chunk_bytes),
        "parts": parts,
    }
    (destination / "chunks.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def reconstruct_chunked_file(
    chunk_dir: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    chunk_dir = Path(chunk_dir)
    destination = Path(destination)
    manifest_path = chunk_dir / "chunks.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for part_info in manifest["parts"]:
            part = chunk_dir / part_info["name"]
            if not part.is_file():
                raise FileNotFoundError(part)
            if _sha256(part) != part_info["sha256"]:
                raise RuntimeError(f"Chunk SHA-256 mismatch: {part}")
            output.write(part.read_bytes())
    if temporary.stat().st_size != int(manifest["original_bytes"]):
        raise RuntimeError("Reconstructed byte count does not match manifest")
    if _sha256(temporary) != manifest["original_sha256"]:
        raise RuntimeError("Reconstructed SHA-256 does not match manifest")
    os.replace(temporary, destination)
    return {
        "success": True,
        "destination": str(destination),
        "bytes": int(destination.stat().st_size),
        "sha256": _sha256(destination),
    }


def _git(args: list[str], *, check: bool = True):
    return subprocess.run(args, text=True, capture_output=True, check=check)


def _discover_token() -> tuple[str, str]:
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
        value = os.environ.get(key, "").strip()
        if value:
            return value, f"environment:{key}"
    try:
        from google.colab import userdata
        for key in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
            try:
                value = str(userdata.get(key)).strip()
            except Exception:
                value = ""
            if value:
                return value, f"colab_secret:{key}"
    except Exception:
        pass
    return "", "git_credential_or_gh"


def _manifest(root: Path) -> dict[str, dict[str, Any]]:
    import hashlib

    output = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        output[str(path.relative_to(root))] = {
            "bytes": int(path.stat().st_size),
            "sha256": digest,
        }
    return output


def sync_reliability_run_to_github(
    *,
    run_root: str | Path,
    repository: str = "AzizulHakim00/DFU-ImageGuard",
    branch: str = "reliability-results",
    chunk_bytes: int = 48 * 1024 * 1024,
) -> dict[str, Any]:
    run_root = Path(run_root)
    verification = run_root / "FINAL_RELIABILITY_VERIFICATION.json"
    if not verification.exists():
        raise FileNotFoundError(
            "FINAL_RELIABILITY_VERIFICATION.json is missing; "
            "refusing to sync an incomplete run"
        )
    status = json.loads(verification.read_text())
    if not status.get("verification_passed"):
        raise RuntimeError("The saved run did not pass final verification")

    work = Path("/content/DFU-reliability-results-export")
    if work.exists():
        shutil.rmtree(work)
    _git(["git", "clone", f"https://github.com/{repository}.git", str(work)])
    remote_branch = _git(
        ["git", "-C", str(work), "ls-remote", "--heads", "origin", branch],
        check=False,
    ).stdout.strip()
    if remote_branch:
        _git(["git", "-C", str(work), "fetch", "origin", branch])
        _git(
            [
                "git", "-C", str(work), "checkout", "-B", branch,
                f"origin/{branch}",
            ]
        )
    else:
        _git(["git", "-C", str(work), "checkout", "-b", branch])

    target = work / "results" / "reliability" / run_root.name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for source in run_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(run_root)
        if source.suffix == ".pt":
            chunk_file_for_git(
                source,
                target / (str(relative) + ".chunks"),
                chunk_bytes,
            )
            if "portable_fp16" not in source.name:
                continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    atomic_json(target / "GITHUB_EXPORT_MANIFEST.json", _manifest(target))

    _git([
        "git", "-C", str(work), "config", "user.name",
        "DFU Reliability Export",
    ])
    _git([
        "git", "-C", str(work), "config", "user.email",
        "dfu-reliability@users.noreply.github.com",
    ])
    _git(["git", "-C", str(work), "add", "--", "results/reliability"])
    diff = _git(
        ["git", "-C", str(work), "diff", "--cached", "--quiet"],
        check=False,
    )
    if diff.returncode == 1:
        _git([
            "git", "-C", str(work), "commit", "-m",
            f"Backup verified reliability run {run_root.name}",
        ])

    token, auth_source = _discover_token()
    push_command = [
        "git", "-C", str(work), "push", "origin", f"HEAD:{branch}",
    ]
    if token:
        encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        push_command[3:3] = [
            "-c", f"http.extraHeader=Authorization: Basic {encoded}"
        ]
    push = _git(push_command, check=False)
    if push.returncode != 0:
        raise RuntimeError(
            "GitHub sync authentication failed. The verified Drive run remains "
            "safe and no retraining is needed.\n"
            + (push.stderr or push.stdout)[-800:]
        )
    commit = _git(["git", "-C", str(work), "rev-parse", "HEAD"]).stdout.strip()
    result = {
        "success": True,
        "repository": repository,
        "branch": branch,
        "commit": commit,
        "auth_source": auth_source,
        "run_root": str(run_root),
        "retraining_performed": False,
    }
    atomic_json(run_root / "GITHUB_SYNC_STATUS.json", result)
    return result
