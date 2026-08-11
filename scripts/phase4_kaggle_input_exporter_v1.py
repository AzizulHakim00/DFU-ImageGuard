from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

VERSION = "DFU_PHASE4_KAGGLE_INPUT_EXPORTER_V1_20260812"
RUN_ID = "RELIABLE_DFU_CV_V3_MISSING38"
DRIVE_ROOT = Path("/content/drive/MyDrive/DFU-ImageGuard")
RUN_ROOT = DRIVE_ROOT / "runs" / RUN_ID
PHASE2_ROOT = RUN_ROOT / "PHASE2_FULL"
EXPECTED_CHECKPOINTS = 7
EXPORT_ROOT = RUN_ROOT / "PHASE4_KAGGLE_INPUT_EXPORT"
STAGING = EXPORT_ROOT / "DFU_PHASE4_FROZEN_INPUT"
ZIP_PATH = EXPORT_ROOT / "DFU_PHASE4_FROZEN_INPUT.zip"


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def mount_drive() -> None:
    if Path("/content/drive/MyDrive").is_dir():
        return
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)


def ensure_kaggle_cli() -> None:
    import importlib.util
    if importlib.util.find_spec("kaggle") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle>=2.0.0"])


def authenticate_kaggle_cli() -> None:
    ensure_kaggle_cli()
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        try:
            from google.colab import userdata
            token = str(userdata.get("KAGGLE_API_TOKEN") or "").strip()
        except Exception:
            token = ""
    if not token:
        import getpass
        token = getpass.getpass("Paste your Kaggle API token (hidden input): ").strip()
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required for the private Kaggle upload")
    os.environ["KAGGLE_API_TOKEN"] = token


def prepare_bundle() -> tuple[Path, dict]:
    inv_path = PHASE2_ROOT / "tables" / "14_checkpoint_inventory.csv"
    oof_path = RUN_ROOT / "tables" / "all_oof_predictions.csv"
    metrics_path = RUN_ROOT / "tables" / "fold_seed_metrics.csv"
    phase2_ver = PHASE2_ROOT / "PHASE2_FULL_VERIFICATION.json"
    primary_ver = RUN_ROOT / "REPAIR45_FINAL_VERIFICATION.json"
    required = [inv_path, oof_path, metrics_path, phase2_ver]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Required frozen input missing: {p}")

    inv = pd.read_csv(inv_path)
    usable = inv.loc[(inv["exists"] == True) & (inv["metadata_valid"] == True)].copy()
    if len(usable) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"Expected exactly {EXPECTED_CHECKPOINTS} verified checkpoints; found {len(usable)}")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    (STAGING / "checkpoints").mkdir(parents=True, exist_ok=True)
    (STAGING / "tables").mkdir(parents=True, exist_ok=True)

    rewritten = []
    file_manifest = []
    for _, r in usable.sort_values(["model_key", "seed", "outer_fold"]).iterrows():
        src = Path(str(r["path"]))
        if not src.exists():
            raise FileNotFoundError(src)
        actual = sha256_file(src)
        if actual != str(r["sha256"]):
            raise RuntimeError(f"Checkpoint SHA mismatch before export: {src}")
        name = f"{r['model_key']}_seed{int(r['seed'])}_fold{int(r['outer_fold'])}_{src.name}"
        dst_rel = Path("checkpoints") / name
        dst = STAGING / dst_rel
        shutil.copy2(src, dst)
        copied_sha = sha256_file(dst)
        if copied_sha != actual:
            raise RuntimeError(f"Copied checkpoint SHA mismatch: {dst}")
        rr = r.to_dict()
        rr["bundle_relative_path"] = str(dst_rel)
        rr["path_original_drive"] = str(src)
        rr["path"] = str(dst_rel)
        rewritten.append(rr)
        file_manifest.append({"path": str(dst_rel), "size": dst.stat().st_size, "sha256": copied_sha, "role": "checkpoint"})

    inv_out = pd.DataFrame(rewritten)
    inv_out.to_csv(STAGING / "tables" / "14_checkpoint_inventory.csv", index=False)
    shutil.copy2(oof_path, STAGING / "tables" / "all_oof_predictions.csv")
    shutil.copy2(metrics_path, STAGING / "tables" / "fold_seed_metrics.csv")
    shutil.copy2(phase2_ver, STAGING / "PHASE2_FULL_VERIFICATION.json")
    if primary_ver.exists():
        shutil.copy2(primary_ver, STAGING / "REPAIR45_FINAL_VERIFICATION.json")

    for rel in [
        Path("tables/14_checkpoint_inventory.csv"),
        Path("tables/all_oof_predictions.csv"),
        Path("tables/fold_seed_metrics.csv"),
        Path("PHASE2_FULL_VERIFICATION.json"),
        Path("REPAIR45_FINAL_VERIFICATION.json"),
    ]:
        p = STAGING / rel
        if p.exists():
            file_manifest.append({"path": str(rel), "size": p.stat().st_size, "sha256": sha256_file(p), "role": "table_or_verification"})

    manifest = {
        "status": "PASS",
        "version": VERSION,
        "run_id": RUN_ID,
        "created_at_unix": time.time(),
        "verified_checkpoint_count": int(len(inv_out)),
        "primary_training_performed": False,
        "primary_trials_modified": False,
        "purpose": "Frozen read-only inputs for DFU Phase-4 external evaluation on Kaggle/Colab",
        "files": file_manifest,
    }
    (STAGING / "FROZEN_INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for item in file_manifest:
        p = STAGING / item["path"]
        if not p.exists() or sha256_file(p) != item["sha256"]:
            raise RuntimeError(f"Frozen bundle verification failed: {p}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    tmp = ZIP_PATH.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(STAGING.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(STAGING)))
    os.replace(tmp, ZIP_PATH)
    return STAGING, manifest


def upload_private_dataset(staging: Path) -> str:
    authenticate_kaggle_cli()
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not username:
        try:
            from google.colab import userdata
            username = str(userdata.get("KAGGLE_USERNAME") or "").strip()
        except Exception:
            username = ""
    if not username:
        cfg = Path.home() / ".kaggle" / "kaggle.json"
        if cfg.exists():
            try:
                username = str(json.loads(cfg.read_text())["username"]).strip()
            except Exception:
                username = ""
    if not username:
        username = input("Enter your Kaggle username (not password): ").strip()
    if not username:
        raise RuntimeError("Kaggle username is required")

    handle = f"{username}/dfu-phase4-frozen-input"
    metadata = {
        "title": "DFU Phase4 Frozen Input",
        "id": handle,
        "licenses": [{"name": "unknown"}],
        "description": (
            "PRIVATE research-only frozen model/checkpoint and evaluation metadata bundle for "
            "DFU Phase-4 external validation. It contains no newly downloaded external dataset. "
            "Licensing follows the underlying project assets; this private transfer bundle is not a public redistribution."
        ),
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    status = subprocess.run(
        ["kaggle", "datasets", "status", handle],
        text=True, capture_output=True
    )
    if status.returncode == 0:
        cmd = [
            "kaggle", "datasets", "version", "-p", str(staging),
            "-m", f"{VERSION}: verified frozen Phase-4 inputs", "-r", "zip"
        ]
        action = "version"
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(staging), "-r", "zip"]
        action = "create"

    print(f"Kaggle private dataset {action}: {handle}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle private dataset upload failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{result.stderr}"
        )
    print("Private Kaggle frozen-input handle:", handle)
    return handle


def run() -> None:
    mount_drive()
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print(VERSION)
    print("READ-ONLY EXPORTER: NO TRAINING, NO MODEL MODIFICATION, NO DATASET DOWNLOAD")
    print("=" * 100)
    staging, manifest = prepare_bundle()
    print("Frozen bundle verification: PASS")
    print("Checkpoints:", manifest["verified_checkpoint_count"])
    print("Staging folder:", staging)
    print("ZIP backup:", ZIP_PATH)
    handle = upload_private_dataset(staging)
    receipt = {
        "status": "PASS",
        "version": VERSION,
        "kaggle_dataset_handle": handle,
        "zip_path": str(ZIP_PATH),
        "staging_path": str(staging),
        "verified_checkpoint_count": manifest["verified_checkpoint_count"],
    }
    (EXPORT_ROOT / "KAGGLE_UPLOAD_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("=" * 100)
    print("KAGGLE INPUT EXPORT: PASS")
    print("Private Kaggle dataset handle:", handle)
    print("Next: open the universal Phase-4 notebook in Kaggle or Colab.")
    print("=" * 100)


if __name__ == "__main__":
    run()
