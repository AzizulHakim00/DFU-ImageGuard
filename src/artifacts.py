from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_data import Config, json_default, read_json, relpath, sha256_file, write_json


def write_external_validation_status(dirs: dict[str, Path]) -> None:
    text = """# External Validation Status

A genuine external validation experiment was not run because no scientifically compatible independent dataset with a matching binary DFU-versus-normal task was supplied to this run. The pipeline does not relabel a random internal holdout as external data. Future external evaluation must use frozen preprocessing, weights, fold ensemble/calibrators, and the development-selected threshold, with an independent overlap audit before inference.
"""
    (dirs["root"] / "EXTERNAL_VALIDATION_STATUS.md").write_text(text, encoding="utf-8")


def software_hardware_versions() -> dict[str, Any]:
    import importlib.metadata as metadata
    import torch

    packages = [
        "torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn",
        "scipy", "matplotlib", "pillow", "imagehash", "kagglehub", "shap",
        "lime", "grad-cam",
    ]
    versions: dict[str, Any] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except Exception:
            versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def dataset_manifest(
    manifest: pd.DataFrame,
    cfg: Config,
    dataset_root: Path,
) -> dict[str, Any]:
    ordered = manifest.sort_values("relative_path")
    digest = hashlib.sha256()
    for _, row in ordered.iterrows():
        digest.update(str(row.relative_path).encode())
        digest.update(str(row.file_sha256).encode())
        digest.update(str(int(row.label)).encode())
        digest.update(str(row.group_id).encode())
    return {
        "title": cfg.DATASET_TITLE,
        "handle": cfg.DATASET_HANDLE,
        "download_root": str(dataset_root),
        "license": cfg.DATASET_LICENSE,
        "citations": cfg.DATASET_CITATIONS,
        "included_subset": "Patches only",
        "class_mapping": {"0": "Normal", "1": "DFU"},
        "counts": ordered.label_name.value_counts().to_dict(),
        "n_images": len(ordered),
        "n_duplicate_groups": ordered.group_id.nunique(),
        "dataset_content_sha256": digest.hexdigest(),
        "patient_ids_available": False,
        "label_policy": "Strict mapping from immediate class folders under Patches; no keyword/root-folder inference.",
    }


def write_model_card(cfg: Config, dirs: dict[str, Path], metrics: pd.DataFrame) -> None:
    row = metrics[(metrics.model == cfg.PRIMARY_MODEL_NAME) & (metrics.state == "calibrated")]
    summary = row.iloc[0].to_dict() if len(row) else {}
    text = f"""# DFU-ImageGuard Model Card

## Intended use
Retrospective research on binary classification of explicitly labelled DFU versus normal image patches. It is not a clinical device and is not deployment-ready.

## Evaluation design
Nested, stratified, duplicate-group-aware five-fold out-of-fold evaluation. The outer test fold is isolated from augmentation, early stopping, temperature scaling and threshold selection. The primary model has one saved checkpoint per outer fold.

## Dataset limitation
The public dataset does not provide patient identifiers. Therefore, this project does **not** claim patient-level splitting. Exact and near-duplicate groups are kept within a fold. The Kaggle data card does not declare a license; redistribution and downstream use must be checked by the researcher.

## Primary calibrated OOF summary
```json
{json.dumps(summary, indent=2, default=json_default)}
```

## Safety
Predictions and XAI maps must not be interpreted as causality, diagnosis, lesion localization proof or clinical correctness. External validation is not claimed unless a scientifically compatible independent dataset is explicitly supplied and evaluated with frozen weights, preprocessing, calibration and threshold.
"""
    (dirs["root"] / "MODEL_CARD.md").write_text(text, encoding="utf-8")


def write_limitations(cfg: Config, dirs: dict[str, Path]) -> None:
    text = f"""# Limitations

- Patient/case identifiers are unavailable, so patient-level separation cannot be verified.
- Duplicate-group splitting reduces identifiable leakage but cannot guarantee that all source-image derivatives are detected.
- The selected public Kaggle dataset has no declared license on its data card.
- The task uses curated image patches and may not represent full-foot photographs or clinical acquisition workflows.
- Internal five-fold OOF performance is not external validation.
- Fold variation is not multi-seed stability. Genuine multi-seed analysis requires additional training.
- Calibration and threshold selection are development-only procedures and may shift under domain change.
- XAI visualizations are post-hoc explanations and do not prove causality or clinical correctness.
- No sensitive attributes, hospitals, patient variables or modalities are inferred or fabricated.
- Storage mode for this run: `{cfg.STORAGE_MODE}`. A local fallback is not persistent after the runtime ends unless exported.
- This retrospective research pipeline is not a medical device and is not clinically deployment-ready.
"""
    (dirs["root"] / "LIMITATIONS.md").write_text(text, encoding="utf-8")


def artifact_manifest(root: Path) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp"):
            items.append({
                "path": relpath(path, root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return {
        "created_at": dt.datetime.now().isoformat(),
        "root": str(root),
        "files": items,
    }


def _read_csv_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    return frame.replace({np.nan: None}).to_dict("records")


def create_reproducibility_pkl(
    cfg: Config,
    dirs: dict[str, Path],
    dataset_meta: dict[str, Any],
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    cis: pd.DataFrame,
    stats: pd.DataFrame,
    robustness: pd.DataFrame,
    xai: pd.DataFrame,
    uncertainty: pd.DataFrame,
    calibrations: list[dict[str, Any]],
) -> Path:
    pkl_path = dirs["root"] / "dfu_imageguard_complete_reproducibility.pkl"
    payload = {
        "schema_version": "1.1",
        "created_at": dt.datetime.now().isoformat(),
        "configuration": asdict(cfg),
        "seeds": {
            "global": cfg.SEED,
            "outer_fold_rule": "global+fold+model_hash",
            "dataloader_workers_seeded": True,
        },
        "dataset_metadata": dataset_meta,
        "fold_assignments": manifest[[
            "image_id", "group_id", "label", "outer_fold", "relative_path"
        ]].to_dict("records"),
        "duplicate_screening": {
            "cluster_summary": _read_csv_records(dirs["tables"] / "duplicate_clusters.csv"),
            "pair_count": len(_read_csv_records(dirs["tables"] / "duplicate_pairs.csv")),
            "excluded_images": _read_csv_records(dirs["manifests"] / "excluded_images.csv"),
        },
        "split_integrity_report": read_json(dirs["root"] / "split_integrity_report.json", {}),
        "oof_predictions": predictions.drop(columns=["image_path"], errors="ignore").to_dict("records"),
        "calibration_parameters": calibrations,
        "metrics": metrics.to_dict("records"),
        "confidence_intervals": cis.to_dict("records"),
        "statistical_comparisons": stats.to_dict("records"),
        "uncertainty": uncertainty.to_dict("records"),
        "robustness": robustness.to_dict("records"),
        "subgroup_analysis": {
            "performed": False,
            "reason": "No legitimate clinical subgroup metadata provided.",
        },
        "xai_metadata": xai.to_dict("records"),
        "checkpoint_paths": [str(path) for path in sorted(dirs["models"].glob("*"))],
        "figure_paths": [str(path) for path in sorted(dirs["figures"].glob("*"))],
        "table_paths": [str(path) for path in sorted(dirs["tables"].glob("*"))],
        "software_hardware_versions": software_hardware_versions(),
        "limitations": (dirs["root"] / "LIMITATIONS.md").read_text(encoding="utf-8"),
        "warnings": [
            "No raw images are stored in this PKL.",
            "Do not load untrusted pickle files.",
            "Internal OOF performance is not external validation or clinical readiness.",
        ],
    }
    temporary = pkl_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, pkl_path)
    return pkl_path


def copy_for_github(
    run_root: Path,
    repo_dir: Path,
    cfg: Config,
) -> tuple[Path, list[dict[str, Any]]]:
    target = repo_dir / "results" / "runs" / str(cfg.RUN_ID)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    skipped: list[dict[str, Any]] = []
    for source in run_root.rglob("*"):
        if not source.is_file() or source.name.endswith(".tmp"):
            continue
        relative = source.relative_to(run_root)
        size = source.stat().st_size
        if size >= cfg.GITHUB_MAX_BYTES:
            skipped.append({
                "path": str(relative),
                "bytes": size,
                "sha256": sha256_file(source),
                "storage_path": str(source),
                "reason": "GitHub size guard",
            })
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if skipped:
        write_json(target / "OVERSIZED_STORAGE_ARTIFACTS.json", skipped)
    latest = repo_dir / "results" / "LATEST_RUN.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(cfg.RUN_ID) + "\n", encoding="utf-8")
    return target, skipped


def _run_git(arguments: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=check,
        capture_output=True,
        text=True,
    )


def push_to_github(
    run_root: Path,
    cfg: Config,
    dirs: dict[str, Path],
) -> dict[str, Any]:
    repo_dir = Path(cfg.LOCAL_REPO)
    status: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "commit_sha": None,
        "message": "",
    }
    try:
        from google.colab import userdata

        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = os.getenv("GITHUB_TOKEN")
    if not token:
        status["message"] = "GITHUB_TOKEN was not available; the completed storage run is unchanged."
        write_json(dirs["root"] / "github_push_status.json", status)
        return status

    try:
        if not (repo_dir / ".git").exists():
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            _run_git([
                "git", "clone", f"https://github.com/{cfg.REPO_FULL_NAME}.git", str(repo_dir)
            ])
        pull = _run_git(
            ["git", "-C", str(repo_dir), "pull", "--rebase", "origin", "main"],
            check=False,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"git pull failed: {pull.stderr[-1000:]}")

        _, skipped = copy_for_github(run_root, repo_dir, cfg)
        _run_git(["git", "-C", str(repo_dir), "config", "user.name", "DFU-ImageGuard Colab"])
        _run_git(["git", "-C", str(repo_dir), "config", "user.email", "actions@users.noreply.github.com"])
        _run_git(["git", "-C", str(repo_dir), "add", "results"])
        diff = _run_git(["git", "-C", str(repo_dir), "diff", "--cached", "--quiet"], check=False)
        if diff.returncode == 0:
            sha = _run_git(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).stdout.strip()
            status.update({
                "attempted": True,
                "success": True,
                "commit_sha": sha,
                "message": "No new GitHub-exportable changes; the completed storage run is intact.",
                "skipped_oversized": skipped,
            })
        elif diff.returncode == 1:
            _run_git([
                "git", "-C", str(repo_dir), "commit", "-m", f"Add DFU-ImageGuard run {cfg.RUN_ID}"
            ])
            raw = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            push = _run_git([
                "git", "-C", str(repo_dir), "-c",
                f"http.extraHeader=Authorization: Basic {raw}",
                "push", "origin", "HEAD:main",
            ], check=False)
            if push.returncode != 0:
                raise RuntimeError(push.stderr[-1000:])
            sha = _run_git(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).stdout.strip()
            status.update({
                "attempted": True,
                "success": True,
                "commit_sha": sha,
                "message": "Run artifacts pushed successfully.",
                "skipped_oversized": skipped,
            })
        else:
            raise RuntimeError(f"git diff failed with code {diff.returncode}: {diff.stderr}")
    except Exception as exc:
        status.update({
            "attempted": True,
            "success": False,
            "message": f"GitHub export failed without altering the completed storage run: {exc}",
        })
    write_json(dirs["root"] / "github_push_status.json", status)
    return status


def _validate_checkpoint_payload(path: Path) -> tuple[bool, str]:
    import torch

    try:
        if not path.exists() or path.stat().st_size == 0:
            return False, "missing_or_empty"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            return False, "missing_model_state_dict"
        if not payload["model_state_dict"]:
            return False, "empty_model_state_dict"
        return True, sha256_file(path)
    except Exception as exc:
        return False, repr(exc)


def verify_run(
    cfg: Config,
    dirs: dict[str, Path],
    manifest: pd.DataFrame,
    calibrations: list[dict[str, Any]],
    pkl_path: Path,
    github_status: dict[str, Any],
) -> dict[str, Any]:
    integrity = read_json(dirs["root"] / "split_integrity_report.json", {})
    proposed_calibrations = [
        item for item in calibrations if item["model"] == cfg.PRIMARY_MODEL_NAME
    ]
    expected = [
        dirs["models"] / f"dfu_imageguard_fold_{fold}.pt"
        for fold in range(1, cfg.N_FOLDS + 1)
    ]
    checkpoint_status = {
        path.name: _validate_checkpoint_payload(path)
        for path in expected
    }
    valid_checkpoints = [name for name, (valid, _) in checkpoint_status.items() if valid]
    report = {
        "run_id": cfg.RUN_ID,
        "storage_mode": cfg.STORAGE_MODE,
        "dataset_counts": manifest.label_name.value_counts().to_dict(),
        "valid_fold_count": sum(
            not fold_info["group_overlap"]
            for fold_info in integrity.get("folds", [])
        ),
        "duplicate_and_leakage_audit_status": "PASS" if integrity.get("valid") else "FAIL",
        "number_of_proposed_model_fits_in_this_session": sum(
            bool(item.get("trained_now")) for item in proposed_calibrations
        ),
        "number_of_primary_fold_checkpoints_expected": cfg.N_FOLDS,
        "number_of_valid_primary_checkpoints": len(valid_checkpoints),
        "checkpoint_validation": checkpoint_status,
        "pkl_path": str(pkl_path),
        "storage_path": str(dirs["root"]),
        "github_push_status": github_status.get("success", False),
        "final_commit_sha": github_status.get("commit_sha"),
    }
    if len(proposed_calibrations) != cfg.N_FOLDS:
        raise AssertionError(
            f"Expected {cfg.N_FOLDS} primary fold calibration records; found {len(proposed_calibrations)}"
        )
    if len(valid_checkpoints) != cfg.N_FOLDS:
        raise AssertionError(
            f"Expected {cfg.N_FOLDS} valid primary checkpoints; found {len(valid_checkpoints)}"
        )
    if not pkl_path.exists() or pkl_path.stat().st_size == 0:
        raise AssertionError("Reproducibility PKL is missing or empty")
    write_json(dirs["root"] / "final_verification.json", report)
    print("\n" + "=" * 72)
    print("FINAL VERIFICATION")
    for key, value in report.items():
        print(f"{key}: {value}")
    print("=" * 72)
    return report
