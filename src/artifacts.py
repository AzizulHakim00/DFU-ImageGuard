from __future__ import annotations
import base64, dataclasses, datetime as dt, gc, hashlib, json, math, os, pickle, platform, random, shutil, subprocess, sys, time, traceback, warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
warnings.filterwarnings("ignore", category=UserWarning)

from .config_data import *
from .evaluation import *

def write_external_validation_status(dirs: dict[str, Path]) -> None:
    text = """# External Validation Status

A genuine external validation experiment was not run because no scientifically compatible independent dataset with a matching binary DFU-versus-normal task was supplied to this run. The pipeline does not relabel a random internal holdout as external data. Future external evaluation must use frozen preprocessing, weights, fold ensemble/calibrators, and the development-selected threshold, with an independent overlap audit before inference.
"""
    (dirs["root"] / "EXTERNAL_VALIDATION_STATUS.md").write_text(text, encoding="utf-8")


def software_hardware_versions() -> dict[str, Any]:
    import importlib.metadata as md
    import torch
    packages = ["torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "scipy",
                "matplotlib", "pillow", "imagehash", "kagglehub", "shap", "lime", "grad-cam"]
    versions = {}
    for pkg in packages:
        try: versions[pkg] = md.version(pkg)
        except Exception: versions[pkg] = None
    return {
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "torch_cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def dataset_manifest(manifest: pd.DataFrame, cfg: Config, dataset_root: Path) -> dict[str, Any]:
    ordered = manifest.sort_values("relative_path")
    combined = hashlib.sha256("".join(ordered.file_sha256.astype(str)).encode()).hexdigest()
    return {
        "title": cfg.DATASET_TITLE, "handle": cfg.DATASET_HANDLE, "download_root": str(dataset_root),
        "license": cfg.DATASET_LICENSE, "citations": cfg.DATASET_CITATIONS,
        "included_subset": "Patches only", "class_mapping": {"0": "Normal", "1": "DFU"},
        "counts": ordered.label_name.value_counts().to_dict(), "n_images": len(ordered),
        "n_duplicate_groups": ordered.group_id.nunique(), "dataset_content_sha256": combined,
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
    text = """# Limitations

- Patient/case identifiers are unavailable, so patient-level separation cannot be verified.
- Duplicate-group splitting reduces identifiable leakage but cannot guarantee that all source-image derivatives are detected.
- The selected public Kaggle dataset has no declared license on its data card.
- The task uses curated image patches and may not represent full-foot photographs or clinical acquisition workflows.
- Internal five-fold OOF performance is not external validation.
- Fold variation is not multi-seed stability. Genuine multi-seed analysis requires additional training.
- Calibration and threshold selection are development-only procedures and may shift under domain change.
- XAI visualizations are post-hoc explanations and do not prove causality or clinical correctness.
- No sensitive attributes, hospitals, patient variables or modalities are inferred or fabricated.
- This retrospective research pipeline is not a medical device and is not clinically deployment-ready.
"""
    (dirs["root"] / "LIMITATIONS.md").write_text(text, encoding="utf-8")


def artifact_manifest(root: Path) -> dict[str, Any]:
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            items.append({"path": relpath(p, root), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return {"created_at": dt.datetime.now().isoformat(), "root": str(root), "files": items}


def create_reproducibility_pkl(cfg: Config, dirs: dict[str, Path], dataset_meta: dict[str, Any],
                               manifest: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame,
                               cis: pd.DataFrame, stats: pd.DataFrame, robustness: pd.DataFrame,
                               xai: pd.DataFrame, uncertainty: pd.DataFrame, calibrations: list[dict[str, Any]]) -> Path:
    pkl_path = dirs["root"] / "dfu_imageguard_complete_reproducibility.pkl"
    payload = {
        "schema_version": "1.0", "created_at": dt.datetime.now().isoformat(),
        "configuration": asdict(cfg), "seeds": {"global": cfg.SEED, "outer_fold_rule": "global+fold+model_hash"},
        "dataset_metadata": dataset_meta,
        "fold_assignments": manifest[["image_id", "group_id", "label", "outer_fold", "relative_path"]].to_dict("records"),
        "duplicate_screening": read_json(dirs["root"] / "split_integrity_report.json", {}),
        "split_integrity_report": read_json(dirs["root"] / "split_integrity_report.json", {}),
        "oof_predictions": predictions.drop(columns=["image_path"], errors="ignore").to_dict("records"),
        "calibration_parameters": calibrations,
        "metrics": metrics.to_dict("records"), "confidence_intervals": cis.to_dict("records"),
        "statistical_comparisons": stats.to_dict("records"), "uncertainty": uncertainty.to_dict("records"),
        "robustness": robustness.to_dict("records"),
        "subgroup_analysis": {"performed": False, "reason": "No legitimate clinical subgroup metadata provided."},
        "xai_metadata": xai.to_dict("records"),
        "checkpoint_paths": [str(p) for p in sorted(dirs["models"].glob("*"))],
        "figure_paths": [str(p) for p in sorted(dirs["figures"].glob("*"))],
        "table_paths": [str(p) for p in sorted(dirs["tables"].glob("*"))],
        "software_hardware_versions": software_hardware_versions(),
        "limitations": (dirs["root"] / "LIMITATIONS.md").read_text(encoding="utf-8"),
        "warnings": [
            "No raw images are stored in this PKL.", "Do not load untrusted pickle files.",
            "Internal OOF performance is not external validation or clinical readiness.",
        ],
    }
    with pkl_path.open("wb") as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path


def copy_for_github(run_root: Path, repo_dir: Path, cfg: Config) -> tuple[Path, list[dict[str, Any]]]:
    target = repo_dir / "results" / "runs" / str(cfg.RUN_ID)
    if target.exists(): shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True); skipped = []
    for src in run_root.rglob("*"):
        if not src.is_file(): continue
        relative = src.relative_to(run_root); size = src.stat().st_size
        if size >= cfg.GITHUB_MAX_BYTES:
            skipped.append({"path": str(relative), "bytes": size, "sha256": sha256_file(src),
                            "drive_path": str(src), "reason": "GitHub size guard"}); continue
        dst = target / relative; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    if skipped: write_json(target / "OVERSIZED_DRIVE_ARTIFACTS.json", skipped)
    latest = repo_dir / "results" / "LATEST_RUN.txt"; latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(cfg.RUN_ID) + "\n", encoding="utf-8"); return target, skipped


def push_to_github(run_root: Path, cfg: Config, dirs: dict[str, Path]) -> dict[str, Any]:
    repo_dir = Path(cfg.LOCAL_REPO); status = {"attempted": False, "success": False, "commit_sha": None, "message": ""}
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception: token = os.getenv("GITHUB_TOKEN")
    if not token:
        status["message"] = "GITHUB_TOKEN was not available; Drive run remains complete."
        write_json(dirs["root"] / "github_push_status.json", status); return status
    try:
        if not (repo_dir / ".git").exists():
            if repo_dir.exists(): shutil.rmtree(repo_dir)
            subprocess.run(["git", "clone", f"https://github.com/{cfg.REPO_FULL_NAME}.git", str(repo_dir)],
                           check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--rebase", "origin", "main"], check=False, capture_output=True, text=True)
        _, skipped = copy_for_github(run_root, repo_dir, cfg)
        subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "DFU-ImageGuard Colab"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "actions@users.noreply.github.com"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "add", "results"], check=True)
        diff = subprocess.run(["git", "-C", str(repo_dir), "diff", "--cached", "--quiet"])
        if diff.returncode == 0:
            status.update({"attempted": True, "success": True, "message": "No new GitHub-exportable changes; Drive run is complete.", "skipped_oversized": skipped})
        else:
            subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", f"Add DFU-ImageGuard run {cfg.RUN_ID}"], check=True, capture_output=True, text=True)
            raw = base64.b64encode(f"x-access-token:{token}".encode()).decode(); env = os.environ.copy()
            push = subprocess.run(["git", "-C", str(repo_dir), "-c", f"http.extraHeader=Authorization: Basic {raw}", "push", "origin", "HEAD:main"], check=False, capture_output=True, text=True, env=env)
            if push.returncode != 0: raise RuntimeError(push.stderr[-1000:])
            sha = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            status.update({"attempted": True, "success": True, "commit_sha": sha, "message": "Run artifacts pushed successfully.", "skipped_oversized": skipped})
    except Exception as exc:
        status.update({"attempted": True, "success": False, "message": f"GitHub export failed without altering the completed Drive run: {exc}"})
    write_json(dirs["root"] / "github_push_status.json", status); return status


def verify_run(cfg: Config, dirs: dict[str, Path], manifest: pd.DataFrame,
               calibrations: list[dict[str, Any]], pkl_path: Path, github_status: dict[str, Any]) -> dict[str, Any]:
    integrity = read_json(dirs["root"] / "split_integrity_report.json", {})
    proposed_cal = [c for c in calibrations if c["model"] == cfg.PRIMARY_MODEL_NAME]
    checkpoints = sorted(dirs["models"].glob("dfu_imageguard_fold_*.pt"))
    report = {
        "run_id": cfg.RUN_ID, "dataset_counts": manifest.label_name.value_counts().to_dict(),
        "valid_fold_count": sum(not f["group_overlap"] for f in integrity.get("folds", [])),
        "duplicate_and_leakage_audit_status": "PASS" if integrity.get("valid") else "FAIL",
        "number_of_proposed_model_fits_in_this_session": sum(bool(c.get("trained_now")) for c in proposed_cal),
        "number_of_primary_fold_checkpoints_expected": cfg.N_FOLDS,
        "number_of_valid_primary_checkpoints": len(checkpoints), "pkl_path": str(pkl_path),
        "drive_path": str(dirs["root"]), "github_push_status": github_status.get("success", False),
        "final_commit_sha": github_status.get("commit_sha"),
    }
    if len(proposed_cal) != cfg.N_FOLDS or len(checkpoints) != cfg.N_FOLDS:
        raise AssertionError("The complete primary experiment must contain exactly five fold calibrations and five checkpoints.")
    write_json(dirs["root"] / "final_verification.json", report)
    print("\n" + "=" * 72); print("FINAL VERIFICATION")
    for k, v in report.items(): print(f"{k}: {v}")
    print("=" * 72); return report
