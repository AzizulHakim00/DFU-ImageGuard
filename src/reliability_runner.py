from __future__ import annotations

import json
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config_data import make_inner_partition, seed_everything
from .reliability_io import (
    DIRECTORIES,
    ReliabilitySettings,
    _file_manifest,
    _prepare_locked_manifest,
    _resolved_config,
    atomic_csv,
    atomic_json,
    mirror_full_run,
    persistence_preflight,
)
from .reliability_reporting_core import aggregate_outputs
from .reliability_robustness import _run_trial_robustness
from .reliability_training import _train_and_evaluate_trial
from .reliability_xai import generate_gradcam


def _environment_snapshot(repository_dir: Path) -> dict[str, Any]:
    try:
        git_sha = subprocess.run(
            ["git", "-C", str(repository_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"
    packages = {}
    for name in [
        "torch", "torchvision", "timm", "numpy", "pandas",
        "scikit-learn", "scipy", "Pillow",
    ]:
        try:
            from importlib.metadata import version
            packages[name] = version(name)
        except Exception:
            packages[name] = "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": git_sha,
        "packages": packages,
    }


def run_reliability_experiment(
    settings: ReliabilitySettings | None = None,
    *,
    repository_dir: str | Path = "/content/DFU-ImageGuard-reliability",
) -> dict[str, Any]:
    settings = settings or ReliabilitySettings()
    start_time = time.time()
    run_root, dirs = persistence_preflight(settings)
    cfg = _resolved_config(settings)
    seed_everything(settings.seeds[0])
    atomic_json(
        dirs["configs"] / "environment.json",
        _environment_snapshot(Path(repository_dir)),
    )
    manifest = _prepare_locked_manifest(cfg, dirs)

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    robustness_frames: list[pd.DataFrame] = []
    expected_trials = (
        len(settings.model_keys) * len(settings.seeds) * settings.n_folds
    )
    progress_path = run_root / "EXPERIMENT_PROGRESS.json"

    for model_key in settings.model_keys:
        for seed in settings.seeds:
            for fold in range(settings.n_folds):
                outer_train = manifest[manifest.outer_fold != fold].copy()
                outer_test = manifest[
                    manifest.outer_fold == fold
                ].copy().reset_index(drop=True)
                inner = make_inner_partition(outer_train, cfg, fold)
                inner["protocol_seed"] = int(seed)
                inner_path = (
                    dirs["manifests"]
                    / f"inner_partition_seed_{seed}_fold_{fold + 1}.csv"
                )
                if not inner_path.exists():
                    atomic_csv(inner, inner_path)
                train_df = inner[inner.inner_role == "train"].copy()
                selection_df = inner[
                    inner.inner_role == "selection"
                ].copy()
                calibration_df = inner[
                    inner.inner_role == "calibration"
                ].copy()
                metrics, predictions = _train_and_evaluate_trial(
                    settings=settings,
                    cfg=cfg,
                    run_root=run_root,
                    model_key=model_key,
                    seed=seed,
                    fold=fold,
                    train_df=train_df,
                    selection_df=selection_df,
                    calibration_df=calibration_df,
                    test_df=outer_test,
                )
                metrics_rows.append(metrics)
                prediction_frames.append(predictions)
                robustness = _run_trial_robustness(
                    settings=settings,
                    cfg=cfg,
                    run_root=run_root,
                    model_key=model_key,
                    seed=seed,
                    fold=fold,
                    test_df=outer_test,
                    temperature=float(metrics["temperature"]),
                    threshold=float(metrics["threshold"]),
                )
                if not robustness.empty:
                    robustness_frames.append(robustness)
                atomic_json(
                    progress_path,
                    {
                        "completed_trials": len(metrics_rows),
                        "expected_trials": expected_trials,
                        "last_completed": {
                            "model_key": model_key,
                            "seed": int(seed),
                            "outer_fold": int(fold + 1),
                        },
                        "full_cv_complete": len(metrics_rows) == expected_trials,
                    },
                )

    outputs = aggregate_outputs(
        settings=settings,
        run_root=run_root,
        dirs=dirs,
        metrics_rows=metrics_rows,
        prediction_frames=prediction_frames,
        robustness_frames=robustness_frames,
    )
    primary_predictions = outputs["all_predictions"][
        outputs["all_predictions"].model_key == settings.primary_model_key
    ]
    xai = generate_gradcam(
        settings=settings,
        cfg=cfg,
        run_root=run_root,
        dirs=dirs,
        primary_predictions=primary_predictions,
    )

    reproducibility_path = run_root / "reliability_reproducibility.pkl"
    with reproducibility_path.open("wb") as handle:
        pickle.dump(
            {
                "settings": asdict(settings),
                "trial_metrics": outputs["trial_metrics"].to_dict("records"),
                "oof_metrics": outputs["oof_metrics"].to_dict("records"),
                "summary": outputs["summary"].to_dict("records"),
                "selective_prediction": outputs["selective"].to_dict("records"),
                "bootstrap": outputs["bootstrap"].to_dict("records"),
                "xai_metadata": xai.to_dict("records"),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    checkpoint_count = len(list((run_root / "trials").rglob("best_model.pt")))
    resume_count = len(list((run_root / "trials").rglob("last_resume.pt")))
    portable_count = len(
        list((run_root / "trials").rglob("best_model_portable_fp16.pt"))
    )
    prediction_count = len(
        list((run_root / "trials").rglob("test_predictions.csv"))
    )
    calibration_count = len(
        list((run_root / "trials").rglob("calibration_predictions.csv"))
    )
    metrics_count = len(list((run_root / "trials").rglob("metrics.json")))
    secondary = mirror_full_run(run_root, settings)
    final = {
        "run_id": settings.run_id,
        "status": "COMPLETE",
        "completed_trials": len(metrics_rows),
        "expected_trials": expected_trials,
        "primary_model_key": settings.primary_model_key,
        "architecture_search_closed": True,
        "best_checkpoint_count": checkpoint_count,
        "resume_checkpoint_count": resume_count,
        "portable_checkpoint_count": portable_count,
        "prediction_file_count": prediction_count,
        "calibration_file_count": calibration_count,
        "metrics_file_count": metrics_count,
        "primary_drive": str(run_root),
        "secondary_backup": secondary,
        "reproducibility_pickle": str(reproducibility_path),
        "external_validation_status": "READY_FOR_FROZEN_EXTERNAL_MANIFEST",
        "github_sync_status": "SEPARATE_NO_RETRAINING_SYNC_NOTEBOOK",
        "elapsed_minutes": (time.time() - start_time) / 60,
        "verification_passed": (
            len(metrics_rows) == expected_trials
            and checkpoint_count == expected_trials
            and resume_count == expected_trials
            and portable_count == expected_trials
            and prediction_count == expected_trials
            and calibration_count == expected_trials
            and metrics_count == expected_trials
            and bool(secondary.get("verified"))
        ),
    }
    atomic_json(run_root / "FINAL_RELIABILITY_VERIFICATION.json", final)
    atomic_json(run_root / "ARTIFACT_MANIFEST.json", _file_manifest(run_root))
    mirror_full_run(run_root, settings)
    if not final["verification_passed"]:
        raise RuntimeError(f"Final verification failed: {final}")
    print(json.dumps(final, indent=2, default=str))
    return final


def regenerate_reliability_outputs(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    settings = ReliabilitySettings(
        **json.loads((run_root / "resolved_settings.json").read_text())
    )
    predictions = pd.read_csv(
        run_root / "predictions" / "all_oof_predictions.csv"
    )
    trial_metrics = pd.read_csv(
        run_root / "tables" / "fold_seed_metrics.csv"
    )
    robustness_path = (
        run_root / "robustness" / "all_robustness_predictions.csv"
    )
    robustness_frames = (
        [pd.read_csv(robustness_path)] if robustness_path.exists() else []
    )
    dirs = {
        "root": run_root,
        **{name: run_root / name for name in DIRECTORIES},
    }
    outputs = aggregate_outputs(
        settings=settings,
        run_root=run_root,
        dirs=dirs,
        metrics_rows=trial_metrics.to_dict("records"),
        prediction_frames=[predictions],
        robustness_frames=robustness_frames,
    )
    result = {
        "run_root": str(run_root),
        "regenerated_without_training": True,
        "models": list(settings.model_keys),
        "seeds": list(settings.seeds),
        "oof_rows": int(len(outputs["all_predictions"])),
    }
    atomic_json(run_root / "REGENERATION_STATUS.json", result)
    return result
