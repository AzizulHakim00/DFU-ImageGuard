from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .reliability_io import ReliabilitySettings, atomic_csv
from .reliability_metrics import (
    bootstrap_comparison_table,
    error_review_template,
    save_summary_figures,
    selective_prediction_table,
)
from .reliability_models import MODEL_SPECS
from .reliability_training import _augment_metrics


def aggregate_outputs(
    *,
    settings: ReliabilitySettings,
    run_root: Path,
    dirs: dict[str, Path],
    metrics_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
    robustness_frames: list[pd.DataFrame],
) -> dict[str, Any]:
    trial_metrics = pd.DataFrame(metrics_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    atomic_csv(trial_metrics, dirs["tables"] / "fold_seed_metrics.csv")
    atomic_csv(all_predictions, dirs["predictions"] / "all_oof_predictions.csv")

    oof_rows: list[dict[str, Any]] = []
    selective_frames: list[pd.DataFrame] = []
    for model_key in settings.model_keys:
        for seed in settings.seeds:
            frame = all_predictions[
                (all_predictions.model_key == model_key)
                & (all_predictions.seed == seed)
            ].copy()
            if frame.image_id.nunique() != len(frame):
                raise AssertionError(f"OOF duplication for {model_key}, seed={seed}")
            metrics = _augment_metrics(
                frame.label.to_numpy(dtype=int),
                frame.prob_calibrated.to_numpy(dtype=float),
                frame.pred_calibrated.to_numpy(dtype=int),
            )
            oof_rows.append(
                {
                    "model_key": model_key,
                    "model_name": MODEL_SPECS[model_key].display_name,
                    "seed": int(seed),
                    **metrics,
                }
            )
            selective = selective_prediction_table(
                frame, coverage_levels=settings.coverage_levels
            )
            selective["model_key"] = model_key
            selective["seed"] = int(seed)
            selective_frames.append(selective)
    oof_metrics = pd.DataFrame(oof_rows)
    selective_table = pd.concat(selective_frames, ignore_index=True)
    atomic_csv(oof_metrics, dirs["tables"] / "oof_metrics_by_seed.csv")
    atomic_csv(selective_table, dirs["tables"] / "selective_prediction.csv")

    numeric = [
        "accuracy", "balanced_accuracy", "sensitivity", "specificity",
        "precision", "npv", "f1", "f2", "mcc", "roc_auc", "pr_auc",
        "brier", "log_loss", "ece", "false_negative_rate",
    ]
    summary_rows: list[dict[str, Any]] = []
    for model_key, frame in oof_metrics.groupby("model_key"):
        row: dict[str, Any] = {
            "model_key": model_key,
            "model_name": MODEL_SPECS[model_key].display_name,
            "n_seeds": int(frame.seed.nunique()),
        }
        for metric in numeric:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
            row[f"{metric}_min"] = float(frame[metric].min())
            row[f"{metric}_max"] = float(frame[metric].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    atomic_csv(summary, dirs["tables"] / "model_summary_mean_sd.csv")

    baselines = tuple(
        key for key in settings.model_keys if key != settings.primary_model_key
    )
    bootstrap = bootstrap_comparison_table(
        all_predictions,
        primary_model=settings.primary_model_key,
        baseline_models=baselines,
        reps=settings.bootstrap_reps,
        base_seed=settings.seeds[0],
    )
    atomic_csv(bootstrap, dirs["tables"] / "paired_group_bootstrap.csv")

    error_template = error_review_template(
        all_predictions[all_predictions.model_key == settings.primary_model_key],
        seed=settings.seeds[0],
    )
    atomic_csv(
        error_template,
        dirs["tables"] / "false_negative_error_review_template.csv",
    )

    if robustness_frames:
        robustness = pd.concat(robustness_frames, ignore_index=True)
        robust_rows: list[dict[str, Any]] = []
        for keys, frame in robustness.groupby(
            ["model_key", "seed", "corruption", "level", "severity"],
            sort=True,
        ):
            model_key, seed, corruption, level, severity = keys
            metrics = _augment_metrics(
                frame.label.to_numpy(dtype=int),
                frame.prob_calibrated.to_numpy(dtype=float),
                frame.pred_calibrated.to_numpy(dtype=int),
            )
            robust_rows.append(
                {
                    "model_key": model_key,
                    "seed": int(seed),
                    "corruption": corruption,
                    "level": int(level),
                    "severity": float(severity),
                    **metrics,
                }
            )
        robust_summary = pd.DataFrame(robust_rows)
        clean = oof_metrics[
            oof_metrics.model_key == settings.primary_model_key
        ][["seed", "sensitivity", "balanced_accuracy"]].rename(
            columns={
                "sensitivity": "clean_oof_sensitivity",
                "balanced_accuracy": "clean_oof_balanced_accuracy",
            }
        )
        robust_summary = robust_summary.merge(clean, on="seed", how="left")
        robust_summary["sensitivity_drop"] = (
            robust_summary.clean_oof_sensitivity - robust_summary.sensitivity
        )
        robust_summary["balanced_accuracy_drop"] = (
            robust_summary.clean_oof_balanced_accuracy
            - robust_summary.balanced_accuracy
        )
        atomic_csv(
            robustness,
            dirs["robustness"] / "all_robustness_predictions.csv",
        )
        atomic_csv(
            robust_summary,
            dirs["robustness"] / "robustness_summary.csv",
        )
    else:
        robustness = pd.DataFrame()

    save_summary_figures(all_predictions, selective_table, dirs["figures"])
    return {
        "trial_metrics": trial_metrics,
        "all_predictions": all_predictions,
        "oof_metrics": oof_metrics,
        "summary": summary,
        "selective": selective_table,
        "bootstrap": bootstrap,
        "robustness": robustness,
        "error_template": error_template,
    }
