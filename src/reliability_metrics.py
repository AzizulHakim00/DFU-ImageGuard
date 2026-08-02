from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    probability_superior: float
    valid_replicates: int


def metrics_from_decisions(
    y: Iterable[int],
    probability: Iterable[float],
    decision: Iterable[int],
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        fbeta_score,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_arr = np.asarray(list(y), dtype=int).reshape(-1)
    p_arr = np.clip(np.asarray(list(probability), dtype=float).reshape(-1), 1e-7, 1 - 1e-7)
    d_arr = np.asarray(list(decision), dtype=int).reshape(-1)
    if not (len(y_arr) == len(p_arr) == len(d_arr)):
        raise ValueError("y, probability and decision lengths differ")
    if len(y_arr) == 0:
        raise ValueError("Cannot compute metrics on an empty sample")
    if not np.isfinite(p_arr).all():
        raise FloatingPointError("Probability array contains NaN or infinity")

    tn, fp, fn, tp = confusion_matrix(y_arr, d_arr, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    accuracy = float(accuracy_score(y_arr, d_arr))
    both = len(np.unique(y_arr)) == 2
    return {
        "n": int(len(y_arr)),
        "prevalence": float(y_arr.mean()),
        "accuracy": accuracy,
        "error_rate": float(1 - accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(y_arr, d_arr)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1_score(y_arr, d_arr, zero_division=0)),
        "f2": float(fbeta_score(y_arr, d_arr, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_arr, d_arr)),
        "cohen_kappa": float(cohen_kappa_score(y_arr, d_arr)),
        "roc_auc": float(roc_auc_score(y_arr, p_arr)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_arr, p_arr)) if both else float("nan"),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, np.c_[1 - p_arr, p_arr], labels=[0, 1])),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def selective_prediction_table(
    prediction_frame: pd.DataFrame,
    coverage_levels: tuple[float, ...] = (1.0, 0.95, 0.90, 0.80),
) -> pd.DataFrame:
    required = {"label", "prob_calibrated", "pred_calibrated", "confidence"}
    missing = required - set(prediction_frame.columns)
    if missing:
        raise ValueError(f"Missing selective-prediction columns: {sorted(missing)}")
    frame = prediction_frame.sort_values(
        ["confidence", "image_id"], ascending=[False, True]
    ).reset_index(drop=True)
    rows: list[dict[str, float]] = []
    for requested in coverage_levels:
        if not 0 < requested <= 1:
            raise ValueError(f"Coverage must be in (0,1], got {requested}")
        keep_n = max(1, int(math.ceil(len(frame) * requested)))
        kept = frame.iloc[:keep_n]
        metrics = metrics_from_decisions(
            kept.label, kept.prob_calibrated, kept.pred_calibrated
        )
        rows.append(
            {
                "requested_coverage": float(requested),
                "actual_coverage": float(keep_n / len(frame)),
                "retained_n": int(keep_n),
                "referred_n": int(len(frame) - keep_n),
                "minimum_retained_confidence": float(kept.confidence.min()),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _metric_callable(name: str) -> Callable[[pd.DataFrame], float]:
    supported = {
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
    }
    if name not in supported:
        raise KeyError(f"Unsupported bootstrap metric: {name}")

    def compute(frame: pd.DataFrame) -> float:
        result = metrics_from_decisions(
            frame.label, frame.prob_calibrated, frame.pred_calibrated
        )
        return float(result[name])

    return compute


def paired_group_bootstrap(
    model_a: pd.DataFrame,
    model_b: pd.DataFrame,
    *,
    metric: str,
    reps: int = 2000,
    seed: int = 2026,
) -> BootstrapResult:
    key = ["image_id", "group_id", "label"]
    a = model_a[key + ["prob_calibrated", "pred_calibrated"]].rename(
        columns={
            "prob_calibrated": "prob_a",
            "pred_calibrated": "pred_a",
        }
    )
    b = model_b[key + ["prob_calibrated", "pred_calibrated"]].rename(
        columns={
            "prob_calibrated": "prob_b",
            "pred_calibrated": "pred_b",
        }
    )
    merged = a.merge(b, on=key, how="inner", validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise AssertionError("Paired bootstrap requires identical image sets")
    groups = np.asarray(sorted(merged.group_id.unique()), dtype=object)
    if len(groups) < 2:
        raise ValueError("At least two groups are required for bootstrap")
    grouped = {g: merged[merged.group_id == g] for g in groups}
    metric_fn = _metric_callable(metric)

    def view(sample: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "label": sample.label.to_numpy(),
                "prob_calibrated": sample[f"prob_{suffix}"].to_numpy(),
                "pred_calibrated": sample[f"pred_{suffix}"].to_numpy(),
            }
        )

    observed = metric_fn(view(merged, "a")) - metric_fn(view(merged, "b"))
    rng = np.random.default_rng(int(seed))
    differences: list[float] = []
    for _ in range(int(reps)):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([grouped[g] for g in sampled_groups], ignore_index=True)
        try:
            delta = metric_fn(view(sampled, "a")) - metric_fn(view(sampled, "b"))
        except ValueError:
            continue
        if np.isfinite(delta):
            differences.append(float(delta))
    if len(differences) < max(100, reps // 5):
        raise RuntimeError("Too few valid bootstrap replicates")
    array = np.asarray(differences, dtype=float)
    return BootstrapResult(
        metric=metric,
        estimate=float(observed),
        ci_low=float(np.quantile(array, 0.025)),
        ci_high=float(np.quantile(array, 0.975)),
        probability_superior=float(np.mean(array > 0)),
        valid_replicates=int(len(array)),
    )


def bootstrap_comparison_table(
    all_predictions: pd.DataFrame,
    *,
    primary_model: str,
    baseline_models: tuple[str, ...],
    metrics: tuple[str, ...] = (
        "balanced_accuracy",
        "sensitivity",
        "roc_auc",
        "brier",
    ),
    reps: int = 2000,
    base_seed: int = 2026,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in sorted(all_predictions.seed.unique()):
        primary = all_predictions[
            (all_predictions.model_key == primary_model)
            & (all_predictions.seed == seed)
        ]
        for baseline_key in baseline_models:
            baseline = all_predictions[
                (all_predictions.model_key == baseline_key)
                & (all_predictions.seed == seed)
            ]
            for offset, metric in enumerate(metrics):
                result = paired_group_bootstrap(
                    primary,
                    baseline,
                    metric=metric,
                    reps=reps,
                    seed=base_seed + int(seed) + offset,
                )
                sign = -1.0 if metric in {"brier", "log_loss"} else 1.0
                rows.append(
                    {
                        "seed": int(seed),
                        "primary_model": primary_model,
                        "baseline_model": baseline_key,
                        "metric": metric,
                        "raw_delta_primary_minus_baseline": result.estimate,
                        "benefit_oriented_delta": result.estimate * sign,
                        "ci_low_raw": result.ci_low,
                        "ci_high_raw": result.ci_high,
                        "probability_primary_better": (
                            result.probability_superior
                            if sign > 0
                            else 1 - result.probability_superior
                        ),
                        "valid_replicates": result.valid_replicates,
                    }
                )
    return pd.DataFrame(rows)


def error_review_template(
    primary_predictions: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    frame = primary_predictions[primary_predictions.seed == seed].copy()
    frame["error_type"] = np.select(
        [
            (frame.label == 1) & (frame.pred_calibrated == 0),
            (frame.label == 0) & (frame.pred_calibrated == 1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )
    review = frame[frame.error_type != "correct"].copy()
    for column in [
        "small_lesion",
        "low_contrast",
        "blur",
        "partial_visibility",
        "lighting_variation",
        "background_distraction",
        "possible_label_uncertainty",
        "reviewer_1_comment",
        "reviewer_2_comment",
        "consensus_category",
    ]:
        review[column] = ""
    keep = [
        "image_id",
        "relative_path",
        "group_id",
        "outer_fold",
        "label",
        "prob_calibrated",
        "threshold",
        "pred_calibrated",
        "confidence",
        "error_type",
        "small_lesion",
        "low_contrast",
        "blur",
        "partial_visibility",
        "lighting_variation",
        "background_distraction",
        "possible_label_uncertainty",
        "reviewer_1_comment",
        "reviewer_2_comment",
        "consensus_category",
    ]
    return review[keep].sort_values(
        ["error_type", "confidence"], ascending=[True, False]
    )


def save_summary_figures(
    predictions: pd.DataFrame,
    selective: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    output_dir.mkdir(parents=True, exist_ok=True)
    for model_key in sorted(predictions.model_key.unique()):
        frame = predictions[predictions.model_key == model_key]
        aggregate = frame.groupby(["image_id", "label"], as_index=False).agg(
            prob_calibrated=("prob_calibrated", "mean")
        )
        y = aggregate.label.to_numpy(dtype=int)
        p = aggregate.prob_calibrated.to_numpy(dtype=float)

        fpr, tpr, _ = roc_curve(y, p)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=model_key)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False-positive rate")
        plt.ylabel("Sensitivity")
        plt.title(f"ROC — {model_key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_{model_key}.png", dpi=220)
        plt.close()

        precision, recall, _ = precision_recall_curve(y, p)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=model_key)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–recall — {model_key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"pr_{model_key}.png", dpi=220)
        plt.close()

        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        plt.figure(figsize=(6, 5))
        plt.plot(predicted, observed, marker="o", label=model_key)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed DFU frequency")
        plt.title(f"Calibration — {model_key}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"calibration_{model_key}.png", dpi=220)
        plt.close()

        risk = selective[selective.model_key == model_key]
        if not risk.empty:
            summary = risk.groupby("requested_coverage", as_index=False).agg(
                sensitivity=("sensitivity", "mean"),
                error_rate=("error_rate", "mean"),
            )
            plt.figure(figsize=(6, 5))
            plt.plot(summary.requested_coverage, summary.sensitivity, marker="o", label="Sensitivity")
            plt.plot(summary.requested_coverage, summary.error_rate, marker="s", label="Error rate")
            plt.xlabel("Coverage")
            plt.ylabel("Metric")
            plt.title(f"Selective prediction — {model_key}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"selective_{model_key}.png", dpi=220)
            plt.close()
