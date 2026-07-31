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

def bootstrap_metric_cis(predictions: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.SEED + 900)
    metrics_to_ci = ["accuracy", "balanced_accuracy", "recall_sensitivity", "specificity", "f1", "mcc",
                     "roc_auc", "pr_auc", "brier_score", "log_loss", "ece"]
    rows = []
    for model, frame in predictions.groupby("model"):
        groups = frame.group_id.unique()
        observed = metric_dict(frame.label.values, frame.prob_calibrated.values, 0.5)
        base_pred = frame.pred_calibrated.values
        from sklearn.metrics import balanced_accuracy_score, fbeta_score, matthews_corrcoef, confusion_matrix
        tn, fp, fn, tp = confusion_matrix(frame.label, base_pred, labels=[0, 1]).ravel()
        observed.update({
            "accuracy": float((base_pred == frame.label.values).mean()),
            "balanced_accuracy": balanced_accuracy_score(frame.label, base_pred),
            "recall_sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "f1": fbeta_score(frame.label, base_pred, beta=1, zero_division=0),
            "mcc": matthews_corrcoef(frame.label, base_pred),
        })
        boots = {m: [] for m in metrics_to_ci}
        for _ in range(cfg.BOOTSTRAP_REPS):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            pieces = [frame[frame.group_id == g] for g in sampled]
            b = pd.concat(pieces, ignore_index=True)
            if b.label.nunique() < 2:
                continue
            m = metric_dict(b.label.values, b.prob_calibrated.values, 0.5)
            bp = b.pred_calibrated.values
            tn, fp, fn, tp = confusion_matrix(b.label, bp, labels=[0, 1]).ravel()
            m.update({
                "accuracy": float((bp == b.label.values).mean()),
                "balanced_accuracy": balanced_accuracy_score(b.label, bp),
                "recall_sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                "f1": fbeta_score(b.label, bp, beta=1, zero_division=0),
                "mcc": matthews_corrcoef(b.label, bp),
            })
            for key in metrics_to_ci:
                boots[key].append(m[key])
        for key in metrics_to_ci:
            vals = np.asarray(boots[key], dtype=float)
            rows.append({
                "model": model, "metric": key, "estimate": observed.get(key, np.nan),
                "ci_low": float(np.nanpercentile(vals, 2.5)), "ci_high": float(np.nanpercentile(vals, 97.5)),
                "bootstrap_repetitions_requested": cfg.BOOTSTRAP_REPS,
                "bootstrap_repetitions_valid": int(np.isfinite(vals).sum()), "resampling_unit": "duplicate_group",
            })
    out = pd.DataFrame(rows)
    out.to_csv(dirs["tables"] / "group_bootstrap_95ci.csv", index=False)
    return out


def _holm(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def statistical_comparisons(predictions: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    from scipy.stats import binomtest
    rng = np.random.default_rng(cfg.SEED + 901)
    proposed = predictions[predictions.model == cfg.PRIMARY_MODEL_NAME].copy()
    rows = []
    for baseline in sorted(set(predictions.model) - {cfg.PRIMARY_MODEL_NAME}):
        base = predictions[predictions.model == baseline].copy()
        merged = proposed.merge(base, on=["image_id", "group_id", "label"], suffixes=("_p", "_b"))
        groups = merged.group_id.unique()
        diffs = []
        for _ in range(cfg.BOOTSTRAP_REPS):
            sampled = rng.choice(groups, len(groups), replace=True)
            b = pd.concat([merged[merged.group_id == g] for g in sampled], ignore_index=True)
            acc_p = (b.pred_calibrated_p == b.label).mean()
            acc_b = (b.pred_calibrated_b == b.label).mean()
            diffs.append(acc_p - acc_b)
        p_correct = merged.pred_calibrated_p.values == merged.label.values
        b_correct = merged.pred_calibrated_b.values == merged.label.values
        n10 = int(np.sum(p_correct & ~b_correct))
        n01 = int(np.sum(~p_correct & b_correct))
        discordant = n10 + n01
        mcnemar_p = float(binomtest(min(n10, n01), discordant, 0.5).pvalue) if discordant else 1.0
        rows.append({
            "comparison": f"{cfg.PRIMARY_MODEL_NAME} vs {baseline}",
            "metric": "accuracy", "paired_group_bootstrap_difference": float(np.mean(diffs)),
            "difference_ci_low": float(np.percentile(diffs, 2.5)),
            "difference_ci_high": float(np.percentile(diffs, 97.5)),
            "mcnemar_n_proposed_only_correct": n10, "mcnemar_n_baseline_only_correct": n01,
            "mcnemar_p": mcnemar_p,
        })
    if rows:
        adjusted = _holm([r["mcnemar_p"] for r in rows])
        for r, adj in zip(rows, adjusted):
            r["mcnemar_p_holm"] = adj
            r["interpretation"] = "descriptive paired comparison; no equivalence or deployment claim"
    out = pd.DataFrame(rows)
    out.to_csv(dirs["tables"] / "statistical_comparisons.csv", index=False)
    return out


def savefig_multi(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def make_core_figures(manifest: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame,
                      dirs: dict[str, Path]) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    fig, ax = plt.subplots(figsize=(6, 4))
    manifest.label_name.value_counts().reindex(["Normal", "DFU"]).plot(kind="bar", ax=ax)
    ax.set_title("Class distribution after strict audit")
    ax.set_ylabel("Images")
    savefig_multi(fig, dirs["figures"] / "class_distribution")
    plt.close(fig)

    clusters = pd.read_csv(dirs["tables"] / "duplicate_clusters.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [int((clusters.n_images == 1).sum()), int((clusters.n_images > 1).sum()), int((clusters.n_labels > 1).sum())]
    ax.bar(["Singleton", "Duplicate groups", "Label-conflict groups"], values)
    ax.set_title("Duplicate screening summary")
    ax.tick_params(axis="x", rotation=15)
    savefig_multi(fig, dirs["figures"] / "duplicate_screening")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, frame in predictions.groupby("model"):
        fpr, tpr, _ = roc_curve(frame.label, frame.prob_calibrated)
        ax.plot(fpr, tpr, label=model)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(xlabel="False-positive rate", ylabel="Sensitivity", title="OOF ROC curves")
    ax.legend(fontsize=7)
    savefig_multi(fig, dirs["figures"] / "roc_curves")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, frame in predictions.groupby("model"):
        precision, recall, _ = precision_recall_curve(frame.label, frame.prob_calibrated)
        ax.plot(recall, precision, label=model)
    ax.set(xlabel="Recall", ylabel="Precision", title="OOF precision–recall curves")
    ax.legend(fontsize=7)
    savefig_multi(fig, dirs["figures"] / "precision_recall_curves")
    plt.close(fig)

    for model, frame in predictions.groupby("model"):
        safe = model.lower().replace("-", "_").replace(" ", "_")
        for normalized in [False, True]:
            cm = confusion_matrix(frame.label, frame.pred_calibrated, labels=[0, 1], normalize="true" if normalized else None)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(cm)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:.2f}" if normalized else str(int(cm[i, j])), ha="center", va="center")
            ax.set_xticks([0, 1], ["Normal", "DFU"])
            ax.set_yticks([0, 1], ["Normal", "DFU"])
            ax.set(xlabel="Predicted", ylabel="True", title=f"{model} confusion matrix")
            fig.colorbar(im, ax=ax)
            savefig_multi(fig, dirs["figures"] / f"confusion_{safe}_{'normalized' if normalized else 'raw'}")
            plt.close(fig)

    calibrated = metrics[metrics.state == "calibrated"].copy()
    selected = ["balanced_accuracy", "recall_sensitivity", "specificity", "f1", "mcc", "roc_auc", "pr_auc"]
    plot_df = calibrated.set_index("model")[selected]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Calibrated OOF metric comparison")
    ax.legend(fontsize=7, ncol=2)
    savefig_multi(fig, dirs["figures"] / "metric_comparison")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(calibrated.model, calibrated.error_rate)
    ax.set_ylabel("Error rate")
    ax.set_title("OOF error-rate comparison")
    ax.tick_params(axis="x", rotation=30)
    savefig_multi(fig, dirs["figures"] / "error_rate_comparison")
    plt.close(fig)

    bins = np.linspace(0, 1, 11)
    for state, prob_col in [("raw", "prob_raw"), ("calibrated", "prob_calibrated")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, frame in predictions.groupby("model"):
            xs, ys = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (frame[prob_col] >= lo) & (frame[prob_col] < hi if hi < 1 else frame[prob_col] <= hi)
                if m.any():
                    xs.append(frame.loc[m, prob_col].mean())
                    ys.append(frame.loc[m, "label"].mean())
            ax.plot(xs, ys, marker="o", label=model)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed DFU frequency", title=f"Reliability diagram ({state})")
        ax.legend(fontsize=7)
        savefig_multi(fig, dirs["figures"] / f"reliability_diagram_{state}")
        plt.close(fig)

    history_files = sorted(dirs["logs"].glob("history_*_fold_*.csv"))
    if history_files:
        fig, ax = plt.subplots(figsize=(9, 5))
        for hp in history_files:
            h = pd.read_csv(hp)
            if not h.empty:
                ax.plot(h.epoch, h.selection_auc, alpha=0.45, label=hp.stem.replace("history_", ""))
        ax.set(xlabel="Epoch", ylabel="Selection ROC-AUC", title="Persisted learning curves")
        if len(history_files) <= 12:
            ax.legend(fontsize=6, ncol=2)
        savefig_multi(fig, dirs["figures"] / "learning_curves")
        plt.close(fig)

    for columns, name, title in [
        (["ece", "mce"], "ece_mce_comparison", "Calibration error"),
        (["brier_score", "log_loss"], "brier_logloss_comparison", "Probabilistic loss"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        calibrated.set_index("model")[columns].plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        savefig_multi(fig, dirs["figures"] / name)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, frame in predictions.groupby("model"):
        ax.hist(frame.confidence, bins=20, alpha=0.35, density=True, label=model)
    ax.set(xlabel="Confidence", ylabel="Density", title="Confidence distributions")
    ax.legend(fontsize=7)
    savefig_multi(fig, dirs["figures"] / "confidence_histogram")
    plt.close(fig)

    fold_rows = []
    for (model, fold), f in predictions.groupby(["model", "outer_fold"]):
        fold_rows.append({"model": model, "fold": fold + 1,
                          "balanced_accuracy": metric_dict(f.label, f.prob_calibrated, float(f.threshold.iloc[0]))["balanced_accuracy"]})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(dirs["tables"] / "fold_stability.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, f in fold_df.groupby("model"):
        ax.plot(f.fold, f.balanced_accuracy, marker="o", label=model)
    ax.set(xlabel="Outer fold", ylabel="Balanced accuracy", title="Five-fold stability (not multi-seed stability)")
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=7)
    savefig_multi(fig, dirs["figures"] / "fold_stability")
    plt.close(fig)

    risk_rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, frame in predictions.groupby("model"):
        order = np.argsort(-frame.confidence.values)
        correct = frame.correct_calibrated.values[order]
        coverages = np.linspace(0.1, 1.0, 19)
        risks = []
        for cov in coverages:
            n = max(1, int(len(frame) * cov))
            risk = 1 - correct[:n].mean()
            risks.append(risk)
            risk_rows.append({"model": model, "coverage": cov, "risk": risk, "selective_accuracy": 1 - risk})
        ax.plot(coverages, risks, label=model)
    pd.DataFrame(risk_rows).to_csv(dirs["tables"] / "risk_coverage.csv", index=False)
    ax.set(xlabel="Coverage", ylabel="Risk (error rate)", title="Risk–coverage curves")
    ax.legend(fontsize=7)
    savefig_multi(fig, dirs["figures"] / "risk_coverage")
    plt.close(fig)
