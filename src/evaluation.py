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
from .models_training import *

def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    def nll(log_t):
        t = math.exp(float(log_t))
        z = np.clip(logits / t, -40, 40)
        return float(np.mean(np.logaddexp(0, z) - y * z))
    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(result.x))


def select_threshold(y: np.ndarray, p: np.ndarray, target_sensitivity: float) -> tuple[float, dict[str, Any]]:
    from sklearn.metrics import confusion_matrix
    candidates = np.unique(np.r_[0.0, p, 1.0])
    feasible: list[tuple[float, float, float]] = []
    fallback: list[tuple[float, float]] = []
    for t in candidates:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        bal = 0.5 * (sens + spec)
        fallback.append((bal, float(t)))
        if sens >= target_sensitivity:
            feasible.append((spec, sens, float(t)))
    if feasible:
        spec, sens, t = sorted(feasible, key=lambda x: (x[0], x[2]), reverse=True)[0]
        return t, {"rule": "max_specificity_subject_to_sensitivity", "target_sensitivity": target_sensitivity,
                   "calibration_sensitivity": sens, "calibration_specificity": spec}
    bal, t = max(fallback)
    return t, {"rule": "fallback_max_balanced_accuracy", "target_sensitivity": target_sensitivity,
               "calibration_balanced_accuracy": bal}


def ece_mce(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        gap = abs(acc - conf)
        ece += gap * mask.mean()
        mce = max(mce, gap)
    return float(ece), float(mce)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(lp, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def metric_dict(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
        cohen_kappa_score, confusion_matrix, fbeta_score, log_loss, matthews_corrcoef,
        roc_auc_score,
    )
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p).astype(float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    ece, mce = ece_mce(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "threshold": float(threshold), "accuracy": accuracy_score(y, pred),
        "error_rate": 1 - accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_ppv": ppv, "recall_sensitivity": sensitivity, "specificity": specificity,
        "npv": npv, "f1": fbeta_score(y, pred, beta=1, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y, pred), "cohen_kappa": cohen_kappa_score(y, pred),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "log_loss": log_loss(y, np.c_[1 - p, p], labels=[0, 1]),
        "brier_score": brier_score_loss(y, p), "ece": ece, "mce": mce,
        "calibration_slope": slope, "calibration_intercept": intercept,
        "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p), 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def create_prediction_frame(frame: pd.DataFrame, logits: np.ndarray, temperature: float,
                            threshold: float, model_name: str, fold: int) -> pd.DataFrame:
    out = frame.reset_index(drop=True)[["image_id", "image_path", "relative_path", "group_id", "label", "label_name"]].copy()
    out["model"] = model_name
    out["outer_fold"] = fold
    out["logit_raw"] = logits
    out["prob_raw"] = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    out["temperature"] = temperature
    out["prob_calibrated"] = 1 / (1 + np.exp(-np.clip(logits / temperature, -40, 40)))
    out["threshold"] = threshold
    out["pred_raw_0_5"] = (out.prob_raw >= 0.5).astype(int)
    out["pred_calibrated"] = (out.prob_calibrated >= threshold).astype(int)
    out["confidence"] = np.maximum(out.prob_calibrated, 1 - out.prob_calibrated)
    out["predictive_entropy"] = entropy_binary(out.prob_calibrated.values)
    out["top_two_margin"] = np.abs(2 * out.prob_calibrated.values - 1)
    out["correct_calibrated"] = (out.pred_calibrated == out.label).astype(int)
    return out


def run_torch_fold(model_name: str, factory: Callable[[], Any], fold: int, manifest: pd.DataFrame,
                   cfg: Config, dirs: dict[str, Path], primary: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch

    outer_train = manifest[manifest.outer_fold != fold].copy()
    outer_test = manifest[manifest.outer_fold == fold].copy().reset_index(drop=True)
    inner = make_inner_partition(outer_train, cfg, fold)
    train_df = inner[inner.inner_role == "train"].copy()
    sel_df = inner[inner.inner_role == "selection"].copy()
    cal_df = inner[inner.inner_role == "calibration"].copy()
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    ckpt_name = f"dfu_imageguard_fold_{fold + 1}.pt" if primary else f"{safe_name}_fold_{fold + 1}.pt"
    checkpoint = dirs["models"] / ckpt_name
    history_path = dirs["logs"] / f"history_{safe_name}_fold_{fold + 1}.csv"
    trained_now = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint.exists() and not cfg.FORCE_RETRAIN:
        model = load_checkpoint_model(factory, checkpoint, device)
    else:
        model, _, trained_now = train_model(factory(), train_df, sel_df, cfg, checkpoint, history_path,
                                            cfg.SEED + fold + hash_text(model_name) % 1000)
    _, eval_tf = build_transforms(cfg)
    cal_loader = build_loader(cal_df, eval_tf, cfg, False, cfg.SEED + fold + 300)
    test_loader = build_loader(outer_test, eval_tf, cfg, False, cfg.SEED + fold + 400)
    cal_logits, cal_y, _ = predict_logits(model, cal_loader, device)
    temperature = fit_temperature(cal_logits, cal_y)
    cal_prob = 1 / (1 + np.exp(-np.clip(cal_logits / temperature, -40, 40)))
    threshold, threshold_info = select_threshold(cal_y, cal_prob, cfg.TARGET_SENSITIVITY)
    test_logits, _, _ = predict_logits(model, test_loader, device)
    pred = create_prediction_frame(outer_test, test_logits, temperature, threshold, model_name, fold)
    cal_info = {
        "model": model_name, "fold": fold + 1, "temperature": temperature,
        "threshold": threshold, "threshold_selection": threshold_info,
        "n_train": len(train_df), "n_selection": len(sel_df), "n_calibration": len(cal_df),
        "n_outer_test": len(outer_test), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "trained_now": trained_now,
    }
    write_json(dirs["configs"] / f"calibration_{safe_name}_fold_{fold + 1}.json", cal_info)
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pred, cal_info


def extract_frozen_embeddings(frame: pd.DataFrame, cfg: Config, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    class D(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
            self.tf = transforms.Compose([
                transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            with Image.open(self.df.iloc[i].image_path) as im:
                return self.tf(im.convert("RGB")), i

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(D(frame), batch_size=64, shuffle=False, num_workers=cfg.NUM_WORKERS)
    emb = np.zeros((len(frame), 512), dtype=np.float32)
    with torch.inference_mode():
        for xb, idx in loader:
            z = model(xb.to(device)).cpu().numpy()
            emb[idx.numpy()] = z
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return emb


def run_linear_baseline(manifest: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    emb = extract_frozen_embeddings(manifest, cfg, dirs["cache"] / "resnet18_frozen_embeddings.npy")
    all_preds, infos = [], []
    for fold in range(cfg.N_FOLDS):
        outer_train = manifest[manifest.outer_fold != fold].copy()
        outer_test = manifest[manifest.outer_fold == fold].copy()
        inner = make_inner_partition(outer_train, cfg, fold)
        train_ids = set(inner.loc[inner.inner_role == "train", "image_id"])
        cal_ids = set(inner.loc[inner.inner_role == "calibration", "image_id"])
        tr_idx = manifest.index[manifest.image_id.isin(train_ids)].to_numpy()
        cal_idx = manifest.index[manifest.image_id.isin(cal_ids)].to_numpy()
        te_idx = outer_test.index.to_numpy()
        model_path = dirs["models"] / f"linear_logreg_fold_{fold + 1}.joblib"
        if model_path.exists() and not cfg.FORCE_RETRAIN:
            pipe = joblib.load(model_path)
            trained_now = False
        else:
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                          random_state=cfg.SEED + fold)),
            ])
            pipe.fit(emb[tr_idx], manifest.loc[tr_idx, "label"].values)
            joblib.dump(pipe, model_path)
            trained_now = True
        cal_logits = pipe.decision_function(emb[cal_idx])
        cal_y = manifest.loc[cal_idx, "label"].values
        temperature = fit_temperature(cal_logits, cal_y)
        cal_prob = 1 / (1 + np.exp(-np.clip(cal_logits / temperature, -40, 40)))
        threshold, threshold_info = select_threshold(cal_y, cal_prob, cfg.TARGET_SENSITIVITY)
        test_logits = pipe.decision_function(emb[te_idx])
        pred = create_prediction_frame(manifest.loc[te_idx].copy(), test_logits, temperature, threshold,
                                       cfg.LINEAR_BASELINE_NAME, fold)
        info = {
            "model": cfg.LINEAR_BASELINE_NAME, "fold": fold + 1, "temperature": temperature,
            "threshold": threshold, "threshold_selection": threshold_info,
            "model_path": str(model_path), "model_sha256": sha256_file(model_path),
            "trained_now": trained_now,
        }
        write_json(dirs["configs"] / f"calibration_linear_logreg_fold_{fold + 1}.json", info)
        all_preds.append(pred)
        infos.append(info)
    return pd.concat(all_preds, ignore_index=True), infos


def aggregate_metrics(predictions: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for model, frame in predictions.groupby("model"):
        raw = metric_dict(frame.label.values, frame.prob_raw.values, 0.5)
        cal_thresholds = frame.groupby("outer_fold")["threshold"].first().to_dict()
        cal = metric_dict(frame.label.values, frame.prob_calibrated.values, 0.5)
        pred = frame.pred_calibrated.values
        from sklearn.metrics import confusion_matrix, balanced_accuracy_score, fbeta_score, matthews_corrcoef, cohen_kappa_score
        tn, fp, fn, tp = confusion_matrix(frame.label, pred, labels=[0, 1]).ravel()
        cal.update({
            "threshold": np.nan, "accuracy": float((pred == frame.label.values).mean()),
            "error_rate": float((pred != frame.label.values).mean()),
            "balanced_accuracy": balanced_accuracy_score(frame.label, pred),
            "precision_ppv": tp / max(tp + fp, 1), "recall_sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1), "npv": tn / max(tn + fn, 1),
            "f1": fbeta_score(frame.label, pred, beta=1, zero_division=0),
            "f2": fbeta_score(frame.label, pred, beta=2, zero_division=0),
            "mcc": matthews_corrcoef(frame.label, pred), "cohen_kappa": cohen_kappa_score(frame.label, pred),
            "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
        for state, values in [("raw", raw), ("calibrated", cal)]:
            rows.append({"model": model, "state": state, "fold_thresholds": json.dumps(cal_thresholds), **values})
        for fold, ff in frame.groupby("outer_fold"):
            mm = metric_dict(ff.label.values, ff.prob_calibrated.values, float(ff.threshold.iloc[0]))
            rows.append({"model": model, "state": f"fold_{fold + 1}_calibrated", "fold_thresholds": "", **mm})
    result = pd.DataFrame(rows)
    result.to_csv(dirs["tables"] / "all_metrics_raw_and_calibrated.csv", index=False)
    return result
