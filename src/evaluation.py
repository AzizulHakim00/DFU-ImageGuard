from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image

from .config_data import Config, hash_text, make_inner_partition, seed_worker, sha256_file, write_json
from .models_training import (
    build_loader,
    build_transforms,
    instantiate_model,
    load_checkpoint_model,
    predict_logits,
    train_model,
)


def _finite_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")
    return array


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = _finite_array(logits, "logits")
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    logits = _finite_array(logits, "calibration logits")
    y = np.asarray(y, dtype=int).reshape(-1)
    if len(logits) != len(y):
        raise ValueError("Calibration logits and labels have different lengths")
    if len(np.unique(y)) != 2:
        raise ValueError("Temperature scaling requires both classes")

    def nll(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        z = np.clip(logits / temperature, -40, 40)
        return float(np.mean(np.logaddexp(0, z) - y * z))

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(f"Temperature optimization failed: {result}")
    temperature = float(math.exp(result.x))
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Temperature scaling produced an invalid temperature")
    return temperature


def select_threshold(
    y: np.ndarray,
    p: np.ndarray,
    target_sensitivity: float,
) -> tuple[float, dict[str, Any]]:
    from sklearn.metrics import confusion_matrix

    y = np.asarray(y, dtype=int).reshape(-1)
    p = _finite_array(p, "calibration probabilities")
    if len(y) != len(p):
        raise ValueError("Calibration probabilities and labels have different lengths")
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both classes")
    p = np.clip(p, 0.0, 1.0)

    candidates = np.unique(np.r_[0.0, p, np.nextafter(p, np.inf), 1.0])
    feasible: list[tuple[float, float, float]] = []
    fallback: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        balanced = 0.5 * (sensitivity + specificity)
        fallback.append((balanced, sensitivity, specificity, float(threshold)))
        if sensitivity >= target_sensitivity:
            feasible.append((specificity, sensitivity, float(threshold)))

    if feasible:
        specificity, sensitivity, threshold = max(
            feasible,
            key=lambda item: (item[0], item[2]),
        )
        return threshold, {
            "rule": "max_specificity_subject_to_sensitivity",
            "target_sensitivity": float(target_sensitivity),
            "calibration_sensitivity": float(sensitivity),
            "calibration_specificity": float(specificity),
        }

    balanced, sensitivity, specificity, threshold = max(
        fallback,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    return threshold, {
        "rule": "fallback_max_balanced_accuracy",
        "target_sensitivity": float(target_sensitivity),
        "calibration_balanced_accuracy": float(balanced),
        "calibration_sensitivity": float(sensitivity),
        "calibration_specificity": float(specificity),
    }


def ece_mce(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(_finite_array(p, "probabilities"), 0.0, 1.0)
    if len(y) != len(p):
        raise ValueError("Probabilities and labels have different lengths")
    bins = np.linspace(0, 1, int(n_bins) + 1)
    ece = 0.0
    mce = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if not mask.any():
            continue
        predicted = float(p[mask].mean())
        observed = float(y[mask].mean())
        gap = abs(observed - predicted)
        ece += gap * float(mask.mean())
        mce = max(mce, gap)
    return float(ece), float(mce)


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(_finite_array(p, "probabilities"), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logit_probability = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(logit_probability, y)
        return float(model.coef_[0, 0]), float(model.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def metric_dict(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        cohen_kappa_score,
        confusion_matrix,
        fbeta_score,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )

    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(_finite_array(p, "probabilities"), 1e-7, 1 - 1e-7)
    if len(y) != len(p):
        raise ValueError("Probabilities and labels have different lengths")
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    ece, mce = ece_mce(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    accuracy = accuracy_score(y, pred)
    both_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "error_rate": float(1 - accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_ppv": float(precision),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(fbeta_score(y, pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "log_loss": float(log_loss(y, np.c_[1 - p, p], labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "mce": float(mce),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(_finite_array(p, "probabilities"), 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def create_prediction_frame(
    frame: pd.DataFrame,
    logits: np.ndarray,
    temperature: float,
    threshold: float,
    model_name: str,
    fold: int,
) -> pd.DataFrame:
    logits = _finite_array(logits, "outer-test logits")
    if len(frame) != len(logits):
        raise ValueError("Outer-test dataframe and logits have different lengths")
    out = frame.reset_index(drop=True)[[
        "image_id",
        "image_path",
        "relative_path",
        "group_id",
        "label",
        "label_name",
    ]].copy()
    out["model"] = model_name
    out["outer_fold"] = int(fold)
    out["logit_raw"] = logits
    out["prob_raw"] = sigmoid_np(logits)
    out["temperature"] = float(temperature)
    out["prob_calibrated"] = sigmoid_np(logits / float(temperature))
    out["threshold"] = float(threshold)
    out["pred_raw_0_5"] = (out.prob_raw >= 0.5).astype(int)
    out["pred_calibrated"] = (out.prob_calibrated >= threshold).astype(int)
    out["confidence"] = np.maximum(out.prob_calibrated, 1 - out.prob_calibrated)
    out["predictive_entropy"] = entropy_binary(out.prob_calibrated.values)
    out["top_two_margin"] = np.abs(2 * out.prob_calibrated.values - 1)
    out["correct_calibrated"] = (out.pred_calibrated == out.label).astype(int)
    return out


def _quarantine_bad_file(path: Path) -> None:
    if not path.exists():
        return
    destination = path.with_suffix(path.suffix + ".invalid")
    counter = 1
    while destination.exists():
        destination = path.with_suffix(path.suffix + f".invalid{counter}")
        counter += 1
    os.replace(path, destination)


def run_torch_fold(
    model_name: str,
    factory: Callable[..., Any],
    fold: int,
    manifest: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
    primary: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch

    outer_train = manifest[manifest.outer_fold != fold].copy()
    outer_test = manifest[manifest.outer_fold == fold].copy().reset_index(drop=True)
    if outer_test.empty:
        raise RuntimeError(f"Outer fold {fold + 1} has no test images")
    inner = make_inner_partition(outer_train, cfg, fold)
    inner_path = dirs["manifests"] / f"inner_partition_outer_fold_{fold + 1}.csv"
    if not inner_path.exists():
        inner.to_csv(inner_path, index=False)

    train_df = inner[inner.inner_role == "train"].copy()
    selection_df = inner[inner.inner_role == "selection"].copy()
    calibration_df = inner[inner.inner_role == "calibration"].copy()
    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    checkpoint_name = (
        f"dfu_imageguard_fold_{fold + 1}.pt"
        if primary
        else f"{safe_name}_fold_{fold + 1}.pt"
    )
    checkpoint = dirs["models"] / checkpoint_name
    history_path = dirs["logs"] / f"history_{safe_name}_fold_{fold + 1}.csv"
    trained_now = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if checkpoint.exists() and not cfg.FORCE_RETRAIN:
        try:
            model = load_checkpoint_model(factory, checkpoint, device)
        except Exception as exc:
            print(f"Checkpoint validation failed for {checkpoint.name}; retraining this fold: {exc}")
            _quarantine_bad_file(checkpoint)
    if model is None:
        model, _, trained_now = train_model(
            instantiate_model(factory, pretrained=True),
            train_df,
            selection_df,
            cfg,
            checkpoint,
            history_path,
            cfg.SEED + fold + hash_text(model_name) % 1000,
        )

    _, eval_tf = build_transforms(cfg)
    calibration_loader = build_loader(
        calibration_df,
        eval_tf,
        cfg,
        False,
        cfg.SEED + fold + 300,
    )
    test_loader = build_loader(
        outer_test,
        eval_tf,
        cfg,
        False,
        cfg.SEED + fold + 400,
    )
    calibration_logits, calibration_y, _ = predict_logits(model, calibration_loader, device)
    temperature = fit_temperature(calibration_logits, calibration_y)
    calibration_prob = sigmoid_np(calibration_logits / temperature)
    threshold, threshold_info = select_threshold(
        calibration_y,
        calibration_prob,
        cfg.TARGET_SENSITIVITY,
    )
    test_logits, test_y, _ = predict_logits(model, test_loader, device)
    if not np.array_equal(test_y, outer_test.label.to_numpy(dtype=int)):
        raise AssertionError("Outer-test DataLoader order no longer matches the locked manifest")
    prediction = create_prediction_frame(
        outer_test,
        test_logits,
        temperature,
        threshold,
        model_name,
        fold,
    )

    calibration_frame = pd.DataFrame({
        "image_id": calibration_df.reset_index(drop=True).image_id,
        "label": calibration_y,
        "logit_raw": calibration_logits,
        "prob_calibrated": calibration_prob,
    })
    calibration_frame.to_csv(
        dirs["predictions"] / f"calibration_{safe_name}_fold_{fold + 1}.csv",
        index=False,
    )

    calibration_info = {
        "model": model_name,
        "fold": fold + 1,
        "temperature": temperature,
        "threshold": threshold,
        "threshold_selection": threshold_info,
        "n_train": len(train_df),
        "n_selection": len(selection_df),
        "n_calibration": len(calibration_df),
        "n_outer_test": len(outer_test),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "trained_now": trained_now,
    }
    write_json(
        dirs["configs"] / f"calibration_{safe_name}_fold_{fold + 1}.json",
        calibration_info,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction, calibration_info


def extract_frozen_embeddings(
    frame: pd.DataFrame,
    cfg: Config,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(frame), 512) and np.isfinite(cached).all():
            return cached
        _quarantine_bad_file(cache_path)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    class FrozenEmbeddingDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            self.data = data.reset_index(drop=True)
            self.transform = transforms.Compose([
                transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, index: int):
            with Image.open(self.data.iloc[index].image_path) as image:
                return self.transform(image.convert("RGB")), index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    generator = torch.Generator().manual_seed(cfg.SEED + 700)
    workers = max(0, int(cfg.NUM_WORKERS))
    kwargs: dict[str, Any] = {
        "batch_size": 64,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    loader = DataLoader(FrozenEmbeddingDataset(frame), **kwargs)
    embeddings = np.zeros((len(frame), 512), dtype=np.float32)
    with torch.inference_mode():
        for xb, indices in loader:
            z = model(xb.to(device, non_blocking=device.type == "cuda")).float().cpu().numpy()
            embeddings[indices.numpy()] = z
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("Frozen embeddings contain NaN or infinity")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def run_linear_baseline(
    manifest: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    embeddings = extract_frozen_embeddings(
        manifest,
        cfg,
        dirs["cache"] / "resnet18_frozen_embeddings.npy",
    )
    all_predictions: list[pd.DataFrame] = []
    infos: list[dict[str, Any]] = []
    for fold in range(cfg.N_FOLDS):
        outer_train = manifest[manifest.outer_fold != fold].copy()
        outer_test = manifest[manifest.outer_fold == fold].copy()
        inner = make_inner_partition(outer_train, cfg, fold)
        train_ids = set(inner.loc[inner.inner_role == "train", "image_id"])
        calibration_ids = set(inner.loc[inner.inner_role == "calibration", "image_id"])
        train_idx = manifest.index[manifest.image_id.isin(train_ids)].to_numpy()
        calibration_idx = manifest.index[manifest.image_id.isin(calibration_ids)].to_numpy()
        test_idx = outer_test.index.to_numpy()
        model_path = dirs["models"] / f"linear_logreg_fold_{fold + 1}.joblib"

        pipeline = None
        trained_now = False
        if model_path.exists() and not cfg.FORCE_RETRAIN:
            try:
                pipeline = joblib.load(model_path)
                _ = pipeline.predict_proba(embeddings[train_idx[:1]])
            except Exception as exc:
                print(f"Invalid linear baseline artifact {model_path.name}; retraining: {exc}")
                _quarantine_bad_file(model_path)
                pipeline = None
        if pipeline is None:
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=cfg.SEED + fold,
                )),
            ])
            pipeline.fit(embeddings[train_idx], manifest.loc[train_idx, "label"].values)
            joblib.dump(pipeline, model_path)
            trained_now = True

        calibration_logits = pipeline.decision_function(embeddings[calibration_idx])
        calibration_y = manifest.loc[calibration_idx, "label"].values
        temperature = fit_temperature(calibration_logits, calibration_y)
        calibration_prob = sigmoid_np(calibration_logits / temperature)
        threshold, threshold_info = select_threshold(
            calibration_y,
            calibration_prob,
            cfg.TARGET_SENSITIVITY,
        )
        test_logits = pipeline.decision_function(embeddings[test_idx])
        prediction = create_prediction_frame(
            manifest.loc[test_idx].copy(),
            test_logits,
            temperature,
            threshold,
            cfg.LINEAR_BASELINE_NAME,
            fold,
        )
        info = {
            "model": cfg.LINEAR_BASELINE_NAME,
            "fold": fold + 1,
            "temperature": temperature,
            "threshold": threshold,
            "threshold_selection": threshold_info,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "trained_now": trained_now,
        }
        write_json(
            dirs["configs"] / f"calibration_linear_logreg_fold_{fold + 1}.json",
            info,
        )
        all_predictions.append(prediction)
        infos.append(info)
    return pd.concat(all_predictions, ignore_index=True), infos


def _classification_metrics_from_predictions(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        fbeta_score,
        matthews_corrcoef,
    )

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    accuracy = float((pred == y).mean())
    return {
        "threshold": float("nan"),
        "accuracy": accuracy,
        "error_rate": 1 - accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_ppv": float(tp / max(tp + fp, 1)),
        "recall_sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "f1": float(fbeta_score(y, pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def aggregate_metrics(predictions: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    if predictions.empty:
        raise ValueError("No OOF predictions were produced")
    rows: list[dict[str, Any]] = []
    for model, frame in predictions.groupby("model"):
        if frame.image_id.duplicated().any():
            raise AssertionError(f"Model {model} contains duplicate OOF image IDs")
        raw = metric_dict(frame.label.values, frame.prob_raw.values, 0.5)
        calibrated = metric_dict(frame.label.values, frame.prob_calibrated.values, 0.5)
        calibrated.update(
            _classification_metrics_from_predictions(
                frame.label.to_numpy(dtype=int),
                frame.pred_calibrated.to_numpy(dtype=int),
            )
        )
        fold_thresholds = frame.groupby("outer_fold")["threshold"].first().to_dict()
        for state, values in [("raw", raw), ("calibrated", calibrated)]:
            rows.append({
                "model": model,
                "state": state,
                "fold_thresholds": json.dumps(fold_thresholds),
                **values,
            })
        for fold, fold_frame in frame.groupby("outer_fold"):
            fold_metrics = metric_dict(
                fold_frame.label.values,
                fold_frame.prob_calibrated.values,
                float(fold_frame.threshold.iloc[0]),
            )
            rows.append({
                "model": model,
                "state": f"fold_{fold + 1}_calibrated",
                "fold_thresholds": "",
                **fold_metrics,
            })
    result = pd.DataFrame(rows)
    result.to_csv(dirs["tables"] / "all_metrics_raw_and_calibrated.csv", index=False)
    return result
