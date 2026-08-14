from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from .config_data import Config, seed_everything, sha256_file
from .evaluation import (
    calibration_slope_intercept,
    ece_mce,
    fit_temperature,
    select_threshold,
    sigmoid_np,
)
from .models_training import FrameDataset, build_loader, predict_logits
from .reliability_io import (
    ReliabilitySettings,
    _sync_trial_verified,
    atomic_csv,
    atomic_json,
    atomic_torch,
)
from .reliability_metrics import metrics_from_decisions
from .reliability_models import MODEL_SPECS, create_classifier, parameter_summary

def _device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU runtime is required for the final experiment")
    return torch.device("cuda")


def _optimizer(model, settings: ReliabilitySettings):
    import torch

    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": settings.backbone_lr, "name": "backbone"},
            {
                "params": list(model.head.parameters()) + list(model.dropout.parameters()),
                "lr": settings.head_lr,
                "name": "head",
            },
        ],
        weight_decay=settings.weight_decay,
    )


def _build_reliability_transforms(cfg: Config):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                cfg.IMAGE_SIZE, scale=(0.90, 1.00), ratio=(0.95, 1.05)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(
                brightness=0.10, contrast=0.10, saturation=0.05, hue=0.01
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, eval_transform


def _selection_score(y: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    probability = sigmoid_np(logits)
    auc = float(roc_auc_score(y, probability))
    nll = float(log_loss(y, np.c_[1 - probability, probability], labels=[0, 1]))
    return auc, nll


def _is_better(
    auc: float,
    nll: float,
    best_auc: float,
    best_nll: float,
) -> bool:
    return auc > best_auc + 1e-5 or (
        abs(auc - best_auc) <= 1e-5 and nll < best_nll - 1e-5
    )


def _augment_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    decision: np.ndarray,
) -> dict[str, Any]:
    result = metrics_from_decisions(y, probability, decision)
    ece, mce = ece_mce(y, probability, 15)
    slope, intercept = calibration_slope_intercept(y, probability)
    result.update(
        {
            "ece": float(ece),
            "mce": float(mce),
            "calibration_slope": float(slope),
            "calibration_intercept": float(intercept),
        }
    )
    return result


def _trial_paths(
    run_root: Path,
    model_key: str,
    seed: int,
    fold: int,
) -> dict[str, Path]:
    root = run_root / "trials" / model_key / f"seed_{seed}" / f"fold_{fold + 1}"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "last": root / "last_resume.pt",
        "best": root / "best_model.pt",
        "portable": root / "best_model_portable_fp16.pt",
        "history": root / "history.csv",
        "predictions": root / "test_predictions.csv",
        "calibration": root / "calibration_predictions.csv",
        "metrics": root / "metrics.json",
        "complete": root / "COMPLETE.json",
    }


def _train_and_evaluate_trial(
    *,
    settings: ReliabilitySettings,
    cfg: Config,
    run_root: Path,
    model_key: str,
    seed: int,
    fold: int,
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    import torch
    import torch.nn.functional as F

    paths = _trial_paths(run_root, model_key, seed, fold)
    if paths["complete"].exists() and paths["predictions"].exists() and not settings.force_retrain:
        metrics = json.loads(paths["complete"].read_text())
        predictions = pd.read_csv(paths["predictions"])
        return metrics, predictions

    device = _device()
    seed_everything(seed + fold * 1000)
    model = create_classifier(
        model_key,
        pretrained=True,
        dropout=settings.dropout,
    ).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    optimizer = _optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(settings.max_epochs, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    train_tf, eval_tf = _build_reliability_transforms(cfg)
    train_loader = build_loader(train_df, train_tf, cfg, True, seed + fold * 1000)
    selection_loader = build_loader(
        selection_df, eval_tf, cfg, False, seed + fold * 1000 + 1
    )

    positive = float((train_df.label == 1).sum())
    negative = float((train_df.label == 0).sum())
    pos_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    start_epoch = 1
    best_auc = -np.inf
    best_nll = np.inf
    best_epoch = -1
    patience_left = settings.patience
    history: list[dict[str, Any]] = []

    if paths["last"].exists() and not settings.force_retrain:
        payload = torch.load(paths["last"], map_location=device, weights_only=False)
        if (
            payload.get("model_key") == model_key
            and int(payload.get("seed")) == int(seed)
            and int(payload.get("fold")) == int(fold)
        ):
            model.load_state_dict(payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            scaler.load_state_dict(payload["scaler_state_dict"])
            start_epoch = int(payload["epoch"]) + 1
            best_auc = float(payload["best_auc"])
            best_nll = float(payload["best_nll"])
            best_epoch = int(payload["best_epoch"])
            patience_left = int(payload["patience_left"])
            if paths["history"].exists():
                history = pd.read_csv(paths["history"]).to_dict("records")
            print(
                f"RESUME {model_key} seed={seed} fold={fold + 1} at epoch {start_epoch}"
            )

    if start_epoch > settings.freeze_epochs:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True

    for epoch in range(start_epoch, settings.max_epochs + 1):
        if epoch == settings.freeze_epochs + 1:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True

        model.train()
        total_loss = 0.0
        examples = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(xb).reshape(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits, yb, pos_weight=pos_weight
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for {model_key}, seed={seed}, fold={fold + 1}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(yb)
            examples += len(yb)
        scheduler.step()

        selection_logits, selection_y, _ = predict_logits(
            model, selection_loader, device
        )
        selection_auc, selection_nll = _selection_score(
            selection_y, selection_logits
        )
        row = {
            "epoch": int(epoch),
            "train_loss": total_loss / max(examples, 1),
            "selection_auc": selection_auc,
            "selection_log_loss": selection_nll,
            "backbone_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[1]["lr"]),
        }
        history.append(row)
        atomic_csv(pd.DataFrame(history), paths["history"])
        print(pd.DataFrame([row]).round(6).to_string(index=False))

        improved = _is_better(
            selection_auc,
            selection_nll,
            best_auc,
            best_nll,
        )
        if improved:
            best_auc = selection_auc
            best_nll = selection_nll
            best_epoch = epoch
            patience_left = settings.patience
            atomic_torch(
                {
                    "model_key": model_key,
                    "resolved_name": model.resolved_name,
                    "seed": int(seed),
                    "fold": int(fold),
                    "epoch": int(epoch),
                    "best_selection_auc": best_auc,
                    "best_selection_log_loss": best_nll,
                    "model_state_dict": model.state_dict(),
                    "parameter_summary": parameter_summary(model),
                },
                paths["best"],
            )
        else:
            patience_left -= 1

        atomic_torch(
            {
                "model_key": model_key,
                "resolved_name": model.resolved_name,
                "seed": int(seed),
                "fold": int(fold),
                "epoch": int(epoch),
                "best_auc": float(best_auc),
                "best_nll": float(best_nll),
                "best_epoch": int(best_epoch),
                "patience_left": int(patience_left),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "settings": asdict(settings),
            },
            paths["last"],
        )
        if settings.require_secondary_backup:
            _sync_trial_verified(paths["root"], run_root, settings)
        if patience_left <= 0:
            break

    if not paths["best"].exists():
        raise RuntimeError(f"No best checkpoint for {model_key}, seed={seed}, fold={fold + 1}")
    best_payload = torch.load(paths["best"], map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    model.eval()

    portable = {
        key: (
            value.detach().cpu().half()
            if torch.is_floating_point(value)
            else value.detach().cpu()
        )
        for key, value in model.state_dict().items()
    }
    atomic_torch(
        {
            "model_key": model_key,
            "resolved_name": model.resolved_name,
            "seed": int(seed),
            "fold": int(fold),
            "state_dict": portable,
            "parameter_summary": parameter_summary(model),
        },
        paths["portable"],
    )

    calibration_loader = build_loader(
        calibration_df, eval_tf, cfg, False, seed + fold * 1000 + 300
    )
    test_loader = build_loader(
        test_df, eval_tf, cfg, False, seed + fold * 1000 + 400
    )
    calibration_logits, calibration_y, _ = predict_logits(
        model, calibration_loader, device
    )
    temperature = fit_temperature(calibration_logits, calibration_y)
    calibration_probability = sigmoid_np(calibration_logits / temperature)
    threshold, threshold_rule = select_threshold(
        calibration_y,
        calibration_probability,
        settings.target_sensitivity,
    )
    test_logits, test_y, _ = predict_logits(model, test_loader, device)
    expected_y = test_df.reset_index(drop=True).label.to_numpy(dtype=int)
    if not np.array_equal(test_y, expected_y):
        raise AssertionError("Outer-test order changed")
    probability_raw = sigmoid_np(test_logits)
    probability_calibrated = sigmoid_np(test_logits / temperature)
    decision = (probability_calibrated >= threshold).astype(int)

    calibration_frame = calibration_df.reset_index(drop=True)[
        ["image_id", "group_id", "label", "relative_path"]
    ].copy()
    calibration_frame["logit_raw"] = calibration_logits
    calibration_frame["temperature"] = temperature
    calibration_frame["prob_calibrated"] = calibration_probability
    calibration_frame["threshold"] = threshold
    atomic_csv(calibration_frame, paths["calibration"])

    prediction = test_df.reset_index(drop=True)[
        [
            "image_id",
            "image_path",
            "relative_path",
            "group_id",
            "label",
            "label_name",
            "outer_fold",
        ]
    ].copy()
    prediction["outer_fold"] = int(fold + 1)
    prediction["model_key"] = model_key
    prediction["model_name"] = MODEL_SPECS[model_key].display_name
    prediction["resolved_model_name"] = model.resolved_name
    prediction["seed"] = int(seed)
    prediction["logit_raw"] = test_logits
    prediction["prob_raw"] = probability_raw
    prediction["temperature"] = temperature
    prediction["prob_calibrated"] = probability_calibrated
    prediction["threshold"] = threshold
    prediction["pred_raw_0_5"] = (probability_raw >= 0.5).astype(int)
    prediction["pred_calibrated"] = decision
    prediction["confidence"] = np.maximum(
        probability_calibrated, 1 - probability_calibrated
    )
    prediction["predictive_entropy"] = -(
        probability_calibrated * np.log(np.clip(probability_calibrated, 1e-8, 1))
        + (1 - probability_calibrated)
        * np.log(np.clip(1 - probability_calibrated, 1e-8, 1))
    )
    prediction["correct_calibrated"] = (
        prediction.pred_calibrated == prediction.label
    ).astype(int)
    atomic_csv(prediction, paths["predictions"])

    raw_metrics = _augment_metrics(
        test_y, probability_raw, (probability_raw >= 0.5).astype(int)
    )
    calibrated_metrics = _augment_metrics(
        test_y, probability_calibrated, decision
    )
    metrics = {
        **calibrated_metrics,
        "model_key": model_key,
        "model_name": MODEL_SPECS[model_key].display_name,
        "resolved_model_name": model.resolved_name,
        "seed": int(seed),
        "outer_fold": int(fold + 1),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "threshold_rule": threshold_rule,
        "best_epoch": int(best_payload["epoch"]),
        "best_selection_auc": float(best_payload["best_selection_auc"]),
        "best_selection_log_loss": float(best_payload["best_selection_log_loss"]),
        "n_train": int(len(train_df)),
        "n_selection": int(len(selection_df)),
        "n_calibration": int(len(calibration_df)),
        "n_outer_test": int(len(test_df)),
        "raw_brier": raw_metrics["brier"],
        "raw_ece": raw_metrics["ece"],
        "raw_log_loss": raw_metrics["log_loss"],
        "best_model_sha256": sha256_file(paths["best"]),
        "last_resume_sha256": sha256_file(paths["last"]),
        "portable_sha256": sha256_file(paths["portable"]),
        "parameter_summary": best_payload["parameter_summary"],
    }
    atomic_json(paths["metrics"], metrics)
    atomic_json(paths["complete"], metrics)
    if settings.require_secondary_backup:
        _sync_trial_verified(paths["root"], run_root, settings)

    del model, train_loader, selection_loader, calibration_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, prediction
