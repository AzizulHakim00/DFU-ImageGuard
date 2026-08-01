from __future__ import annotations

import gc
import json
import math
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .artifacts import artifact_manifest, push_to_github, software_hardware_versions
from .config_data import (
    Config,
    assign_duplicate_groups,
    build_manifest,
    download_dataset,
    make_inner_partition,
    make_outer_folds,
    mount_drive,
    now_run_id,
    prepare_run_dirs,
    seed_everything,
    sha256_file,
    write_json,
)
from .evaluation import create_prediction_frame, fit_temperature, metric_dict, select_threshold
from .models_training import build_loader, build_transforms
from .polarmorph_model import build_polarmorph_model
from .q1_posthoc import apply_q1_posthoc_corrections
from .statistics_figures import bootstrap_metric_cis, make_core_figures, statistical_comparisons


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_payload(model, cfg: Config, fold: int, epoch: int, score: float) -> dict[str, Any]:
    return {
        "architecture": "DFU-PolarMorphNet",
        "architecture_version": 1,
        "model_state_dict": model.state_dict(),
        "outer_fold": fold + 1,
        "best_epoch": epoch,
        "best_selection_auc": score,
        "config": asdict(cfg),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_model(path: Path, cfg: Config, device):
    import torch

    model = build_polarmorph_model(cfg).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("architecture") != "DFU-PolarMorphNet":
        raise RuntimeError(f"Not a DFU-PolarMorphNet checkpoint: {path}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _predict(model, loader, device, save_aux: bool = False):
    import torch

    logits, labels, indices, aux_rows = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for xb, yb, idx in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            output = model(xb, return_aux=save_aux)
            batch_logits = output["logits"] if save_aux else output
            logits.append(batch_logits.detach().float().cpu().numpy())
            labels.append(yb.numpy())
            indices.append(idx.numpy())
            if save_aux:
                for position in range(len(yb)):
                    aux_rows.append(
                        {
                            "loader_index": int(idx[position]),
                            "center_x": float(output["center"][position, 0].cpu()),
                            "center_y": float(output["center"][position, 1].cpu()),
                            "lesion_scale": float(output["scale"][position].cpu()),
                            "fusion_radial": float(output["fusion_weights"][position, 0].cpu()),
                            "fusion_contour": float(output["fusion_weights"][position, 1].cpu()),
                            "fusion_global": float(output["fusion_weights"][position, 2].cpu()),
                        }
                    )
    return (
        np.concatenate(logits).reshape(-1),
        np.concatenate(labels).astype(int),
        np.concatenate(indices).astype(int),
        pd.DataFrame(aux_rows),
    )


def _auc(y: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return float(roc_auc_score(y, probability))


def _train_fold(model, train_df, selection_df, cfg: Config, fold: int, checkpoint: Path, history_path: Path):
    import torch
    import torch.nn.functional as F

    device = _device()
    seed = int(cfg.SEED + fold * 1009)
    seed_everything(seed)
    model = model.to(device)
    train_tf, eval_tf = build_transforms(cfg)
    train_loader = build_loader(train_df, train_tf, cfg, True, seed)
    selection_loader = build_loader(selection_df, eval_tf, cfg, False, seed + 1)
    positives = float((train_df.label == 1).sum())
    negatives = float((train_df.label == 0).sum())
    pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.LEARNING_RATE), weight_decay=float(cfg.WEIGHT_DECAY))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(cfg.MAX_EPOCHS), 1))
    amp_enabled = bool(cfg.USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_auc, patience = -np.inf, int(cfg.PATIENCE)
    history = []

    for epoch in range(1, int(cfg.MAX_EPOCHS) + 1):
        model.train()
        sums = {"total": 0.0, "classification": 0.0, "background": 0.0, "mask": 0.0}
        seen = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(xb, return_aux=True)
                classification = F.binary_cross_entropy_with_logits(output["logits"], yb, pos_weight=pos_weight)
                background = F.binary_cross_entropy_with_logits(output["background_logits"], yb)
                mask = output["mask"]
                tv = (mask[:, :, 1:] - mask[:, :, :-1]).abs().mean() + (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
                area = (mask.mean(dim=(1, 2, 3)) - 0.25).abs().mean()
                mask_regularization = tv + 0.25 * area
                total = classification + 0.10 * background + 0.05 * mask_regularization
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss in fold {fold + 1}, epoch {epoch}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.GRAD_CLIP_NORM))
            scaler.step(optimizer)
            scaler.update()
            n = len(yb)
            sums["total"] += float(total.detach().cpu()) * n
            sums["classification"] += float(classification.detach().cpu()) * n
            sums["background"] += float(background.detach().cpu()) * n
            sums["mask"] += float(mask_regularization.detach().cpu()) * n
            seen += n
        scheduler.step()
        val_logits, val_y, _, _ = _predict(model, selection_loader, device)
        selection_auc = _auc(val_y, val_logits)
        row = {
            "fold": fold + 1,
            "epoch": epoch,
            "train_total_loss": sums["total"] / max(seen, 1),
            "train_classification_loss": sums["classification"] / max(seen, 1),
            "train_background_loss": sums["background"] / max(seen, 1),
            "train_mask_regularization": sums["mask"] / max(seen, 1),
            "selection_auc": selection_auc,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(pd.DataFrame([row]).round(5).to_string(index=False))
        if selection_auc > best_auc + 1e-5:
            best_auc, patience = selection_auc, int(cfg.PATIENCE)
            _atomic_torch_save(_checkpoint_payload(model, cfg, fold, epoch, selection_auc), checkpoint)
        else:
            patience -= 1
            if patience <= 0:
                break
    model, payload = _load_model(checkpoint, cfg, device)
    return model, pd.DataFrame(history), payload


def _plot_fold_live(history: pd.DataFrame, fold_metrics: pd.DataFrame, dirs: dict[str, Path]) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import display

    display(fold_metrics.round(5))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.epoch, history.train_total_loss, label="Train loss")
    ax2 = ax.twinx()
    ax2.plot(history.epoch, history.selection_auc, label="Selection AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("AUC")
    ax.set_title(f"DFU-PolarMorphNet fold {int(history.fold.iloc[0])}")
    path = dirs["figures"] / f"live_fold_{int(history.fold.iloc[0])}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _small_xai(model, outer_test: pd.DataFrame, cfg: Config, fold: int, dirs: dict[str, Path]) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    _, eval_tf = build_transforms(cfg)
    device = _device()
    sample = outer_test.sample(n=min(4, len(outer_test)), random_state=int(cfg.SEED + fold)).reset_index(drop=True)
    records = []
    for _, row in sample.iterrows():
        with Image.open(row.image_path) as image:
            original = image.convert("RGB").resize((int(cfg.IMAGE_SIZE), int(cfg.IMAGE_SIZE)))
        tensor = eval_tf(original).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(tensor, return_aux=True)
            probability = float(torch.sigmoid(output["logits"])[0].cpu())
            mask = output["mask"][0, 0].cpu().numpy()
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.imshow(original)
        axis.imshow(mask, alpha=0.45, extent=(0, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 0))
        axis.axis("off")
        axis.set_title(f"Fold {fold + 1} | true={row.label_name} | p={probability:.3f}")
        path = dirs["xai"] / f"polarmorph_fold_{fold + 1}_{row.image_id}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        records.append({"image_id": row.image_id, "fold": fold + 1, "true_label": int(row.label), "probability": probability, "method": "weak_lesion_map", "path": str(path), "causality_claim": False})
    return pd.DataFrame(records)


def run_polarmorph_complete(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = Config()
    for key, value in (overrides or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.PRIMARY_MODEL_NAME = "DFU-PolarMorphNet"
    cfg.RUN_ID = cfg.RUN_ID or f"POLAR_{now_run_id()}"
    mount_drive(cfg)
    seed_everything(cfg.SEED)
    dirs = prepare_run_dirs(cfg)
    started = time.time()
    dataset_root = download_dataset(cfg, dirs)
    manifest = make_outer_folds(assign_duplicate_groups(build_manifest(dataset_root, cfg, dirs), cfg, dirs), cfg, dirs)
    predictions, calibration_rows, xai_rows = [], [], []

    for fold in range(int(cfg.N_FOLDS)):
        outer_train = manifest[manifest.outer_fold != fold].copy()
        outer_test = manifest[manifest.outer_fold == fold].copy().reset_index(drop=True)
        inner = make_inner_partition(outer_train, cfg, fold)
        train_df = inner[inner.inner_role == "train"].copy()
        selection_df = inner[inner.inner_role == "selection"].copy()
        calibration_df = inner[inner.inner_role == "calibration"].copy()
        checkpoint = dirs["models"] / f"dfu_polarmorphnet_fold_{fold + 1}.pt"
        history_path = dirs["logs"] / f"polarmorph_history_fold_{fold + 1}.csv"
        trained_now = False
        if checkpoint.exists() and not bool(cfg.FORCE_RETRAIN):
            model, payload = _load_model(checkpoint, cfg, _device())
            history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
            print(f"Fold {fold + 1}: reused checkpoint {checkpoint.name}")
        else:
            model, history, payload = _train_fold(build_polarmorph_model(cfg), train_df, selection_df, cfg, fold, checkpoint, history_path)
            trained_now = True
        _, eval_tf = build_transforms(cfg)
        calibration_loader = build_loader(calibration_df, eval_tf, cfg, False, int(cfg.SEED + fold + 300))
        test_loader = build_loader(outer_test, eval_tf, cfg, False, int(cfg.SEED + fold + 400))
        cal_logits, cal_y, _, _ = _predict(model, calibration_loader, _device())
        temperature = fit_temperature(cal_logits, cal_y)
        cal_probability = 1.0 / (1.0 + np.exp(-np.clip(cal_logits / temperature, -30, 30)))
        threshold, threshold_info = select_threshold(cal_y, cal_probability, cfg.TARGET_SENSITIVITY)
        test_logits, test_y, _, aux = _predict(model, test_loader, _device(), save_aux=True)
        frame = create_prediction_frame(outer_test, test_logits, temperature, threshold, cfg.PRIMARY_MODEL_NAME, fold)
        frame = frame.merge(aux, left_index=True, right_on="loader_index", how="left").drop(columns=["loader_index"])
        predictions.append(frame)
        frame.to_csv(dirs["predictions"] / f"polarmorph_oof_fold_{fold + 1}.csv", index=False)
        combined = pd.concat(predictions, ignore_index=True)
        combined.to_csv(dirs["predictions"] / "polarmorph_oof_progress.csv", index=False)
        fold_metric = pd.DataFrame([{"fold": fold + 1, **metric_dict(test_y, frame.prob_calibrated.to_numpy(), threshold)}])
        fold_metric.to_csv(dirs["tables"] / f"polarmorph_metrics_fold_{fold + 1}.csv", index=False)
        if not history.empty:
            _plot_fold_live(history, pd.concat([pd.read_csv(p) for p in sorted(dirs["tables"].glob("polarmorph_metrics_fold_*.csv"))], ignore_index=True), dirs)
        xai_rows.append(_small_xai(model, outer_test, cfg, fold, dirs))
        calibration_rows.append({"fold": fold + 1, "temperature": temperature, "threshold": threshold, "threshold_info": threshold_info, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "trained_now": trained_now, "best_epoch": payload.get("best_epoch")})
        write_json(dirs["configs"] / f"polarmorph_calibration_fold_{fold + 1}.json", calibration_rows[-1])
        del model
        gc.collect()

    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(dirs["predictions"] / "dfu_polarmorphnet_oof_predictions.csv", index=False)
    xai = pd.concat(xai_rows, ignore_index=True) if xai_rows else pd.DataFrame()
    xai.to_csv(dirs["tables"] / "polarmorph_small_xai.csv", index=False)
    metric_rows = []
    for fold, frame in oof.groupby("outer_fold"):
        metric_rows.append({"scope": f"fold_{fold + 1}", **metric_dict(frame.label, frame.prob_calibrated, float(frame.threshold.iloc[0]))})
    aggregate_pred = oof.pred_calibrated.to_numpy()
    aggregate_metric = metric_dict(oof.label, oof.prob_calibrated, 0.5)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(oof.label, aggregate_pred, labels=[0, 1]).ravel()
    aggregate_metric.update({"scope": "OOF", "accuracy": float((aggregate_pred == oof.label.to_numpy()).mean()), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
    metrics = pd.DataFrame(metric_rows + [aggregate_metric])
    metrics.to_csv(dirs["tables"] / "polarmorph_fold_and_oof_metrics.csv", index=False)
    from IPython.display import display
    display(metrics.round(5))
    bootstrap_metric_cis(oof, cfg, dirs)
    make_core_figures(manifest, oof, pd.DataFrame([{"model": cfg.PRIMARY_MODEL_NAME, "state": "calibrated", **aggregate_metric}]), dirs)
    write_json(dirs["root"] / "software_versions.json", software_hardware_versions())
    payload = {"run_id": cfg.RUN_ID, "architecture": "DFU-PolarMorphNet", "config": asdict(cfg), "manifest": manifest.to_dict("records"), "predictions": oof.to_dict("records"), "metrics": metrics.to_dict("records"), "calibration": calibration_rows, "xai": xai.to_dict("records"), "checkpoint_paths": [row["checkpoint"] for row in calibration_rows], "raw_images_stored": False}
    pkl_path = dirs["root"] / "dfu_polarmorphnet_complete_reproducibility.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(dirs["root"] / "manifest.json", artifact_manifest(dirs["root"]))
    q1 = apply_q1_posthoc_corrections(dirs["root"], primary_model=cfg.PRIMARY_MODEL_NAME)
    github_status = push_to_github(dirs["root"], cfg, dirs)
    verification = {"run_id": cfg.RUN_ID, "architecture": cfg.PRIMARY_MODEL_NAME, "valid_folds": int(oof.outer_fold.nunique()), "valid_checkpoints": sum(Path(row["checkpoint"]).exists() for row in calibration_rows), "fits_executed_now": sum(bool(row["trained_now"]) for row in calibration_rows), "pkl_path": str(pkl_path), "drive_path": str(dirs["root"]), "github": github_status, "q1_posthoc": q1, "elapsed_minutes": (time.time() - started) / 60.0}
    write_json(dirs["root"] / "final_verification.json", verification)
    print(json.dumps(verification, indent=2, default=str))
    return verification


def regenerate_polarmorph_artifacts(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    prediction_path = run_root / "predictions" / "dfu_polarmorphnet_oof_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    predictions = pd.read_csv(prediction_path)
    result = apply_q1_posthoc_corrections(run_root, primary_model="DFU-PolarMorphNet")
    print(json.dumps(result, indent=2, default=str))
    return result
