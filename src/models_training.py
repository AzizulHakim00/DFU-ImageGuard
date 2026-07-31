from __future__ import annotations

import gc
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image

from .config_data import Config, seed_everything, seed_worker


class FrameDataset:
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        import torch

        row = self.frame.iloc[idx]
        with Image.open(row.image_path) as image:
            x = self.transform(image.convert("RGB"))
        y = torch.tensor(float(row.label), dtype=torch.float32)
        return x, y, idx


def build_transforms(cfg: Config):
    from torchvision import transforms

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_loader(
    frame: pd.DataFrame,
    transform,
    cfg: Config,
    shuffle: bool,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader

    if frame.empty:
        raise ValueError("Cannot build a DataLoader from an empty dataframe")
    generator = torch.Generator().manual_seed(int(seed))
    workers = max(0, int(cfg.NUM_WORKERS))
    kwargs: dict[str, Any] = {
        "dataset": FrameDataset(frame, transform),
        "batch_size": max(1, int(cfg.BATCH_SIZE)),
        "shuffle": bool(shuffle),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _create_timm_model(model_name: str, pretrained: bool, **kwargs):
    import timm

    try:
        return timm.create_model(model_name, pretrained=pretrained, **kwargs)
    except Exception as exc:
        mode = "pretrained" if pretrained else "architecture-only"
        raise RuntimeError(
            f"Could not create {mode} timm model '{model_name}'. "
            "Check the internet connection, timm version, and optional HF_TOKEN."
        ) from exc


def build_proposed_model(cfg: Config, pretrained: bool = True):
    import torch
    import torch.nn as nn

    class LesionAwareAttention(nn.Module):
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            hidden = max(channels // reduction, 16)
            self.channel = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )
            self.spatial = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = x * self.channel(x)
            avg = torch.mean(x, dim=1, keepdim=True)
            maximum = torch.amax(x, dim=1, keepdim=True)
            return x * self.spatial(torch.cat([avg, maximum], dim=1))

    class GeM(nn.Module):
        def __init__(self, p: float = 3.0, eps: float = 1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            p = self.p.clamp(min=0.25, max=10.0)
            return torch.nn.functional.avg_pool2d(
                x.clamp(min=self.eps).pow(p),
                (x.size(-2), x.size(-1)),
            ).pow(1.0 / p).flatten(1)

    class DFUImageGuard(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _create_timm_model(
                cfg.PROPOSED_BACKBONE,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
            )
            channels = int(self.backbone.num_features)
            self.lesion_attention = LesionAwareAttention(channels)
            self.pool = GeM()
            self.dropout = nn.Dropout(cfg.DROPOUT)
            self.head = nn.Linear(channels, 1)

        def forward(self, x):
            features = self.backbone(x)
            if features.ndim != 4:
                raise RuntimeError(
                    f"Expected a spatial feature map from {cfg.PROPOSED_BACKBONE}; got shape {tuple(features.shape)}"
                )
            features = self.lesion_attention(features)
            return self.head(self.dropout(self.pool(features))).squeeze(1)

    return DFUImageGuard()


def build_baseline_model(timm_name: str, pretrained: bool = True):
    return _create_timm_model(timm_name, pretrained=pretrained, num_classes=1)


def instantiate_model(factory: Callable[..., Any], pretrained: bool):
    try:
        return factory(pretrained=pretrained)
    except TypeError as exc:
        if "pretrained" not in str(exc):
            raise
        return factory()


def _binary_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) != 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, yb, idx in loader:
            out = model(xb.to(device, non_blocking=device.type == "cuda")).reshape(-1)
            logits.append(out.detach().float().cpu().numpy())
            labels.append(yb.numpy())
            indices.append(idx.numpy())
    if not logits:
        raise RuntimeError("Prediction loader produced no batches")
    logit_array = np.concatenate(logits)
    if not np.isfinite(logit_array).all():
        raise FloatingPointError("Model produced non-finite logits")
    return logit_array, np.concatenate(labels).astype(int), np.concatenate(indices)


def _atomic_torch_save(payload: dict[str, Any], checkpoint: Path) -> None:
    import torch

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, checkpoint)


def train_model(
    model,
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    cfg: Config,
    checkpoint: Path,
    history_path: Path,
    seed: int,
) -> tuple[Any, pd.DataFrame, bool]:
    import torch
    from torch import nn

    if train_df.empty or selection_df.empty:
        raise ValueError("Training and selection partitions must both be non-empty")
    if train_df.label.nunique() != 2 or selection_df.label.nunique() != 2:
        raise ValueError("Training and selection partitions must each contain both classes")

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_tf, eval_tf = build_transforms(cfg)
    train_loader = build_loader(train_df, train_tf, cfg, True, seed)
    selection_loader = build_loader(selection_df, eval_tf, cfg, False, seed + 1)

    positives = float((train_df.label == 1).sum())
    negatives = float((train_df.label == 0).sum())
    pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.MAX_EPOCHS, 1),
    )
    amp_enabled = bool(cfg.USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_auc = -np.inf
    best_epoch = -1
    patience_left = int(cfg.PATIENCE)
    history: list[dict[str, float]] = []
    history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg.MAX_EPOCHS) + 1):
        model.train()
        running_loss = 0.0
        n_examples = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(xb).reshape(-1)
                loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu()) * len(yb)
            n_examples += len(yb)

        scheduler.step()
        selection_logits, selection_y, _ = predict_logits(model, selection_loader, device)
        selection_prob = 1 / (1 + np.exp(-np.clip(selection_logits, -30, 30)))
        selection_auc = _binary_auc(selection_y, selection_prob)
        if not np.isfinite(selection_auc):
            raise FloatingPointError("Selection ROC-AUC is non-finite")

        history.append({
            "epoch": epoch,
            "train_loss": running_loss / max(n_examples, 1),
            "selection_auc": selection_auc,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        pd.DataFrame(history).to_csv(history_path, index=False)

        if selection_auc > best_auc + 1e-5:
            best_auc = selection_auc
            best_epoch = epoch
            patience_left = int(cfg.PATIENCE)
            _atomic_torch_save({
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_selection_auc": best_auc,
                "seed": int(seed),
                "config": asdict(cfg),
            }, checkpoint)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    history_frame = pd.DataFrame(history)
    if not checkpoint.exists():
        raise RuntimeError("Training finished without a valid checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        raise RuntimeError(f"Invalid checkpoint payload: {checkpoint}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, history_frame, True


def load_checkpoint_model(factory: Callable[..., Any], checkpoint: Path, device):
    import torch

    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint is missing or empty: {checkpoint}")
    model = instantiate_model(factory, pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Invalid checkpoint format: {checkpoint}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def release_torch_memory(*objects: Any) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
