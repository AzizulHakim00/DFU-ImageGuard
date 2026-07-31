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


def build_loader(frame: pd.DataFrame, transform, cfg: Config, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class FrameDataset(Dataset):
        def __init__(self, df): self.df = df.reset_index(drop=True)
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            with Image.open(row.image_path) as im:
                x = transform(im.convert("RGB"))
            return x, torch.tensor(float(row.label), dtype=torch.float32), idx

    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        FrameDataset(frame), batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        worker_init_fn=_worker_init, generator=g, persistent_workers=cfg.NUM_WORKERS > 0,
    )


def build_proposed_model(cfg: Config):
    import torch
    import torch.nn as nn
    import timm

    class LesionAwareAttention(nn.Module):
        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            hidden = max(channels // reduction, 16)
            self.channel = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.GELU(),
                nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
            )
            self.spatial = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid()
            )
        def forward(self, x):
            x = x * self.channel(x)
            avg = torch.mean(x, dim=1, keepdim=True)
            mx = torch.amax(x, dim=1, keepdim=True)
            return x * self.spatial(torch.cat([avg, mx], dim=1))

    class GeM(nn.Module):
        def __init__(self, p: float = 3.0, eps: float = 1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps
        def forward(self, x):
            return torch.nn.functional.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
            ).pow(1.0 / self.p).flatten(1)

    class DFUImageGuard(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(cfg.PROPOSED_BACKBONE, pretrained=True,
                                              num_classes=0, global_pool="")
            channels = int(self.backbone.num_features)
            self.lesion_attention = LesionAwareAttention(channels)
            self.pool = GeM()
            self.dropout = nn.Dropout(cfg.DROPOUT)
            self.head = nn.Linear(channels, 1)
        def forward(self, x):
            feat = self.backbone(x)
            feat = self.lesion_attention(feat)
            return self.head(self.dropout(self.pool(feat))).squeeze(1)

    return DFUImageGuard()


def build_baseline_model(timm_name: str):
    import timm
    return timm.create_model(timm_name, pretrained=True, num_classes=1)


def _binary_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    model.eval()
    logits, labels, indices = [], [], []
    with torch.inference_mode():
        for xb, yb, idx in loader:
            out = model(xb.to(device, non_blocking=True)).reshape(-1)
            logits.append(out.detach().cpu().numpy())
            labels.append(yb.numpy())
            indices.append(idx.numpy())
    return np.concatenate(logits), np.concatenate(labels).astype(int), np.concatenate(indices)


def train_model(model, train_df: pd.DataFrame, selection_df: pd.DataFrame, cfg: Config,
                checkpoint: Path, history_path: Path, seed: int) -> tuple[Any, pd.DataFrame, bool]:
    import torch
    from torch import nn

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_tf, eval_tf = build_transforms(cfg)
    train_loader = build_loader(train_df, train_tf, cfg, True, seed)
    sel_loader = build_loader(selection_df, eval_tf, cfg, False, seed + 1)
    pos = float((train_df.label == 1).sum())
    neg = float((train_df.label == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.MAX_EPOCHS, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc, best_epoch, patience_left = -np.inf, -1, cfg.PATIENCE
    history: list[dict[str, float]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train()
        running_loss, n = 0.0, 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb).reshape(-1)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(yb)
            n += len(yb)
        scheduler.step()
        val_logits, val_y, _ = predict_logits(model, sel_loader, device)
        val_prob = 1 / (1 + np.exp(-np.clip(val_logits, -30, 30)))
        val_auc = _binary_auc(val_y, val_prob)
        history.append({
            "epoch": epoch, "train_loss": running_loss / max(n, 1),
            "selection_auc": val_auc, "lr": optimizer.param_groups[0]["lr"],
        })
        if val_auc > best_auc + 1e-5:
            best_auc, best_epoch, patience_left = val_auc, epoch, cfg.PATIENCE
            torch.save({
                "model_state_dict": model.state_dict(), "best_epoch": best_epoch,
                "best_selection_auc": best_auc, "seed": seed, "config": asdict(cfg),
            }, checkpoint)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    hist = pd.DataFrame(history)
    hist.to_csv(history_path, index=False)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return model, hist, True


def load_checkpoint_model(model_factory: Callable[[], Any], checkpoint: Path, device):
    import torch
    model = model_factory().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
