from __future__ import annotations

import base64
import gc
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_data import (
    Config,
    assign_duplicate_groups,
    build_manifest,
    download_dataset,
    make_inner_partition,
    make_outer_folds,
    mount_drive,
    seed_everything,
    sha256_file,
    write_json,
)
from .evaluation import ece_mce, fit_temperature, select_threshold, sigmoid_np
from .models_training import FrameDataset
from .radial_adapter_model import RadialAdapterSpec, build_model, model_parameter_summary
from .checkpoint_backup import chunk_file_for_git, reconstruct_chunked_checkpoint


@dataclass
class PilotSettings:
    run_id: str = "RADIAL_ADAPTER_PILOT_V1"
    drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard"
    outer_fold: int = 0
    seeds: tuple[int, ...] = (2026, 2027)
    model_kinds: tuple[str, ...] = ("convnextv2_baseline", "dfu_radial_adapter")
    max_epochs: int = 25
    patience: int = 7
    freeze_epochs: int = 2
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    adapter_lr: float = 2e-4
    gate_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    target_sensitivity: float = 0.95
    use_amp: bool = True
    github_export: bool = True
    github_export_required: bool = True
    github_branch: str = "radial-pilot-results"
    github_max_file_bytes: int = 94 * 1024 * 1024
    github_chunk_full_checkpoints: bool = True
    github_chunk_bytes: int = 48 * 1024 * 1024
    github_export_after_each_trial: bool = True
    secondary_drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard-Backup"
    require_secondary_drive_backup: bool = True
    xai_cases: int = 4
    adapter_spec: RadialAdapterSpec = field(default_factory=RadialAdapterSpec)


MODEL_LABELS = {
    "convnextv2_baseline": "ConvNeXtV2-Tiny",
    "dfu_radial_adapter": "DFU-RadialAdapter",
}


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(type(value))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _storage_preflight(settings: PilotSettings) -> Path:
    root = Path(settings.drive_root)
    allowed_prefixes = [Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")]
    resolved = root.resolve()
    if not any(
        _is_relative_to(resolved, prefix.resolve())
        for prefix in allowed_prefixes
        if prefix.exists()
    ):
        raise RuntimeError(
            f"Pilot refuses non-Drive storage: {root}. "
            "Use a real MyDrive or Shared drives path."
        )
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / "runs" / settings.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    sentinel = run_root / "DRIVE_PERSISTENCE_SENTINEL.json"
    nonce = hashlib.sha256(f"{time.time_ns()}-{os.getpid()}".encode()).hexdigest()
    payload = {"nonce": nonce, "created_at_unix": time.time(), "run_root": str(run_root)}
    _atomic_json(sentinel, payload)
    loaded = json.loads(sentinel.read_text(encoding="utf-8"))
    if loaded.get("nonce") != nonce:
        raise RuntimeError("Google Drive write/read verification failed")
    print(f"DRIVE PERSISTENCE PREFLIGHT: PASS -> {run_root}")
    return run_root


def _pilot_transforms(settings: PilotSettings):
    from torchvision import transforms

    train = transforms.Compose([
        transforms.RandomResizedCrop(
            settings.image_size,
            scale=(0.90, 1.00),
            ratio=(0.95, 1.05),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    evaluate = transforms.Compose([
        transforms.Resize((settings.image_size, settings.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, evaluate


def _loader(frame: pd.DataFrame, transform, settings: PilotSettings, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader
    from .config_data import seed_worker

    generator = torch.Generator().manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "dataset": FrameDataset(frame.reset_index(drop=True), transform),
        "batch_size": int(settings.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": max(0, int(settings.num_workers)),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if settings.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for the pilot. Select a GPU Colab runtime.")
    return torch.device("cuda")


def _set_backbone_phase(model, unfrozen: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = bool(unfrozen)


def _optimizer(model, settings: PilotSettings):
    import torch

    backbone = []
    base_head = []
    adapter = []
    gate = []
    for name, parameter in model.named_parameters():
        if name.endswith("adapter.alpha"):
            gate.append(parameter)
        elif "adapter" in name:
            adapter.append(parameter)
        elif "backbone" in name:
            backbone.append(parameter)
        else:
            base_head.append(parameter)
    groups = [
        {"params": backbone, "lr": settings.backbone_lr, "name": "backbone"},
        {"params": base_head, "lr": settings.head_lr, "name": "base_head"},
    ]
    if adapter:
        groups.append({"params": adapter, "lr": settings.adapter_lr, "name": "adapter"})
    if gate:
        groups.append({"params": gate, "lr": settings.gate_lr, "weight_decay": 0.0, "name": "gate"})
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def _infer(model, loader, device, auxiliary: bool = False):
    import torch

    rows: dict[str, list[np.ndarray]] = {
        "logits": [],
        "base_logits": [],
        "adapter_logit": [],
        "adapter_contribution": [],
        "labels": [],
        "indices": [],
    }
    scalar_gate: float = 0.0
    model.eval()
    with torch.inference_mode():
        for xb, yb, idx in loader:
            output = model(xb.to(device, non_blocking=True), return_aux=True)
            for key in ["logits", "base_logits", "adapter_logit", "adapter_contribution"]:
                rows[key].append(output[key].detach().float().cpu().numpy().reshap(-1))
            rows["labels"].append(yb.numpy().reshap(-1))
            rows["indices"].append(idx.numpy().reshape(-1))
            scalar_gate = float(output["gate"].detach().float().cpu())
    result = {key: np.concatenate(value) for key, value in rows.items()}
    result["labels"] = result["labels"].astype(int)
    result["indices"] = result["indices"].astype(int)
    result["gate"] = scalar_gate
    return result


def _auc(labels: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, sigmoid_np(logits)))


def _portable_state_dict(model) -> dict[str, Any]:
    import torch

    portable = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu()
        if torch.is_floating_point(tensor):
            tensor = tensor.half()
        portable[name] = tensor
    return portable


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    settings: PilotSettings,
    model_kind: str,
    seed: int,
    epoch: int,
    best_auc: float,
    patience_left: int,
) 