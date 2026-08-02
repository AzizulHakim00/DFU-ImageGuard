from __future__ import annotations

import gc
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from .config_data import Config
from .evaluation import sigmoid_np
from .models_training import FrameDataset, predict_logits
from .reliability_io import ReliabilitySettings, _sync_trial_verified, atomic_csv
from .reliability_models import create_classifier
from .reliability_training import _augment_metrics, _device, _trial_paths

class CorruptionTransform:
    def __init__(self, image_size: int, kind: str, severity: float, seed: int):
        self.image_size = int(image_size)
        self.kind = kind
        self.severity = float(severity)
        self.seed = int(seed)

    def __call__(self, image: Image.Image):
        import torch
        from torchvision.transforms import functional as TF

        image = image.convert("RGB").resize((self.image_size, self.image_size))
        if self.kind == "brightness":
            image = ImageEnhance.Brightness(image).enhance(self.severity)
        elif self.kind == "contrast":
            image = ImageEnhance.Contrast(image).enhance(self.severity)
        elif self.kind == "gaussian_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=self.severity))
        elif self.kind == "jpeg_quality":
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=int(self.severity))
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        elif self.kind == "rotation":
            image = image.rotate(self.severity, resample=Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(image)
        if self.kind == "gaussian_noise":
            pixel_seed = self.seed + int(float(tensor.sum()) * 1000) % 1_000_003
            generator = torch.Generator().manual_seed(pixel_seed)
            noise = torch.randn(tensor.shape, generator=generator) * self.severity
            tensor = (tensor + noise).clamp(0, 1)
        elif self.kind == "occlusion":
            side = max(1, int(self.image_size * np.sqrt(self.severity)))
            start = (self.image_size - side) // 2
            tensor[:, start : start + side, start : start + side] = 0.5
        return TF.normalize(
            tensor,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )


def _run_trial_robustness(
    *,
    settings: ReliabilitySettings,
    cfg: Config,
    run_root: Path,
    model_key: str,
    seed: int,
    fold: int,
    test_df: pd.DataFrame,
    temperature: float,
    threshold: float,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    if model_key != settings.primary_model_key or not settings.run_robustness:
        return pd.DataFrame()
    paths = _trial_paths(run_root, model_key, seed, fold)
    device = _device()
    payload = torch.load(paths["best"], map_location=device, weights_only=False)
    model = create_classifier(
        model_key,
        pretrained=False,
        dropout=settings.dropout,
        resolved_name=str(payload["resolved_name"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    base = test_df.reset_index(drop=True)
    for corruption, levels in settings.robustness_levels.items():
        for level_index, severity in enumerate(levels, start=1):
            transform = CorruptionTransform(
                cfg.IMAGE_SIZE,
                corruption,
                severity,
                seed + fold * 1000 + level_index,
            )
            generator = torch.Generator().manual_seed(seed + fold * 1000 + level_index)
            loader = DataLoader(
                FrameDataset(base, transform),
                batch_size=cfg.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=True,
                worker_init_fn=None,
                generator=generator,
            )
            logits, y, _ = predict_logits(model, loader, device)
            if not np.array_equal(y, base.label.to_numpy(dtype=int)):
                raise AssertionError("Robustness DataLoader order changed")
            probability = sigmoid_np(logits / temperature)
            decision = (probability >= threshold).astype(int)
            metrics = _augment_metrics(y, probability, decision)
            metric_rows.append(
                {
                    "model_key": model_key,
                    "seed": int(seed),
                    "outer_fold": int(fold + 1),
                    "corruption": corruption,
                    "level": int(level_index),
                    "severity": float(severity),
                    **metrics,
                }
            )
            prediction = base[["image_id", "group_id", "label", "relative_path"]].copy()
            prediction["model_key"] = model_key
            prediction["seed"] = int(seed)
            prediction["outer_fold"] = int(fold + 1)
            prediction["corruption"] = corruption
            prediction["level"] = int(level_index)
            prediction["severity"] = float(severity)
            prediction["temperature_frozen"] = float(temperature)
            prediction["threshold_frozen"] = float(threshold)
            prediction["prob_calibrated"] = probability
            prediction["pred_calibrated"] = decision
            prediction_frames.append(prediction)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    atomic_csv(predictions, paths["root"] / "robustness_predictions.csv")
    atomic_csv(pd.DataFrame(metric_rows), paths["root"] / "robustness_metrics.csv")
    if settings.require_secondary_backup:
        _sync_trial_verified(paths["root"], run_root, settings)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions
