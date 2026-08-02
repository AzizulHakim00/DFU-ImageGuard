from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .config_data import Config
from .reliability_io import ReliabilitySettings, atomic_csv
from .reliability_models import create_classifier, find_last_conv2d
from .reliability_training import _device, _trial_paths


def select_gradcam_cases(
    predictions: pd.DataFrame,
    seed: int,
    limit: int,
) -> pd.DataFrame:
    frame = predictions[predictions.seed == seed].copy()
    false_negative = frame[
        (frame.label == 1) & (frame.pred_calibrated == 0)
    ].sort_values("confidence", ascending=False)
    false_positive = frame[
        (frame.label == 0) & (frame.pred_calibrated == 1)
    ].sort_values("confidence", ascending=False)
    uncertain = frame[frame.correct_calibrated == 1].sort_values("confidence")
    confident = frame[frame.correct_calibrated == 1].sort_values(
        "confidence", ascending=False
    )
    buckets = [
        ("false_negative", false_negative),
        ("false_positive", false_positive),
        ("uncertain_correct", uncertain),
        ("confident_correct", confident),
    ]
    per_bucket = max(1, limit // len(buckets))
    pieces = []
    for category, bucket in buckets:
        selected = bucket.head(per_bucket).copy()
        selected["xai_category"] = category
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True).head(limit)


def gradcam_image(model, image_tensor, target_layer, target_positive: bool):
    import torch
    import torch.nn.functional as F

    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(image_tensor).reshape(-1)[0]
        target = logit if target_positive else -logit
        target.backward()
        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy(), float(logit.detach().cpu())
    finally:
        forward_handle.remove()
        backward_handle.remove()


def generate_gradcam(
    *,
    settings: ReliabilitySettings,
    cfg: Config,
    run_root: Path,
    dirs: dict[str, Path],
    primary_predictions: pd.DataFrame,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import torch
    from torchvision import transforms

    if not settings.run_gradcam:
        return pd.DataFrame()
    cases = select_gradcam_cases(
        primary_predictions,
        settings.seeds[0],
        settings.gradcam_cases,
    )
    if cases.empty:
        return cases
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    device = _device()
    metadata_rows = []
    cache: dict[int, tuple[Any, str, Any]] = {}
    for _, row in cases.iterrows():
        fold = int(row.outer_fold) - 1
        if fold not in cache:
            paths = _trial_paths(
                run_root,
                settings.primary_model_key,
                settings.seeds[0],
                fold,
            )
            payload = torch.load(
                paths["best"], map_location=device, weights_only=False
            )
            model = create_classifier(
                settings.primary_model_key,
                pretrained=False,
                dropout=settings.dropout,
                resolved_name=str(payload["resolved_name"]),
            ).to(device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval()
            layer_name, layer = find_last_conv2d(model)
            cache[fold] = (model, layer_name, layer)
        model, layer_name, layer = cache[fold]
        with Image.open(row.image_path) as image:
            rgb = image.convert("RGB").resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
            tensor = transform(rgb).unsqueeze(0).to(device)
        cam, logit = gradcam_image(
            model,
            tensor,
            layer,
            target_positive=bool(int(row.pred_calibrated)),
        )
        output = dirs["xai"] / f"{row.xai_category}_{row.image_id}.png"
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb)
        plt.imshow(cam, alpha=0.42, cmap="jet")
        plt.axis("off")
        plt.title(
            f"{row.xai_category} | p={row.prob_calibrated:.3f} | y={int(row.label)}"
        )
        plt.tight_layout()
        plt.savefig(output, dpi=220, bbox_inches="tight")
        plt.close()
        metadata_rows.append(
            {
                "image_id": row.image_id,
                "relative_path": row.relative_path,
                "xai_category": row.xai_category,
                "outer_fold": int(row.outer_fold),
                "seed": int(settings.seeds[0]),
                "prob_calibrated": float(row.prob_calibrated),
                "label": int(row.label),
                "pred_calibrated": int(row.pred_calibrated),
                "target_layer": layer_name,
                "target_class": int(row.pred_calibrated),
                "recomputed_logit": logit,
                "figure": str(output),
                "interpretation_warning": (
                    "Grad-CAM is a model-attention visualization, not a lesion segmentation or clinical explanation."
                ),
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    atomic_csv(metadata, dirs["xai"] / "gradcam_metadata.csv")
    for model, _, _ in cache.values():
        del model
    gc.collect()
    torch.cuda.empty_cache()
    return metadata
