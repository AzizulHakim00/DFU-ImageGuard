from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    candidates: tuple[str, ...]
    primary: bool = False


MODEL_SPECS: dict[str, ModelSpec] = {
    "convnextv2_tiny": ModelSpec(
        key="convnextv2_tiny",
        display_name="ConvNeXtV2-Tiny",
        candidates=(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            "convnextv2_tiny.fcmae_ft_in1k",
            "convnextv2_tiny.fcmae",
        ),
        primary=True,
    ),
    "mobilenetv3_large": ModelSpec(
        key="mobilenetv3_large",
        display_name="MobileNetV3-Large",
        candidates=(
            "mobilenetv3_large_100.ra_in1k",
            "mobilenetv3_large_100",
        ),
    ),
    "densenet121": ModelSpec(
        key="densenet121",
        display_name="DenseNet121",
        candidates=(
            "densenet121.ra_in1k",
            "densenet121",
        ),
    ),
}


def _backbone_output_features(backbone: Any) -> int:
    """Return the dimension produced by a timm model with num_classes=0.

    Most timm models expose the final pooled dimension as ``num_features``.
    MobileNetV3 additionally retains a 1280-dimensional pre-classifier head
    after ``num_classes=0`` while ``num_features`` still reports 960.  In that
    case ``head_hidden_size`` is the actual forward-output dimension.
    """

    head_hidden = int(getattr(backbone, "head_hidden_size", 0) or 0)
    num_features = int(getattr(backbone, "num_features", 0) or 0)
    output_features = head_hidden or num_features
    if output_features <= 0:
        raise RuntimeError(
            "The timm backbone does not expose a valid pooled feature dimension"
        )
    return output_features


class BinaryImageClassifier:
    """Thin timm wrapper with an explicit dropout and one-logit head."""

    def __new__(
        cls,
        backbone: Any,
        model_key: str,
        resolved_name: str,
        dropout: float,
    ):
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.dropout = nn.Dropout(float(dropout))
                self.feature_dim = _backbone_output_features(backbone)
                self.head = nn.Linear(self.feature_dim, 1)
                self.model_key = model_key
                self.resolved_name = resolved_name

            def forward(self, x):
                features = self.backbone(x)
                if features.ndim > 2:
                    features = features.flatten(1)
                if features.ndim != 2 or features.shape[1] != self.feature_dim:
                    raise RuntimeError(
                        f"Unexpected pooled feature shape {tuple(features.shape)} "
                        f"for {self.resolved_name}; expected (*, {self.feature_dim})"
                    )
                return self.head(self.dropout(features)).squeeze(1)

        return _Impl()


def create_classifier(
    model_key: str,
    *,
    pretrained: bool = True,
    dropout: float = 0.20,
    resolved_name: str | None = None,
):
    import timm

    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key: {model_key}")
    spec = MODEL_SPECS[model_key]
    candidates = (resolved_name,) if resolved_name else spec.candidates
    errors: list[str] = []
    for name in candidates:
        try:
            backbone = timm.create_model(
                name,
                pretrained=bool(pretrained),
                num_classes=0,
                global_pool="avg",
            )
            model = BinaryImageClassifier(
                backbone=backbone,
                model_key=model_key,
                resolved_name=name,
                dropout=dropout,
            )
            return model
        except Exception as exc:  # pragma: no cover - depends on remote model registry
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    detail = "\n".join(errors)
    mode = "pretrained" if pretrained else "architecture-only"
    raise RuntimeError(
        f"Could not build {mode} classifier for {model_key}. Tried:\n{detail}"
    )


def parameter_summary(model) -> dict[str, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    backbone = sum(int(p.numel()) for p in model.backbone.parameters())
    head = total - backbone
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "backbone_parameters": backbone,
        "head_parameters": head,
        "pooled_feature_dimension": int(model.feature_dim),
    }


def find_last_conv2d(model):
    import torch.nn as nn

    last = None
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last = module
            last_name = name
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return last_name, last


def model_registry_table() -> list[dict[str, object]]:
    return [
        {
            "key": spec.key,
            "display_name": spec.display_name,
            "primary": spec.primary,
            "candidate_count": len(spec.candidates),
            "candidates": list(spec.candidates),
        }
        for spec in MODEL_SPECS.values()
    ]
