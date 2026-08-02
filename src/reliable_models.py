from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReliableModelSpec:
    dropout: float = 0.20
    candidates: dict[str, tuple[str, ...]] | None = None

    def resolved(self) -> dict[str, tuple[str, ...]]:
        return self.candidates or {
            "convnextv2_tiny": (
                "convnextv2_tiny.fcmae_ft_in22k_in1k",
                "convnextv2_tiny.fcmae_ft_in1k",
                "convnextv2_tiny.fcmae",
            ),
            "mobilenetv3_large": ("mobilenetv3_large_100.ra_in1k", "mobilenetv3_large_100"),
            "densenet121": ("densenet121.ra_in1k", "densenet121"),
        }


def build_reliable_model(model_key: str, pretrained: bool = True, spec: ReliableModelSpec | None = None):
    import timm

    spec = spec or ReliableModelSpec()
    candidates = spec.resolved()
    if model_key not in candidates:
        raise ValueError(f"Unknown model_key={model_key}; available={sorted(candidates)}")
    errors: list[str] = []
    for name in candidates[model_key]:
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=1, drop_rate=spec.dropout)
            model.model_key = model_key
            model.pretrained_model_name = name
            return model
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not create model: " + " | ".join(errors))


def parameter_summary(model) -> dict[str, int | str | None]:
    return {
        "model_key": getattr(model, "model_key", type(model).__name__),
        "pretrained_model_name": getattr(model, "pretrained_model_name", None),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
