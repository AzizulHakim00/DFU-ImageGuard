from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RadialAdapterSpec:
    model_candidates: tuple[str, ...] = (
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "convnextv2_tiny.fcmae_ft_in1k",
        "convnextv2_tiny.fcmae",
    )
    projection_dim: int = 96
    radial_rings: int = 5
    angular_sectors: int = 16
    dropout: float = 0.20


def _ensure_nchw(feature, expected_channels: int):
    if feature.ndim != 4:
        raise RuntimeError(f"Expected a 4-D feature map, received {tuple(feature.shape)}")
    if feature.shape[1] == expected_channels:
        return feature
    if feature.shape[-1] == expected_channels:
        return feature.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(
        f"Cannot identify channel axis for shape {tuple(feature.shape)}; "
        f"expected {expected_channels} channels"
    )


def create_convnextv2_feature_backbone(
    pretrained: bool = True,
    spec: RadialAdapterSpec | None = None,
):
    import timm

    spec = spec or RadialAdapterSpec()
    errors: list[str] = []
    for model_name in spec.model_candidates:
        try:
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(2, 3),
            )
            channels = tuple(int(c) for c in backbone.feature_info.channels())
            if len(channels) != 2:
                raise RuntimeError(f"Expected two feature stages, received {channels}")
            return backbone, channels, model_name
        except Exception as exc:  # pragma: no cover - depends on external model registry
            errors.append(f"{model_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "No compatible pretrained ConvNeXtV2-Tiny model could be created. "
        "Tried: " + " | ".join(errors)
    )


class RadialMorphologyAdapter:
    """Small lesion-centred radial/cyclic difference adapter.

    This class is intentionally independent of timm so its geometry can be unit-tested
    with synthetic tensors. It is wrapped as an nn.Module at construction time.
    """

    @staticmethod
    def build(in_channels: int, spec: RadialAdapterSpec | None = None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        spec = spec or RadialAdapterSpec()
        if spec.radial_rings < 3:
            raise ValueError("At least three radial rings are required")
        if spec.angular_sectors < 8:
            raise ValueError("At least eight angular sectors are required")

        class _Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                d = int(spec.projection_dim)
                self.spec = spec
                self.projection = nn.Sequential(
                    nn.Conv2d(in_channels, d, kernel_size=1, bias=False),
                    nn.GroupNorm(1, d),
                    nn.GELU(),
                )
                self.center_head = nn.Conv2d(d, 1, kernel_size=1)
                self.radial_mixer = nn.Sequential(
                    nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False),
                    nn.GELU(),
                    nn.Conv1d(d, d, kernel_size=1, bias=False),
                )
                self.angular_depthwise = nn.Conv1d(
                    d, d, kernel_size=3, padding=0, groups=d, bias=False
                )
                self.angular_pointwise = nn.Conv1d(d, d, kernel_size=1, bias=False)
                self.head = nn.Sequential(
                    nn.LayerNorm(2 * d),
                    nn.Linear(2 * d, d),
                    nn.GELU(),
                    nn.Dropout(float(spec.dropout)),
                    nn.Linear(d, 1),
                )
                self.alpha = nn.Parameter(torch.zeros(()))
                angles = torch.linspace(
                    0.0,
                    2.0 * torch.pi,
                    int(spec.angular_sectors) + 1,
                    dtype=torch.float32,
                )[:-1]
                radius_fractions = torch.linspace(
                    0.0, 1.0, int(spec.radial_rings), dtype=torch.float32
                )
                self.register_buffer("angles", angles, persistent=True)
                self.register_buffer("radius_fractions", radius_fractions, persistent=True)

            def _lesion_coordinate_system(self, projected):
                b, _, h, w = projected.shape
                logits = self.center_head(projected)
                attention = torch.softmax(logits.flatten(2), dim=-1).view(b, 1, h, w)
                x_axis = torch.linspace(-1.0, 1.0, w, device=projected.device, dtype=projected.dtype)
                y_axis = torch.linspace(-1.0, 1.0, h, device=projected.device, dtype=projected.dtype)
                yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
                cx = (attention[:, 0] * xx).sum(dim=(1, 2))
                cy = (attention[:, 0] * yy).sum(dim=(1, 2))
                variance = (
                    attention[:, 0]
                    * ((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2)
                ).sum(dim=(1, 2))
                scale = variance.clamp_min(1e-6).sqrt().clamp(0.18, 0.68)
                center = torch.stack([cx, cy], dim=1)
                return attention, center, scale

            def _sample_polar(self, projected, center, scale):
                b = projected.shape[0]
                radii = scale[:, None] * self.radius_fractions[None].to(projected.dtype) * 0.95
                cos_a = torch.cos(self.angles).to(projected.dtype)
                sin_a = torch.sin(self.angles).to(projected.dtype)
                gx = center[:, 0, None, None] + radii[:, :, None] * cos_a[None, None, :]
                gy = center[:, 1, None, None] + radii[:, :, None] * sin_a[None, None, :]
                grid = torch.stack([gx, gy], dim=-1).clamp(-1.0, 1.0)
                sampled = F.grid_sample(
                    projected,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                if sampled.shape[0] != b:
                    raise RuntimeError("Polar sampling changed the batch size")
                return sampled

            def forward(self, feature):
                projected = self.projection(feature)
                attention, center, scale = self._lesion_coordinate_system(projected)
                tokens = self._sample_polar(projected, center, scale)
                b, d, r, a = tokens.shape

                radial_difference = tokens[:, :, 1:, :] - tokens[:, :, :-1, :]
                radial_sequence = radial_difference.permute(0, 3, 1, 2).reshape(b * a, d, r - 1)
                radial_encoded = self.radial_mixer(radial_sequence).mean(dim=-1)
                radial_vector = radial_encoded.view(b, a, d).mean(dim=1)

                angular_difference = torch.roll(tokens, shifts=-1, dims=3) - tokens
                angular_sequence = angular_difference.permute(0, 2, 1, 3).reshape(b * r, d, a)
                angular_sequence = F.pad(angular_sequence, (1, 1), mode="circular")
                angular_encoded = self.angular_pointwise(
                    F.gelu(self.angular_depthwise(angular_sequence))
                ).mean(dim=-1)
                angular_vector = angular_encoded.view(b, r, d).mean(dim=1)

                morphology = torch.cat([radial_vector, angular_vector], dim=1)
                adapter_logit = self.head(morphology).squeeze(1)
                gate = torch.tanh(self.alpha)
                contribution = gate * adapter_logit
                return {
                    "adapter_logit": adapter_logit,
                    "adapter_contribution": contribution,
                    "gate": gate,
                    "attention": attention,
                    "center": center,
                    "scale": scale,
                    "polar_tokens": tokens,
                }

        return _Adapter()


class _CommonConvNeXtV2BinaryBase:
    @staticmethod
    def build(pretrained: bool, spec: RadialAdapterSpec | None = None):
        import torch.nn as nn

        spec = spec or RadialAdapterSpec()
        backbone, channels, model_name = create_convnextv2_feature_backbone(pretrained, spec)

        class _Common(nn.Module):
            def __init__(self):
                super().__init__()
                self.spec = spec
                self.backbone = backbone
                self.feature_channels = channels
                self.pretrained_model_name = model_name
                self.global_norm = nn.LayerNorm(channels[1])
                self.global_dropout = nn.Dropout(float(spec.dropout))
                self.base_head = nn.Linear(channels[1], 1)

            def extract_features(self, x):
                features = self.backbone(x)
                if not isinstance(features, (list, tuple)) or len(features) != 2:
                    raise RuntimeError("ConvNeXtV2 feature extractor did not return two stages")
                stage3 = _ensure_nchw(features[0], self.feature_channels[0])
                stage4 = _ensure_nchw(features[1], self.feature_channels[1])
                return stage3, stage4

            def base_logits(self, stage4):
                pooled = stage4.mean(dim=(2, 3))
                return self.base_head(self.global_dropout(self.global_norm(pooled))).squeeze(1)

        return _Common()


class ConvNeXtV2BinaryBaseline:
    @staticmethod
    def build(pretrained: bool = True, spec: RadialAdapterSpec | None = None):
        import torch.nn as nn

        common = _CommonConvNeXtV2BinaryBase.build(pretrained, spec)

        class _Baseline(nn.Module):
            model_kind = "convnextv2_baseline"

            def __init__(self):
                super().__init__()
                self.common = common
                self.pretrained_model_name = common.pretrained_model_name

            @property
            def backbone(self):
                return self.common.backbone

            def forward(self, x, return_aux: bool = False):
                _, stage4 = self.common.extract_features(x)
                logits = self.common.base_logits(stage4)
                if return_aux:
                    return {
                        "logits": logits,
                        "base_logits": logits,
                        "adapter_logit": logits.new_zeros(logits.shape),
                        "adapter_contribution": logits.new_zeros(logits.shape),
                        "gate": logits.new_zeros(()),
                    }
                return logits

        return _Baseline()


class DFURadialAdapterNet:
    @staticmethod
    def build(pretrained: bool = True, spec: RadialAdapterSpec | None = None):
        import torch.nn as nn

        spec = spec or RadialAdapterSpec()
        common = _CommonConvNeXtV2BinaryBase.build(pretrained, spec)
        adapter = RadialMorphologyAdapter.build(common.feature_channels[0], spec)

        class _Model(nn.Module):
            model_kind = "dfu_radial_adapter"

            def __init__(self):
                super().__init__()
                self.common = common
                self.adapter = adapter
                self.pretrained_model_name = common.pretrained_model_name

            @property
            def backbone(self):
                return self.common.backbone

            def forward(self, x, return_aux: bool = False):
                stage3, stage4 = self.common.extract_features(x)
                base_logits = self.common.base_logits(stage4)
                auxiliary = self.adapter(stage3)
                logits = base_logits + auxiliary["adapter_contribution"]
                if return_aux:
                    return {"logits": logits, "base_logits": base_logits, **auxiliary}
                return logits

        return _Model()


def build_model(model_kind: str, pretrained: bool = True, spec: RadialAdapterSpec | None = None):
    if model_kind == "convnextv2_baseline":
        return ConvNeXtV2BinaryBaseline.build(pretrained=pretrained, spec=spec)
    if model_kind == "dfu_radial_adapter":
        return DFURadialAdapterNet.build(pretrained=pretrained, spec=spec)
    raise ValueError(f"Unknown model kind: {model_kind}")


def model_parameter_summary(model) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = sum(p.numel() for name, p in model.named_parameters() if "adapter" in name)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "adapter_parameters": int(adapter),
        "model_kind": getattr(model, "model_kind", type(model).__name__),
        "pretrained_model_name": getattr(model, "pretrained_model_name", None),
    }
