from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolarMorphConfig:
    stem_channels: int = 64
    feature_channels: int = 256
    radial_rings: int = 8
    angular_sectors: int = 24
    radial_depth: int = 3
    contour_depth: int = 3
    dropout: float = 0.25
    grl_lambda: float = 0.20


def build_polarmorph_model(cfg: Any | None = None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    pcfg = PolarMorphConfig(
        stem_channels=int(getattr(cfg, "POLAR_STEM_CHANNELS", 64)),
        feature_channels=int(getattr(cfg, "POLAR_FEATURE_CHANNELS", 256)),
        radial_rings=int(getattr(cfg, "POLAR_RADIAL_RINGS", 8)),
        angular_sectors=int(getattr(cfg, "POLAR_ANGULAR_SECTORS", 24)),
        radial_depth=int(getattr(cfg, "POLAR_RADIAL_DEPTH", 3)),
        contour_depth=int(getattr(cfg, "POLAR_CONTOUR_DEPTH", 3)),
        dropout=float(getattr(cfg, "DROPOUT", 0.25)),
        grl_lambda=float(getattr(cfg, "POLAR_GRL_LAMBDA", 0.20)),
    )

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale):
            ctx.scale = float(scale)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.scale * grad_output, None

    class MorphBlock(nn.Module):
        def __init__(self, channels: int, dilation: int = 2, drop_path: float = 0.0):
            super().__init__()
            self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
            self.context = nn.Conv2d(channels, channels, 5, padding=2 * dilation, dilation=dilation, groups=channels)
            self.norm = nn.GroupNorm(1, channels)
            self.mix = nn.Sequential(nn.Conv2d(channels, 4 * channels, 1), nn.GELU(), nn.Conv2d(4 * channels, channels, 1))
            self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
            self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
            self.drop = nn.Dropout2d(drop_path) if drop_path > 0 else nn.Identity()

        def forward(self, x):
            morphology = self.local(x) + self.context(x)
            morphology = self.mix(self.norm(morphology))
            return x + self.drop(self.scale * self.gate(x) * morphology)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            c0, c1, c2 = pcfg.stem_channels, pcfg.stem_channels * 2, pcfg.feature_channels
            self.stem = nn.Sequential(nn.Conv2d(3, c0, 4, stride=4), nn.GroupNorm(1, c0), nn.GELU(), MorphBlock(c0, 1), MorphBlock(c0, 2))
            self.stage1 = nn.Sequential(nn.Conv2d(c0, c1, 2, stride=2), nn.GroupNorm(1, c1), MorphBlock(c1, 1), MorphBlock(c1, 2), MorphBlock(c1, 3))
            self.stage2 = nn.Sequential(nn.Conv2d(c1, c2, 2, stride=2), nn.GroupNorm(1, c2), MorphBlock(c2, 1), MorphBlock(c2, 2), MorphBlock(c2, 3))

        def forward(self, x):
            return self.stage2(self.stage1(self.stem(x)))

    class LesionDiscovery(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.mask_head = nn.Sequential(nn.Conv2d(3 * channels, channels // 2, 3, padding=1), nn.GELU(), nn.Conv2d(channels // 2, 1, 1))

        def forward(self, features):
            gx = F.pad(features[..., :, 1:] - features[..., :, :-1], (0, 1, 0, 0))
            gy = F.pad(features[..., 1:, :] - features[..., :-1, :], (0, 0, 0, 1))
            mask = torch.sigmoid(self.mask_head(torch.cat([features, gx.abs(), gy.abs()], dim=1)))
            weights = mask / mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            h, w = features.shape[-2:]
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=features.device, dtype=features.dtype), torch.linspace(-1, 1, w, device=features.device, dtype=features.dtype), indexing="ij")
            cx = (weights[:, 0] * xx).sum(dim=(1, 2))
            cy = (weights[:, 0] * yy).sum(dim=(1, 2))
            distance2 = (xx[None] - cx[:, None, None]).square() + (yy[None] - cy[:, None, None]).square()
            scale = torch.sqrt((weights[:, 0] * distance2).sum(dim=(1, 2)).clamp_min(1e-6)).clamp(0.20, 1.25)
            return mask, torch.stack([cx, cy], dim=1), scale

    class PolarTokenizer(nn.Module):
        def __init__(self, rings: int, sectors: int):
            super().__init__()
            self.register_buffer("radius", torch.linspace(0.05, 1.0, rings), persistent=True)
            self.register_buffer("angle", torch.arange(sectors, dtype=torch.float32) * (2.0 * torch.pi / sectors), persistent=True)
            self.radius_offset = nn.Parameter(torch.zeros(rings, sectors))
            self.angle_offset = nn.Parameter(torch.zeros(rings, sectors))

        def forward(self, features, center, scale):
            radius = self.radius[:, None] + 0.10 * torch.tanh(self.radius_offset)
            angle = self.angle[None, :] + 0.15 * torch.tanh(self.angle_offset)
            gx = center[:, 0, None, None] + scale[:, None, None] * radius[None] * torch.cos(angle)[None]
            gy = center[:, 1, None, None] + scale[:, None, None] * radius[None] * torch.sin(angle)[None]
            grid = torch.stack([gx, gy], dim=-1).clamp(-1.15, 1.15)
            polar = F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=True)
            return polar, grid.reshape(features.shape[0], -1, 2)

    class SequenceMixer(nn.Module):
        def __init__(self, channels: int, depth: int, cyclic: bool = False):
            super().__init__()
            self.cyclic = cyclic
            self.blocks = nn.ModuleList([nn.ModuleDict({"norm": nn.LayerNorm(channels), "dw": nn.Conv1d(channels, channels, 5, groups=channels), "gate": nn.Linear(channels, channels), "mix": nn.Linear(channels, channels)}) for _ in range(depth)])

        def forward(self, x):
            for block in self.blocks:
                residual = x
                z = block["norm"](x).transpose(1, 2)
                z = F.pad(z, (2, 2), mode="circular" if self.cyclic else "replicate")
                z = block["dw"](z).transpose(1, 2)
                x = residual + torch.sigmoid(block["gate"](z)) * block["mix"](z)
            return x

    class BidirectionalRadialEncoder(nn.Module):
        def __init__(self, channels: int, depth: int):
            super().__init__()
            self.forward_mixer = SequenceMixer(channels, depth)
            self.backward_mixer = SequenceMixer(channels, depth)
            self.fuse = nn.Linear(2 * channels, channels)

        def forward(self, polar):
            b, c, r, a = polar.shape
            sequence = polar.permute(0, 3, 2, 1).reshape(b * a, r, c)
            forward = self.forward_mixer(sequence)
            backward = torch.flip(self.backward_mixer(torch.flip(sequence, dims=[1])), dims=[1])
            return self.fuse(torch.cat([forward, backward], dim=-1)).reshape(b, a, r, c).permute(0, 2, 1, 3)

    class DFUPolarMorphNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = pcfg.feature_channels
            self.encoder = Encoder()
            self.discovery = LesionDiscovery(c)
            self.tokenizer = PolarTokenizer(pcfg.radial_rings, pcfg.angular_sectors)
            self.radial = BidirectionalRadialEncoder(c, pcfg.radial_depth)
            self.contour = SequenceMixer(c, pcfg.contour_depth, cyclic=True)
            self.global_mixer = nn.Sequential(MorphBlock(c, 2), MorphBlock(c, 3))
            self.fusion_gate = nn.Sequential(nn.Linear(3 * c, c), nn.GELU(), nn.Linear(c, 3))
            self.classifier = nn.Sequential(nn.LayerNorm(c), nn.Dropout(pcfg.dropout), nn.Linear(c, 1))
            self.background_classifier = nn.Sequential(nn.LayerNorm(c), nn.Dropout(pcfg.dropout), nn.Linear(c, 1))
            self.quality_head = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, 1))

        def encode(self, x):
            features = self.encoder(x)
            mask, center, scale = self.discovery(features)
            polar, sampling_grid = self.tokenizer(features, center, scale)
            radial_tokens = self.radial(polar)
            radial_vector = radial_tokens.mean(dim=(1, 2))
            boundary_index = min(max(pcfg.radial_rings // 2, 0), pcfg.radial_rings - 1)
            contour_tokens = self.contour(radial_tokens[:, boundary_index])
            contour_vector = contour_tokens.mean(dim=1)
            global_vector = self.global_mixer(features).mean(dim=(2, 3))
            joined = torch.cat([radial_vector, contour_vector, global_vector], dim=1)
            weights = torch.softmax(self.fusion_gate(joined), dim=1)
            fused = weights[:, 0:1] * radial_vector + weights[:, 1:2] * contour_vector + weights[:, 2:3] * global_vector
            background = ((1.0 - mask) * features).sum(dim=(2, 3)) / (1.0 - mask).sum(dim=(2, 3)).clamp_min(1e-6)
            return {"features": features, "mask": mask, "center": center, "scale": scale, "sampling_grid": sampling_grid, "radial_tokens": radial_tokens, "contour_tokens": contour_tokens, "fusion_weights": weights, "embedding": fused, "background_embedding": background}

        def forward(self, x, return_aux: bool = False):
            state = self.encode(x)
            logits = self.classifier(state["embedding"]).squeeze(1)
            reversed_background = GradientReverse.apply(state["background_embedding"], pcfg.grl_lambda)
            background_logits = self.background_classifier(reversed_background).squeeze(1)
            quality_logits = self.quality_head(state["embedding"]).squeeze(1)
            if not return_aux:
                return logits
            state.update({"logits": logits, "background_logits": background_logits, "quality_logits": quality_logits})
            return state

    model = DFUPolarMorphNet()
    model.architecture_name = "DFU-PolarMorphNet"
    model.architecture_config = pcfg
    return model
