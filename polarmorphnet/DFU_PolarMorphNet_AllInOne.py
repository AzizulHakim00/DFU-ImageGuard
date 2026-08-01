from __future__ import annotations

"""DFU-PolarMorphNet V2: one-file, resumable, leakage-safe Colab research workflow.

The file deliberately contains the architecture, data audit, training, calibration,
figures, XAI, robustness, reproducibility bundle and GitHub export. It never
contains precomputed results and is not a medical device.
"""

import argparse
import contextlib
import dataclasses
import datetime as dt
import gc
import hashlib
import io
import json
import math
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class Config:
    REPO: str = "AzizulHakim00/DFU-ImageGuard"
    DATASET_SLUG: str = "laithjj/diabetic-foot-ulcer-dfu"
    ARCHITECTURE: str = "DFU-PolarMorphNet-V2"
    MODE: str = os.getenv("DFU_MODE", "train")
    RUN_ID: Optional[str] = os.getenv("DFU_RUN_ID") or None
    DRIVE_ROOT: str = os.getenv("DFU_DRIVE_ROOT", "/content/drive/MyDrive/DFU-PolarMorphNet")
    PREFERRED_FOLDS: str = os.getenv("DFU_PREFERRED_FOLDS", "")
    SEED: int = 2026
    N_FOLDS: int = 5
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    MAX_EPOCHS: int = 35
    PATIENCE: int = 7
    LR: float = 3e-4
    MIN_LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-4
    WARMUP_EPOCHS: int = 2
    AMP: bool = True
    TARGET_SENSITIVITY: float = 0.95
    BOOTSTRAP_REPS: int = 1000
    PHASH_DISTANCE: int = 4
    EMBEDDING_DUPLICATES: bool = True
    EMBEDDING_SIMILARITY: float = 0.995
    CLUSTER_BALANCED_SAMPLER: bool = True
    CLUSTER_WEIGHTED_LOSS: bool = True
    STEM_DIM: int = 64
    EMBED_DIM: int = 192
    RADIAL_RINGS: int = 8
    ANGULAR_SECTORS: int = 24
    DROPOUT: float = 0.25
    DROP_PATH: float = 0.12
    LAMBDA_BG: float = 0.08
    LAMBDA_CF: float = 0.12
    LAMBDA_MASK: float = 0.04
    BG_GRL_LAMBDA: float = 0.20
    RUN_XAI: bool = True
    XAI_CASES: int = 4
    RUN_ROBUSTNESS: bool = True
    ROBUSTNESS_MAX_PER_FOLD: int = 128
    FORCE_RETRAIN: bool = False
    GITHUB_PUSH: bool = True
    GITHUB_MAX_FILE_MB: int = 90
    FOLD_LEASE_HOURS: float = 24.0

    @property
    def preferred_fold_indices(self) -> list[int]:
        if not self.PREFERRED_FOLDS.strip():
            return list(range(self.N_FOLDS))
        values = sorted({int(x.strip()) - 1 for x in self.PREFERRED_FOLDS.split(",") if x.strip()})
        if any(x < 0 or x >= self.N_FOLDS for x in values):
            raise ValueError("PREFERRED_FOLDS must contain fold numbers 1..5")
        return values


def in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_id_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    seed = (os.getpid() + worker_id + int(time.time())) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def mount_drive() -> None:
    if not in_colab():
        return
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)


def make_dirs(root: Path) -> dict[str, Path]:
    names = ["models", "predictions", "manifests", "configs", "tables", "figures", "xai", "robustness", "logs", "locks"]
    dirs = {"root": root}
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        dirs[name] = root / name
        dirs[name].mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_run(cfg: Config) -> tuple[Path, dict[str, Path]]:
    mount_drive()
    project = Path(cfg.DRIVE_ROOT)
    project.mkdir(parents=True, exist_ok=True)
    active = project / "ACTIVE_POLARMORPHNET_RUN.txt"
    if cfg.RUN_ID is None:
        candidate = active.read_text().strip() if active.exists() else ""
        cfg.RUN_ID = candidate if candidate and (project / "runs" / candidate).exists() else run_id_now()
    active.write_text(cfg.RUN_ID + "\n", encoding="utf-8")
    return project, make_dirs(project / "runs" / cfg.RUN_ID)


def download_dataset(cfg: Config, dirs: dict[str, Path]) -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(cfg.DATASET_SLUG))
    write_json(dirs["root"] / "dataset_source.json", {
        "slug": cfg.DATASET_SLUG, "downloaded_at": utcnow(), "cache_root": str(root),
        "license": "unknown_or_undeclared; verify before redistribution",
    })
    return root


def find_patches_root(root: Path) -> Path:
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == "patches"]
    for candidate in sorted(candidates, key=lambda p: len(p.parts)):
        child_names = {p.name.lower() for p in candidate.iterdir() if p.is_dir()}
        if any("normal" in n for n in child_names) and any(any(k in n for k in ("abnormal", "ulcer", "dfu")) for n in child_names):
            return candidate
    raise RuntimeError("Strict Patches/Normal and Patches/Abnormal-or-Ulcer folders were not found")


def class_from_relative(relative: Path) -> tuple[int, str]:
    first = relative.parts[0].lower()
    if "normal" in first and not any(k in first for k in ("abnormal", "ulcer", "dfu")):
        return 0, "Normal"
    if any(k in first for k in ("abnormal", "ulcer", "dfu")):
        return 1, "DFU"
    raise ValueError(f"Ambiguous class folder: {relative.parts[0]}")


def pixel_hash(image: Image.Image) -> str:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_manifest(dataset_root: Path, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    import imagehash

    patches = find_patches_root(dataset_root)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    rows, corrupt = [], []
    for path in sorted(patches.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative = path.relative_to(patches)
        try:
            label, label_name = class_from_relative(relative)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                rows.append({
                    "image_id": hashlib.sha256(relative.as_posix().encode()).hexdigest()[:20],
                    "image_path": str(path), "relative_path": relative.as_posix(),
                    "label": label, "label_name": label_name, "width": width, "height": height,
                    "file_sha256": sha256_file(path), "pixel_sha256": pixel_hash(rgb),
                    "phash": str(imagehash.phash(rgb, hash_size=16)),
                })
        except Exception as exc:
            corrupt.append({"path": str(path), "error": repr(exc)})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.label.nunique() != 2:
        raise RuntimeError("Dataset audit did not produce both strict classes")
    write_csv(frame, dirs["manifests"] / "raw_manifest.csv")
    write_csv(pd.DataFrame(corrupt), dirs["tables"] / "corrupt_images.csv")
    return frame


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def phash_distance(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def exact_and_phash_groups(frame: pd.DataFrame, cfg: Config) -> tuple[UnionFind, list[dict[str, Any]]]:
    uf = UnionFind(len(frame))
    evidence: list[dict[str, Any]] = []
    for column, method in [("file_sha256", "file_sha256"), ("pixel_sha256", "pixel_sha256")]:
        for _, indices in frame.groupby(column).groups.items():
            indices = list(indices)
            for index in indices[1:]:
                uf.union(indices[0], index)
            if len(indices) > 1:
                evidence.append({"method": method, "members": indices, "size": len(indices)})
    for label, subset in frame.groupby("label"):
        indices = subset.index.to_list()
        hashes = subset.phash.to_list()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                distance = phash_distance(hashes[i], hashes[j])
                if distance <= cfg.PHASH_DISTANCE:
                    uf.union(indices[i], indices[j])
                    evidence.append({"method": "phash", "a": indices[i], "b": indices[j], "distance": distance, "label": int(label)})
    return uf, evidence


def add_embedding_groups(frame: pd.DataFrame, uf: UnionFind, cfg: Config, evidence: list[dict[str, Any]]) -> None:
    if not cfg.EMBEDDING_DUPLICATES:
        return
    import torch
    import timm
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=0).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(frame), 32):
            tensors = []
            for path in frame.image_path.iloc[start:start + 32]:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            feature = model(batch)
            feature = torch.nn.functional.normalize(feature, dim=1)
            vectors.append(feature.cpu().numpy())
    embedding = np.concatenate(vectors)
    np.save(frame.attrs["embedding_path"], embedding)
    for label, subset in frame.groupby("label"):
        idx = subset.index.to_numpy()
        similarity = embedding[idx] @ embedding[idx].T
        rows, cols = np.where(np.triu(similarity, 1) >= cfg.EMBEDDING_SIMILARITY)
        for row, col in zip(rows, cols):
            a, b = int(idx[row]), int(idx[col])
            uf.union(a, b)
            evidence.append({"method": "embedding", "a": a, "b": b, "similarity": float(similarity[row, col]), "label": int(label)})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assign_groups(frame: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    frame.attrs["embedding_path"] = str(dirs["manifests"] / "audit_embeddings.npy")
    uf, evidence = exact_and_phash_groups(frame, cfg)
    add_embedding_groups(frame, uf, cfg, evidence)
    roots = [uf.find(i) for i in range(len(frame))]
    mapping = {root: f"grp_{position:04d}" for position, root in enumerate(sorted(set(roots)))}
    frame["group_id"] = [mapping[root] for root in roots]
    sizes = frame.group_id.value_counts()
    frame["group_size"] = frame.group_id.map(sizes).astype(int)
    frame["cluster_weight"] = 1.0 / frame.group_size
    conflicts = frame.groupby("group_id").label.nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        write_csv(frame[frame.group_id.isin(conflicts.index)], dirs["tables"] / "cross_label_duplicate_conflicts.csv")
        raise RuntimeError("Cross-label duplicate/near-duplicate groups detected; manual label audit required")
    summary = frame.groupby("group_id").agg(n_images=("image_id", "size"), label=("label", "first")).reset_index()
    write_csv(frame, dirs["manifests"] / "grouped_manifest.csv")
    write_csv(summary, dirs["tables"] / "duplicate_groups.csv")
    write_json(dirs["tables"] / "duplicate_evidence.json", evidence)
    return frame


def make_outer_folds(frame: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    path = dirs["manifests"] / "outer_fold_assignments.csv"
    if path.exists():
        saved = pd.read_csv(path)
        merged = frame.drop(columns=["outer_fold"], errors="ignore").merge(
            saved[["image_id", "outer_fold"]], on="image_id", how="left", validate="one_to_one"
        )
        if merged.outer_fold.isna().any():
            raise RuntimeError("Saved fold assignments do not match the audited manifest")
        return merged
    splitter = StratifiedGroupKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    frame = frame.copy()
    frame["outer_fold"] = -1
    for fold, (_, test_idx) in enumerate(splitter.split(frame, frame.label, frame.group_id)):
        frame.loc[test_idx, "outer_fold"] = fold
    if (frame.outer_fold < 0).any():
        raise AssertionError("Incomplete outer fold assignment")
    for group, subset in frame.groupby("group_id"):
        if subset.outer_fold.nunique() != 1:
            raise AssertionError(f"Group leakage for {group}")
    write_csv(frame[["image_id", "relative_path", "label", "group_id", "outer_fold"]], path)
    write_json(dirs["root"] / "split_integrity_report.json", {
        "status": "PASS", "n_folds": cfg.N_FOLDS, "group_overlap": 0,
        "patient_level_split": False, "reason": "patient/case identifiers unavailable",
    })
    return frame


def make_inner_partition(outer_train: pd.DataFrame, fold: int, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    path = dirs["manifests"] / f"inner_partition_fold_{fold + 1}.csv"
    if path.exists():
        roles = pd.read_csv(path)
        return outer_train.merge(roles[["image_id", "inner_role"]], on="image_id", how="left", validate="one_to_one")
    first = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED + fold + 100)
    remain_idx, cal_idx = next(first.split(outer_train, outer_train.label, outer_train.group_id))
    remain = outer_train.iloc[remain_idx].copy()
    second = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.SEED + fold + 200)
    train_idx, select_idx = next(second.split(remain, remain.label, remain.group_id))
    role = pd.Series("", index=outer_train.index, dtype=object)
    role.iloc[cal_idx] = "calibration"
    role.loc[remain.iloc[train_idx].index] = "train"
    role.loc[remain.iloc[select_idx].index] = "selection"
    output = outer_train.copy()
    output["inner_role"] = role
    if (output.inner_role == "").any():
        raise AssertionError("Incomplete inner partition")
    for group, subset in output.groupby("group_id"):
        if subset.inner_role.nunique() != 1:
            raise AssertionError(f"Inner leakage for group {group}")
    write_csv(output[["image_id", "group_id", "label", "inner_role"]], path)
    return output


def remap_paths(frame: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    patches = find_patches_root(dataset_root)
    frame = frame.copy()
    frame["image_path"] = frame.relative_path.map(lambda x: str(patches / x))
    missing = [path for path in frame.image_path if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Could not remap {len(missing)} dataset paths")
    return frame


class DFUDataset:
    def __init__(self, frame: pd.DataFrame, transform: Any):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        import torch

        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor, "label": torch.tensor(float(row.label)),
            "cluster_weight": torch.tensor(float(row.cluster_weight)),
            "index": torch.tensor(index),
        }


def build_transforms(cfg: Config):
    from torchvision import transforms

    train = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE + 24, cfg.IMAGE_SIZE + 24)),
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.78, 1.0), ratio=(0.88, 1.12)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    evaluate = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, evaluate


def build_loader(frame: pd.DataFrame, transform: Any, cfg: Config, training: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = DFUDataset(frame, transform)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = training
    if training and cfg.CLUSTER_BALANCED_SAMPLER:
        weights = 1.0 / frame.group_size.to_numpy(dtype=float)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(frame), True, generator=generator)
        shuffle = False
    return DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, sampler=sampler,
        num_workers=cfg.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.NUM_WORKERS > 0, worker_init_fn=seed_worker, generator=generator,
    )


def drop_path(x, probability: float, training: bool):
    if probability <= 0 or not training:
        return x
    import torch

    keep = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.floor(keep + torch.rand(shape, device=x.device, dtype=x.dtype))
    return x * mask / keep


def build_model_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DropPath(nn.Module):
        def __init__(self, probability: float):
            super().__init__(); self.probability = probability
        def forward(self, x):
            return drop_path(x, self.probability, self.training)

    class MorphologyBlock(nn.Module):
        def __init__(self, channels: int, probability: float):
            super().__init__()
            self.norm = nn.GroupNorm(1, channels)
            self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
            self.context = nn.Conv2d(channels, channels, 5, padding=4, dilation=2, groups=channels)
            self.mix = nn.Sequential(nn.Conv2d(channels * 2, channels * 4, 1), nn.GELU(), nn.Conv2d(channels * 4, channels, 1))
            self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
            self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)
            self.path = DropPath(probability)
        def forward(self, x):
            z = self.norm(x)
            response = self.mix(torch.cat([self.local(z), self.context(z)], dim=1)) * self.gate(z)
            return x + self.path(self.gamma * response)

    class WeakLesionDiscovery(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.head = nn.Sequential(nn.Conv2d(channels * 3, channels // 2, 1), nn.GELU(), nn.Conv2d(channels // 2, 1, 1))
        def forward(self, feature):
            gx = F.pad(feature[..., 1:] - feature[..., :-1], (0, 1, 0, 0))
            gy = F.pad(feature[..., 1:, :] - feature[..., :-1, :], (0, 0, 0, 1))
            return torch.sigmoid(self.head(torch.cat([feature, gx.abs(), gy.abs()], dim=1)))

    class PolarTokenizer(nn.Module):
        def __init__(self, channels: int, dim: int, rings: int, sectors: int):
            super().__init__()
            self.rings, self.sectors = rings, sectors
            self.radius_steps = nn.Parameter(torch.zeros(rings))
            self.angle_offset = nn.Parameter(torch.zeros(1))
            self.project = nn.Conv2d(channels, dim, 1)
        def centre_scale(self, mask):
            b, _, h, w = mask.shape
            yy = torch.linspace(-1, 1, h, device=mask.device, dtype=mask.dtype)
            xx = torch.linspace(-1, 1, w, device=mask.device, dtype=mask.dtype)
            yg, xg = torch.meshgrid(yy, xx, indexing="ij")
            weight = mask[:, 0] + 1e-6; weight = weight / weight.sum((1, 2), keepdim=True)
            cx = (weight * xg).sum((1, 2)); cy = (weight * yg).sum((1, 2))
            variance = weight * ((xg - cx[:, None, None]) ** 2 + (yg - cy[:, None, None]) ** 2)
            scale = torch.clamp(torch.sqrt(variance.sum((1, 2)) + 1e-6) * 2.8, 0.35, 1.25)
            return cx, cy, scale
        def forward(self, feature, mask):
            projected = self.project(feature)
            cx, cy, scale = self.centre_scale(mask)
            radii = torch.cumsum(F.softplus(self.radius_steps) + 1e-3, 0); radii = radii / radii[-1]
            angles = torch.arange(self.sectors, device=feature.device, dtype=feature.dtype) * (2 * math.pi / self.sectors) + self.angle_offset
            rr = radii[None, :, None] * scale[:, None, None]; aa = angles[None, None, :]
            grid = torch.stack([cx[:, None, None] + rr * torch.cos(aa), cy[:, None, None] + rr * torch.sin(aa)], dim=-1)
            polar = F.grid_sample(projected, grid, mode="bilinear", padding_mode="border", align_corners=True)
            polar_mask = F.grid_sample(mask, grid, mode="bilinear", padding_mode="border", align_corners=True)
            return polar, polar_mask, grid, (cx, cy, scale)

    class SelectiveSSM(nn.Module):
        """Input-selective diagonal state-space recurrence implemented in pure PyTorch."""
        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.delta = nn.Linear(dim, dim)
            self.b = nn.Linear(dim, dim)
            self.c = nn.Linear(dim, dim)
            self.out = nn.Linear(dim, dim)
            self.a_log = nn.Parameter(torch.zeros(dim))
            self.skip = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            residual = x; x = self.norm(x)
            delta = F.softplus(self.delta(x)) + 1e-4
            b_term = self.b(x); c_term = torch.sigmoid(self.c(x))
            a = -F.softplus(self.a_log)[None, :]
            state = torch.zeros_like(x[:, 0]); outputs = []
            for step in range(x.shape[1]):
                transition = torch.exp(a * delta[:, step])
                state = transition * state + (1 - transition) * b_term[:, step]
                outputs.append(c_term[:, step] * state + self.skip * x[:, step])
            return residual + self.out(torch.stack(outputs, dim=1))

    class BidirectionalRadialSSM(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.forward_ssm = SelectiveSSM(dim); self.reverse_ssm = SelectiveSSM(dim)
            self.fuse = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        def forward(self, polar):
            b, d, r, a = polar.shape
            sequence = polar.permute(0, 3, 2, 1).reshape(b * a, r, d)
            forward = self.forward_ssm(sequence)
            reverse = torch.flip(self.reverse_ssm(torch.flip(sequence, dims=[1])), dims=[1])
            fused = self.fuse(torch.cat([forward, reverse], dim=-1))
            return fused.reshape(b, a, r, d).permute(0, 3, 2, 1)

    class CyclicContourMixer(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.depthwise = nn.Conv1d(dim, dim, 5, groups=dim)
            self.gate = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.Sigmoid())
            self.mix = nn.Sequential(nn.Conv1d(dim, dim * 2, 1), nn.GELU(), nn.Conv1d(dim * 2, dim, 1))
            self.norm = nn.LayerNorm(dim)
        def forward(self, radial, polar_mask):
            mask = polar_mask[:, 0]
            gradient = torch.zeros_like(mask); gradient[:, 1:] = (mask[:, 1:] - mask[:, :-1]).abs()
            weights = torch.softmax(8.0 * gradient, dim=1)
            contour = (radial * weights[:, None]).sum(dim=2)
            mixed = self.depthwise(F.pad(contour, (2, 2), mode="circular"))
            output = contour + self.mix(mixed) * self.gate(contour)
            return self.norm(output.transpose(1, 2)).transpose(1, 2)

    class GeM(nn.Module):
        def __init__(self):
            super().__init__(); self.p = nn.Parameter(torch.ones(1) * 3.0)
        def forward(self, x):
            p = torch.clamp(self.p, 1.0, 6.0)
            return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(p), 1).pow(1 / p).flatten(1)

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, strength):
            ctx.strength = strength; return x.view_as(x)
        @staticmethod
        def backward(ctx, gradient):
            return -ctx.strength * gradient, None

    class PolarMorphNet(nn.Module):
        def __init__(self, cfg: Config):
            super().__init__()
            c1, c2, c3 = cfg.STEM_DIM, cfg.STEM_DIM * 2, cfg.STEM_DIM * 4
            probabilities = torch.linspace(0, cfg.DROP_PATH, 8).tolist()
            self.cfg = cfg
            self.stem = nn.Sequential(nn.Conv2d(3, c1, 4, stride=4), nn.GroupNorm(1, c1))
            self.stage1 = nn.Sequential(MorphologyBlock(c1, probabilities[0]), MorphologyBlock(c1, probabilities[1]))
            self.down1 = nn.Sequential(nn.GroupNorm(1, c1), nn.Conv2d(c1, c2, 2, stride=2))
            self.stage2 = nn.Sequential(*[MorphologyBlock(c2, probabilities[i]) for i in range(2, 5)])
            self.down2 = nn.Sequential(nn.GroupNorm(1, c2), nn.Conv2d(c2, c3, 2, stride=2))
            self.stage3 = nn.Sequential(*[MorphologyBlock(c3, probabilities[i]) for i in range(5, 8)])
            self.lesion = WeakLesionDiscovery(c3)
            self.polar = PolarTokenizer(c3, cfg.EMBED_DIM, cfg.RADIAL_RINGS, cfg.ANGULAR_SECTORS)
            self.radial = BidirectionalRadialSSM(cfg.EMBED_DIM)
            self.contour = CyclicContourMixer(cfg.EMBED_DIM)
            self.global_project = nn.Conv2d(c3, cfg.EMBED_DIM, 1); self.global_pool = GeM()
            self.radial_norm = nn.LayerNorm(cfg.EMBED_DIM); self.contour_norm = nn.LayerNorm(cfg.EMBED_DIM)
            self.global_norm = nn.LayerNorm(cfg.EMBED_DIM)
            self.fusion_gate = nn.Sequential(nn.Linear(cfg.EMBED_DIM * 3, cfg.EMBED_DIM), nn.GELU(), nn.Linear(cfg.EMBED_DIM, 3))
            self.classifier = nn.Sequential(nn.LayerNorm(cfg.EMBED_DIM), nn.Dropout(cfg.DROPOUT), nn.Linear(cfg.EMBED_DIM, 1))
            self.background_project = nn.Sequential(nn.Conv2d(c3, cfg.EMBED_DIM, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.background_head = nn.Sequential(nn.LayerNorm(cfg.EMBED_DIM), nn.Linear(cfg.EMBED_DIM, 1))
            self.last_feature = None; self.last_mask = None
        def extract(self, x):
            x = self.stage1(self.stem(x)); x = self.stage2(self.down1(x)); return self.stage3(self.down2(x))
        def encode(self, feature, mask):
            polar, polar_mask, grid, geometry = self.polar(feature, mask)
            radial = self.radial(polar); contour = self.contour(radial, polar_mask)
            z_radial = self.radial_norm(radial.mean((2, 3)))
            z_contour = self.contour_norm(contour.mean(2))
            z_global = self.global_norm(self.global_pool(self.global_project(feature)))
            gates = torch.softmax(self.fusion_gate(torch.cat([z_radial, z_contour, z_global], 1)), 1)
            fused = gates[:, 0:1] * z_radial + gates[:, 1:2] * z_contour + gates[:, 2:3] * z_global
            return fused, {"polar_mask": polar_mask, "grid": grid, "geometry": geometry, "fusion_gates": gates}
        def forward(self, x, return_aux: bool = False):
            feature = self.extract(x); mask = self.lesion(feature)
            if feature.requires_grad: feature.retain_grad()
            self.last_feature, self.last_mask = feature, mask
            embedding, aux = self.encode(feature, mask); logits = self.classifier(embedding).squeeze(1)
            background = (1 - mask) * feature
            bg_embedding = self.background_project(background)
            bg_logits = self.background_head(GradientReverse.apply(bg_embedding, self.cfg.BG_GRL_LAMBDA)).squeeze(1)
            cf_logits = None
            if self.training and len(feature) > 1:
                shuffled = torch.roll(feature, 1, 0)
                cf_feature = mask * feature + (1 - mask) * shuffled
                cf_embedding, _ = self.encode(cf_feature, mask)
                cf_logits = self.classifier(cf_embedding).squeeze(1)
            aux.update({"mask": mask, "embedding": embedding, "background_logits": bg_logits, "counterfactual_logits": cf_logits})
            return (logits, aux) if return_aux else logits

    return PolarMorphNet


def instantiate_model(cfg: Config):
    return build_model_class()(cfg)


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def mask_regularization(mask):
    import torch

    tv = (mask[..., 1:] - mask[..., :-1]).abs().mean() + (mask[..., 1:, :] - mask[..., :-1, :]).abs().mean()
    area = (mask.mean((1, 2, 3)) - 0.25).abs().mean()
    return tv + 0.25 * area


def training_loss(logits, aux, labels, cluster_weight, cfg: Config):
    import torch
    import torch.nn.functional as F

    classification = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if cfg.CLUSTER_WEIGHTED_LOSS:
        classification = (classification * cluster_weight).sum() / cluster_weight.sum().clamp_min(1e-6)
    else:
        classification = classification.mean()
    background = F.binary_cross_entropy_with_logits(aux["background_logits"], labels)
    counterfactual = torch.tensor(0.0, device=labels.device)
    if aux["counterfactual_logits"] is not None:
        counterfactual = F.mse_loss(torch.sigmoid(aux["counterfactual_logits"]), torch.sigmoid(logits.detach()))
    mask_loss = mask_regularization(aux["mask"])
    total = classification + cfg.LAMBDA_BG * background + cfg.LAMBDA_CF * counterfactual + cfg.LAMBDA_MASK * mask_loss
    return total, {
        "classification": float(classification.detach()), "background": float(background.detach()),
        "counterfactual": float(counterfactual.detach()), "mask": float(mask_loss.detach()),
    }


def predict_loader(model, loader, device, return_embeddings: bool = False) -> dict[str, np.ndarray]:
    import torch

    model.eval(); logits, labels, indices, embeddings = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            output, aux = model(images, return_aux=True)
            logits.append(output.cpu().numpy()); labels.append(batch["label"].numpy()); indices.append(batch["index"].numpy())
            if return_embeddings:
                embeddings.append(aux["embedding"].cpu().numpy())
    result = {
        "logits": np.concatenate(logits), "labels": np.concatenate(labels).astype(int),
        "indices": np.concatenate(indices).astype(int),
    }
    if return_embeddings:
        result["embeddings"] = np.concatenate(embeddings)
    return result


def selection_auc(model, loader, device) -> float:
    from sklearn.metrics import roc_auc_score

    output = predict_loader(model, loader, device)
    if len(np.unique(output["labels"])) < 2:
        return float("nan")
    probabilities = 1 / (1 + np.exp(-np.clip(output["logits"], -40, 40)))
    return float(roc_auc_score(output["labels"], probabilities))


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def load_checkpoint(path: Path, cfg: Config, device, optimizer=None, scheduler=None, scaler=None):
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("architecture") != cfg.ARCHITECTURE:
        raise RuntimeError(f"Checkpoint architecture mismatch: {payload.get('architecture')}")
    model = instantiate_model(cfg).to(device); model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload: optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload: scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and "scaler_state" in payload: scaler.load_state_dict(payload["scaler_state"])
    return model, payload


def train_fold_model(model, train_loader, selection_loader, cfg: Config, device, best_path: Path, latest_path: Path, history_path: Path, fold: int):
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.MAX_EPOCHS - cfg.WARMUP_EPOCHS), eta_min=cfg.MIN_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.AMP and device.type == "cuda")
    start_epoch, best_auc, stale, history = 0, -np.inf, 0, []
    if latest_path.exists() and not cfg.FORCE_RETRAIN:
        model, payload = load_checkpoint(latest_path, cfg, device, optimizer, scheduler, scaler)
        start_epoch = int(payload["epoch"]) + 1; best_auc = float(payload["best_auc"]); stale = int(payload["stale"])
        history = list(payload.get("history", []))
        print(f"Resuming fold {fold + 1} at epoch {start_epoch + 1}")
    for epoch in range(start_epoch, cfg.MAX_EPOCHS):
        model.train(); totals = []
        if epoch < cfg.WARMUP_EPOCHS:
            warm_lr = cfg.LR * (epoch + 1) / cfg.WARMUP_EPOCHS
            for group in optimizer.param_groups: group["lr"] = warm_lr
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            weights = batch["cluster_weight"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.AMP and device.type == "cuda"):
                logits, aux = model(images, return_aux=True)
                loss, components = training_loss(logits, aux, labels, weights, cfg)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); totals.append(float(loss.detach()))
        if epoch >= cfg.WARMUP_EPOCHS: scheduler.step()
        auc = selection_auc(model, selection_loader, device)
        row = {"epoch": epoch + 1, "loss_total": float(np.mean(totals)), "selection_auc": auc, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"last_{key}": value for key, value in components.items()}); history.append(row)
        improved = np.isfinite(auc) and auc > best_auc + 1e-5
        if improved:
            best_auc, stale = auc, 0
        else:
            stale += 1
        payload = {
            "architecture": cfg.ARCHITECTURE, "config": asdict(cfg), "fold": fold + 1,
            "epoch": epoch, "best_auc": best_auc, "stale": stale, "history": history,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
        }
        save_checkpoint(latest_path, payload)
        if improved: save_checkpoint(best_path, payload)
        write_csv(pd.DataFrame(history), history_path)
        print(f"Fold {fold + 1} | epoch {epoch + 1:02d} | loss {row['loss_total']:.4f} | selection AUC {auc:.4f} | best {best_auc:.4f}")
        if stale >= cfg.PATIENCE:
            print(f"Fold {fold + 1}: early stopping")
            break
    model, _ = load_checkpoint(best_path, cfg, device)
    latest_path.unlink(missing_ok=True)
    return model, pd.DataFrame(history)


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=float), -40, 40)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=float); labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Temperature scaling requires both classes")
    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature)); z = logits / temperature
        return float(np.mean(np.logaddexp(0, z) - labels * z))
    result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    return float(math.exp(result.x))


def select_threshold(labels: np.ndarray, probabilities: np.ndarray, target: float) -> tuple[float, dict[str, Any]]:
    from sklearn.metrics import confusion_matrix

    candidates = np.unique(np.r_[0.0, probabilities, np.nextafter(probabilities, np.inf), 1.0])
    feasible, fallback = [], []
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1)
        balanced = 0.5 * (sensitivity + specificity)
        fallback.append((balanced, sensitivity, specificity, float(threshold)))
        if sensitivity >= target:
            feasible.append((specificity, sensitivity, float(threshold)))
    if feasible:
        specificity, sensitivity, threshold = max(feasible, key=lambda x: (x[0], x[1], x[2]))
        return threshold, {"rule": "max_specificity_subject_to_sensitivity", "target": target, "sensitivity": sensitivity, "specificity": specificity}
    balanced, sensitivity, specificity, threshold = max(fallback, key=lambda x: (x[0], x[1], x[2], x[3]))
    return threshold, {"rule": "fallback_max_balanced_accuracy", "target": target, "balanced_accuracy": balanced, "sensitivity": sensitivity, "specificity": specificity}


def calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> tuple[float, float]:
    labels = np.asarray(labels); probabilities = np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1); ece, mce = 0.0, 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if not mask.any():
            continue
        gap = abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
        ece += gap * float(mask.mean()); mce = max(mce, gap)
    return float(ece), float(mce)


def metric_dict(labels: Iterable[int], probabilities: Iterable[float], predictions: Iterable[int], threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
        cohen_kappa_score, confusion_matrix, fbeta_score, log_loss, matthews_corrcoef,
        precision_score, roc_auc_score,
    )

    y = np.asarray(labels, dtype=int); p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    pred = np.asarray(predictions, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1); accuracy = accuracy_score(y, pred); ece, mce = calibration_error(y, p)
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold), "n": len(y), "accuracy": float(accuracy), "error_rate": float(1 - accuracy),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_ppv": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(sensitivity), "specificity": float(specificity), "npv": float(npv),
        "f1": float(fbeta_score(y, pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)), "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.c_[1 - p, p], labels=[0, 1])),
        "ece": ece, "mce": mce, "fpr": float(fp / max(fp + tn, 1)), "fnr": float(fn / max(fn + tp, 1)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def threshold_confidence(probability: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-7, 1 - 1e-7); threshold = np.clip(threshold, 1e-7, 1 - 1e-7)
    distance = np.abs(np.log(probability / (1 - probability)) - np.log(threshold / (1 - threshold)))
    return 1 - np.exp(-distance)


def prediction_frame(test: pd.DataFrame, output: dict[str, np.ndarray], temperature: float, threshold: float, fold: int) -> pd.DataFrame:
    frame = test.reset_index(drop=True).iloc[output["indices"]].copy().reset_index(drop=True)
    if not np.array_equal(frame.label.to_numpy(), output["labels"]):
        raise AssertionError("Prediction order does not match locked outer-test manifest")
    frame["outer_fold"] = fold; frame["logit_raw"] = output["logits"]
    frame["prob_raw"] = sigmoid_np(output["logits"]); frame["temperature"] = temperature
    frame["prob_calibrated"] = sigmoid_np(output["logits"] / temperature); frame["threshold"] = threshold
    frame["pred_raw_0_5"] = (frame.prob_raw >= 0.5).astype(int)
    frame["pred_calibrated"] = (frame.prob_calibrated >= threshold).astype(int)
    frame["decision_confidence"] = threshold_confidence(frame.prob_calibrated.to_numpy(), np.full(len(frame), threshold))
    p = np.clip(frame.prob_calibrated.to_numpy(), 1e-8, 1 - 1e-8)
    frame["predictive_entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    frame["correct"] = (frame.pred_calibrated == frame.label).astype(int)
    return frame


def aggregate_metrics(predictions: pd.DataFrame, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fold_rows = [], []
    rows.append({"state": "raw_0_5", **metric_dict(predictions.label, predictions.prob_raw, predictions.pred_raw_0_5, 0.5)})
    rows.append({"state": "fold_calibrated", **metric_dict(predictions.label, predictions.prob_calibrated, predictions.pred_calibrated, float("nan"))})
    for fold, frame in predictions.groupby("outer_fold"):
        fold_rows.append({"fold": int(fold) + 1, **metric_dict(frame.label, frame.prob_calibrated, frame.pred_calibrated, float(frame.threshold.iloc[0]))})
    metrics, folds = pd.DataFrame(rows), pd.DataFrame(fold_rows)
    write_csv(metrics, dirs["tables"] / "aggregate_metrics.csv"); write_csv(folds, dirs["tables"] / "fold_metrics.csv")
    return metrics, folds


def group_bootstrap(predictions: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.SEED + 991); groups = predictions.group_id.unique()
    names = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc", "roc_auc", "pr_auc", "brier_score", "ece"]
    observed = metric_dict(predictions.label, predictions.prob_calibrated, predictions.pred_calibrated, float("nan"))
    values = {name: [] for name in names}
    for _ in range(cfg.BOOTSTRAP_REPS):
        sampled = rng.choice(groups, len(groups), replace=True)
        boot = pd.concat([predictions[predictions.group_id == group] for group in sampled], ignore_index=True)
        if boot.label.nunique() < 2:
            continue
        result = metric_dict(boot.label, boot.prob_calibrated, boot.pred_calibrated, float("nan"))
        for name in names: values[name].append(result[name])
    rows = []
    for name in names:
        array = np.asarray(values[name], dtype=float)
        rows.append({"metric": name, "estimate": observed[name], "ci_low": float(np.nanpercentile(array, 2.5)), "ci_high": float(np.nanpercentile(array, 97.5)), "valid_repetitions": int(np.isfinite(array).sum()), "resampling_unit": "duplicate_group"})
    output = pd.DataFrame(rows); write_csv(output, dirs["tables"] / "group_bootstrap_95ci.csv")
    return output


def display_df(frame: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    try:
        from IPython.display import display
        display(frame)
    except Exception:
        print(frame.to_string(index=False))


def display_image(path: Path) -> None:
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(filename=str(path)))
    except Exception:
        print(path)


def savefig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches_inches="tight")


def fold_figures(history: pd.DataFrame, frame: pd.DataFrame, fold: int, dirs: dict[str, Path]) -> list[Path]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    paths = []
    fig, left = plt.subplots(figsize=(7, 4))
    left.plot(history.epoch, history.loss_total, marker="o", label="Training loss")
    right = left.twinx(); right.plot(history.epoch, history.selection_auc, marker="s", label="Selection AUC")
    left.set(xlabel="Epoch", ylabel="Loss", title=f"Fold {fold + 1}: persisted learning curve")
    right.set_ylabel("Selection ROC-AUC")
    base = dirs["figures"] / f"fold_{fold + 1}_learning_curve"; savefig(fig, base); plt.close(fig); paths.append(base.with_suffix(".png"))

    cm = confusion_matrix(frame.label, frame.pred_calibrated, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4)); image = ax.imshow(cm)
    for i in range(2):
        for j in range(2): ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    ax.set_xticks([0, 1], ["Normal", "DFU"]); ax.set_yticks([0, 1], ["Normal", "DFU"])
    ax.set(xlabel="Predicted", ylabel="True", title=f"Fold {fold + 1}: calibrated confusion")
    fig.colorbar(image, ax=ax)
    base = dirs["figures"] / f"fold_{fold + 1}_confusion"; savefig(fig, base); plt.close(fig); paths.append(base.with_suffix(".png"))
    return paths


def risk_coverage(predictions: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []; ordered = predictions.sort_values("decision_confidence", ascending=False).reset_index(drop=True)
    for coverage in np.linspace(0.10, 1.0, 19):
        selected = ordered.iloc[:max(1, int(len(ordered) * coverage))]
        result = metric_dict(selected.label, selected.prob_calibrated, selected.pred_calibrated, float("nan"))
        rows.append({"coverage": coverage, "risk": result["error_rate"], "selective_accuracy": result["accuracy"], "sensitivity": result["sensitivity"], "specificity": result["specificity"], "false_negatives": result["fn"]})
    output = pd.DataFrame(rows); write_csv(output, dirs["tables"] / "risk_coverage.csv")
    return output


def reliability_points(labels: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    rows = []
    for low, high in zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if mask.any():
            rows.append({"mean_probability": probabilities[mask].mean(), "observed_frequency": labels[mask].mean(), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def final_figures(manifest: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame, folds: pd.DataFrame, dirs: dict[str, Path]) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    fig, ax = plt.subplots(figsize=(6, 4)); manifest.label_name.value_counts().reindex(["Normal", "DFU"]).plot(kind="bar", ax=ax)
    ax.set(title="Strict audited class distribution", ylabel="Images"); savefig(fig, dirs["figures"] / "class_distribution"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["balanced_accuracy", "sensitivity", "specificity"]:
        ax.plot(folds.fold, folds[name], marker="o", label=name)
    ax.set_xticks(range(1, 6)); ax.set_ylim(0, 1.02); ax.set(xlabel="Outer fold", ylabel="Score", title="Five-fold stability (not multi-seed)")
    ax.legend(); savefig(fig, dirs["figures"] / "fold_stability"); plt.close(fig)

    fpr, tpr, _ = roc_curve(predictions.label, predictions.prob_calibrated)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC={metrics.loc[metrics.state=='fold_calibrated','roc_auc'].iloc[0]:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--"); ax.set(xlabel="False-positive rate", ylabel="Sensitivity", title="OOF ROC curve"); ax.legend()
    savefig(fig, dirs["figures"] / "roc_curve"); plt.close(fig)

    precision, recall, _ = precision_recall_curve(predictions.label, predictions.prob_calibrated)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(recall, precision, label=f"AP={metrics.loc[metrics.state=='fold_calibrated','pr_auc'].iloc[0]:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="OOF precision-recall curve"); ax.legend(); savefig(fig, dirs["figures"] / "precision_recall_curve"); plt.close(fig)

    for normalized in [False, True]:
        cm = confusion_matrix(predictions.label, predictions.pred_calibrated, labels=[0, 1], normalize="true" if normalized else None)
        fig, ax = plt.subplots(figsize=(4.5, 4)); image = ax.imshow(cm)
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{cm[i, j]:.3f}" if normalized else str(int(cm[i, j])), ha="center", va="center")
        ax.set_xticks([0, 1], ["Normal", "DFU"]); ax.set_yticks([0, 1], ["Normal", "DFU"])
        ax.set(xlabel="Predicted", ylabel="True", title="OOF confusion matrix"); fig.colorbar(image, ax=ax)
        savefig(fig, dirs["figures"] / ("confusion_normalized" if normalized else "confusion_raw")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, column in [("Raw", "prob_raw"), ("Calibrated", "prob_calibrated")]:
        points = reliability_points(predictions.label, predictions[column]); ax.plot(points.mean_probability, points.observed_frequency, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--"); ax.set(xlabel="Mean DFU probability", ylabel="Observed DFU frequency", title="Reliability diagram"); ax.legend()
    savefig(fig, dirs["figures"] / "reliability_diagram"); plt.close(fig)

    risk = risk_coverage(predictions, dirs)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(risk.coverage, risk.risk, marker="o")
    ax.set(xlabel="Coverage", ylabel="Error rate", title="Threshold-aware risk-coverage"); savefig(fig, dirs["figures"] / "risk_coverage"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(risk.coverage, risk.sensitivity, marker="o")
    ax.set_ylim(0, 1.02); ax.set(xlabel="Coverage", ylabel="Sensitivity", title="Sensitivity-coverage"); savefig(fig, dirs["figures"] / "sensitivity_coverage"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(predictions.decision_confidence, bins=25)
    ax.set(xlabel="Threshold-aware decision confidence", ylabel="Images", title="Decision-confidence distribution")
    savefig(fig, dirs["figures"] / "decision_confidence_histogram"); plt.close(fig)


def select_xai_cases(predictions: pd.DataFrame, limit: int) -> pd.DataFrame:
    frame = predictions.copy()
    frame["case"] = np.select([
        (frame.label == 1) & (frame.pred_calibrated == 1),
        (frame.label == 0) & (frame.pred_calibrated == 0),
        (frame.label == 0) & (frame.pred_calibrated == 1),
        (frame.label == 1) & (frame.pred_calibrated == 0),
    ], ["true_positive", "true_negative", "false_positive", "false_negative"], default="other")
    chosen = []
    for case in ["false_negative", "false_positive", "true_positive", "true_negative"]:
        subset = frame[frame.case == case]
        if not subset.empty:
            chosen.append(subset.sort_values("decision_confidence", ascending=False).iloc[0])
    return pd.DataFrame(chosen).drop_duplicates("image_id").head(limit) if chosen else frame.head(0)


def normalize_map(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float); array -= np.nanmin(array)
    return array / max(float(np.nanmax(array)), 1e-8)


def run_small_xai(predictions: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    _, transform = build_transforms(cfg); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for _, row in select_xai_cases(predictions, cfg.XAI_CASES).iterrows():
        fold = int(row.outer_fold); checkpoint = dirs["models"] / f"dfu_polarmorphnet_fold_{fold + 1}.pt"
        model, _ = load_checkpoint(checkpoint, cfg, device); model.eval(); model.zero_grad(set_to_none=True)
        with Image.open(row.image_path) as image:
            original = image.convert("RGB").resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        x = transform(original).unsqueeze(0).to(device)
        logits, aux = model(x, return_aux=True); logits[0].backward()
        feature, gradient = model.last_feature, model.last_feature.grad
        if gradient is None:
            raise RuntimeError("Grad-CAM gradient unavailable")
        weights = gradient.mean((2, 3), keepdim=True)
        cam = torch.relu((weights * feature).sum(1, keepdim=True))
        cam = F.interpolate(cam, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), mode="bilinear", align_corners=False)
        lesion = F.interpolate(aux["mask"], (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam_np = normalize_map(cam[0, 0].detach().cpu().numpy()); lesion_np = normalize_map(lesion[0, 0].detach().cpu().numpy())
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(original); axes[0].set_title("Original")
        axes[1].imshow(original); axes[1].imshow(lesion_np, alpha=0.45); axes[1].set_title("Weak lesion map")
        axes[2].imshow(original); axes[2].imshow(cam_np, alpha=0.45); axes[2].set_title("Held-out-fold Grad-CAM")
        for axis in axes: axis.axis("off")
        fig.suptitle(f"{row.case} | true={row.label_name} | p={row.prob_calibrated:.3f} | fold={fold + 1}")
        base = dirs["xai"] / f"{row.image_id}_{row.case}_xai"; savefig(fig, base); plt.close(fig)
        records.append({
            "image_id": row.image_id, "case": row.case, "fold": fold + 1,
            "true_label": int(row.label), "prediction": int(row.pred_calibrated),
            "probability": float(row.prob_calibrated), "checkpoint": str(checkpoint),
            "figure_png": str(base.with_suffix(".png")), "methods": "weak_lesion_map;gradcam",
            "causality_claim": False, "clinical_localization_claim": False,
        })
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    output = pd.DataFrame(records); write_csv(output, dirs["tables"] / "xai_metadata.csv")
    return output


def corrupt_image(image: Image.Image, condition: str, seed: int) -> Image.Image:
    if condition == "clean": return image
    if condition == "blur_sigma_2": return image.filter(ImageFilter.GaussianBlur(2.0))
    if condition == "brightness_0_7": return ImageEnhance.Brightness(image).enhance(0.7)
    if condition == "jpeg_q30":
        buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=30); buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if condition == "gaussian_noise_0_05":
        array = np.asarray(image, dtype=np.float32) / 255.0
        rng = np.random.default_rng(seed); array = np.clip(array + rng.normal(0, 0.05, array.shape), 0, 1)
        return Image.fromarray(np.uint8(array * 255))
    if condition == "occlusion_20pct":
        array = np.asarray(image).copy(); h, w = array.shape[:2]
        side = int(math.sqrt(0.20) * min(h, w)); y, x = (h - side) // 2, (w - side) // 2
        array[y:y + side, x:x + side] = 0; return Image.fromarray(array)
    raise KeyError(condition)


def run_small_robustness(manifest: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    import torch

    _, transform = build_transforms(cfg); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = ["clean", "gaussian_noise_0_05", "blur_sigma_2", "brightness_0_7", "jpeg_q30", "occlusion_20pct"]
    rows = []
    for fold in range(cfg.N_FOLDS):
        subset = manifest[manifest.outer_fold == fold].copy()
        if cfg.ROBUSTNESS_MAX_PER_FOLD > 0 and len(subset) > cfg.ROBUSTNESS_MAX_PER_FOLD:
            subset = subset.sample(cfg.ROBUSTNESS_MAX_PER_FOLD, random_state=cfg.SEED + fold)
        checkpoint = dirs["models"] / f"dfu_polarmorphnet_fold_{fold + 1}.pt"
        calibration = json.loads((dirs["configs"] / f"calibration_fold_{fold + 1}.json").read_text())
        model, _ = load_checkpoint(checkpoint, cfg, device); model.eval()
        for condition in conditions:
            logits, labels = [], []
            with torch.inference_mode():
                for start in range(0, len(subset), cfg.BATCH_SIZE):
                    tensors, batch_labels = [], []
                    for _, row in subset.iloc[start:start + cfg.BATCH_SIZE].iterrows():
                        with Image.open(row.image_path) as image:
                            changed = corrupt_image(image.convert("RGB"), condition, cfg.SEED + int(row.name))
                        tensors.append(transform(changed)); batch_labels.append(int(row.label))
                    batch = torch.stack(tensors).to(device); logits.append(model(batch).cpu().numpy()); labels.extend(batch_labels)
            logits_np = np.concatenate(logits); labels_np = np.asarray(labels)
            probability = sigmoid_np(logits_np / float(calibration["temperature"])); threshold = float(calibration["threshold"])
            result = metric_dict(labels_np, probability, (probability >= threshold).astype(int), threshold)
            rows.append({"fold": fold + 1, "condition": condition, **result})
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    output = pd.DataFrame(rows); write_csv(output, dirs["tables"] / "small_robustness_results.csv")
    summary = output.groupby("condition")[["balanced_accuracy", "sensitivity", "specificity", "ece"]].mean().reset_index()
    write_csv(summary, dirs["tables"] / "small_robustness_summary.csv")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4)); summary.set_index("condition")[["balanced_accuracy", "sensitivity"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Mean across folds"); ax.set_title("Small fixed-corruption robustness"); ax.tick_params(axis="x", rotation=25)
    savefig(fig, dirs["figures"] / "small_robustness"); plt.close(fig)
    return output


def lease_path(dirs: dict[str, Path], fold: int) -> Path:
    return dirs["locks"] / f"fold_{fold + 1}.lock.json"


def claim_fold(dirs: dict[str, Path], fold: int, cfg: Config) -> bool:
    path = lease_path(dirs, fold)
    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > cfg.FOLD_LEASE_HOURS:
            path.unlink(missing_ok=True)
        else:
            return False
    payload = {"fold": fold + 1, "claimed_at": utcnow(), "pid": os.getpid(), "host": platform.node()}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle: json.dump(payload, handle, indent=2)
        return True
    except FileExistsError:
        return False


def release_fold(dirs: dict[str, Path], fold: int) -> None:
    lease_path(dirs, fold).unlink(missing_ok=True)


def fold_complete(dirs: dict[str, Path], fold: int) -> bool:
    return all([
        (dirs["models"] / f"dfu_polarmorphnet_fold_{fold + 1}.pt").exists(),
        (dirs["predictions"] / f"oof_fold_{fold + 1}.csv").exists(),
        (dirs["configs"] / f"calibration_fold_{fold + 1}.json").exists(),
    ])


def run_fold(fold: int, manifest: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    import torch

    train_transform, eval_transform = build_transforms(cfg)
    outer_train = manifest[manifest.outer_fold != fold].copy()
    outer_test = manifest[manifest.outer_fold == fold].copy().reset_index(drop=True)
    inner = make_inner_partition(outer_train, fold, cfg, dirs)
    train_frame = inner[inner.inner_role == "train"].copy()
    selection_frame = inner[inner.inner_role == "selection"].copy()
    calibration_frame = inner[inner.inner_role == "calibration"].copy()
    train_loader = build_loader(train_frame, train_transform, cfg, True, cfg.SEED + fold)
    selection_loader = build_loader(selection_frame, eval_transform, cfg, False, cfg.SEED + 1000 + fold)
    calibration_loader = build_loader(calibration_frame, eval_transform, cfg, False, cfg.SEED + 2000 + fold)
    test_loader = build_loader(outer_test, eval_transform, cfg, False, cfg.SEED + 3000 + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = dirs["models"] / f"dfu_polarmorphnet_fold_{fold + 1}.pt"
    latest_path = dirs["models"] / f"dfu_polarmorphnet_fold_{fold + 1}_latest.pt"
    history_path = dirs["logs"] / f"history_fold_{fold + 1}.csv"
    prediction_path = dirs["predictions"] / f"oof_fold_{fold + 1}.csv"
    if cfg.FORCE_RETRAIN:
        for path in [best_path, latest_path, prediction_path, dirs["configs"] / f"calibration_fold_{fold + 1}.json"]:
            path.unlink(missing_ok=True)
    trained_now = not best_path.exists() or latest_path.exists()
    model = instantiate_model(cfg).to(device)
    if trained_now:
        model, history = train_fold_model(model, train_loader, selection_loader, cfg, device, best_path, latest_path, history_path, fold)
    else:
        model, _ = load_checkpoint(best_path, cfg, device)
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        print(f"Fold {fold + 1}: reusing completed checkpoint; no training")

    calibration_output = predict_loader(model, calibration_loader, device)
    temperature = fit_temperature(calibration_output["logits"], calibration_output["labels"])
    calibration_probability = sigmoid_np(calibration_output["logits"] / temperature)
    threshold, threshold_info = select_threshold(calibration_output["labels"], calibration_probability, cfg.TARGET_SENSITIVITY)
    test_output = predict_loader(model, test_loader, device, return_embeddings=True)
    frame = prediction_frame(outer_test, test_output, temperature, threshold, fold)
    write_csv(frame, prediction_path)
    np.save(dirs["predictions"] / f"oof_embeddings_fold_{fold + 1}.npy", test_output["embeddings"])
    calibration_info = {
        "fold": fold + 1, "temperature": temperature, "threshold": threshold,
        "threshold_selection": threshold_info, "n_train": len(train_frame),
        "n_selection": len(selection_frame), "n_calibration": len(calibration_frame),
        "n_outer_test": len(outer_test), "checkpoint": str(best_path),
        "checkpoint_sha256": sha256_file(best_path), "parameter_count": parameter_count(model),
        "trained_now": trained_now, "architecture": cfg.ARCHITECTURE,
    }
    write_json(dirs["configs"] / f"calibration_fold_{fold + 1}.json", calibration_info)
    result = metric_dict(frame.label, frame.prob_calibrated, frame.pred_calibrated, threshold)
    live = pd.DataFrame([{**result, "fold": fold + 1, "temperature": temperature, "trained_now": trained_now}])
    display_df(live, f"Fold {fold + 1} completed result")
    if not history.empty:
        for path in fold_figures(history, frame, fold, dirs): display_image(path)
    registry_path = dirs["logs"] / "primary_fit_registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry = [item for item in registry if item.get("fold") != fold + 1] + [calibration_info]
    write_json(registry_path, sorted(registry, key=lambda item: item["fold"]))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return frame


def gather_oof(cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    paths = [dirs["predictions"] / f"oof_fold_{fold + 1}.csv" for fold in range(cfg.N_FOLDS)]
    missing = [path.name for path in paths if not path.exists()]
    if missing: raise FileNotFoundError(f"Incomplete folds: {missing}")
    output = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if output.image_id.duplicated().any(): raise AssertionError("Duplicate OOF image IDs")
    write_csv(output, dirs["predictions"] / "dfu_polarmorphnet_oof_predictions.csv")
    return output


def model_registry(cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for fold in range(cfg.N_FOLDS):
        path = dirs["models"] / f"dfu_polarmorphnet_fold_{fold + 1}.pt"
        if path.exists():
            rows.append({"fold": fold + 1, "checkpoint": str(path), "sha256": sha256_file(path), "size_mb": path.stat().st_size / 1024**2, "architecture": cfg.ARCHITECTURE})
    output = pd.DataFrame(rows); write_csv(output, dirs["tables"] / "model_registry.csv")
    return output


# Redefinition intentionally keeps the assembled source safe if an older fragment contained a typo.
def savefig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def software_versions() -> dict[str, Any]:
    packages = ["torch", "torchvision", "timm", "numpy", "pandas", "sklearn", "scipy", "matplotlib", "PIL", "imagehash", "kagglehub"]
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in packages:
        try:
            module = __import__(name); versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc!r}"
    try:
        import torch
        versions.update({"cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})
    except Exception:
        pass
    return versions


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git_export" not in path.parts:
            rows.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def write_documents(cfg: Config, dirs: dict[str, Path], metrics: pd.DataFrame) -> None:
    calibrated = metrics[metrics.state == "fold_calibrated"].iloc[0]
    model_card = f"""# DFU-PolarMorphNet V2 Model Card

## Architecture

{cfg.ARCHITECTURE}: morphology-preserving stem, weak lesion discovery, differentiable lesion-centred polar tokenization, bidirectional radial selective state-space encoding, cyclic contour mixing, adaptive radial-contour-global fusion and counterfactual background suppression.

## Evaluation

Duplicate/source-group-disjoint nested five-fold OOF evaluation. Inner partitions control early stopping, fold-specific temperature scaling and threshold selection. Outer-test folds are untouched until inference.

## Internal calibrated OOF result

- Accuracy: {calibrated.accuracy:.4f}
- Balanced accuracy: {calibrated.balanced_accuracy:.4f}
- Sensitivity: {calibrated.sensitivity:.4f}
- Specificity: {calibrated.specificity:.4f}
- ROC-AUC: {calibrated.roc_auc:.4f}
- PR-AUC: {calibrated.pr_auc:.4f}
- Brier: {calibrated.brier_score:.4f}
- ECE: {calibrated.ece:.4f}

## Intended use

Retrospective research and reproducibility analysis only. Not a medical device and not clinically deployment-ready.
"""
    limitations = """# Limitations

1. Patient/case identifiers are unavailable; splitting is duplicate-group-aware, not patient-level.
2. The source is a small public retrospective patch dataset.
3. Weak lesion maps are architectural attention maps, not segmentation ground truth.
4. Five-fold variation is not multi-seed stability.
5. Genuine compatible external validation is not part of this primary run.
6. Synthetic corruption tests do not reproduce real acquisition shift.
7. XAI does not prove causality, localization accuracy or clinical correctness.
8. Dataset licensing is unknown/undeclared and must be clarified before redistribution.
9. High-impact claims require five seeds, complete ablation, frozen external validation and clinical/reviewer error analysis.
"""
    (dirs["root"] / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    (dirs["root"] / "LIMITATIONS.md").write_text(limitations, encoding="utf-8")


def create_reproducibility_bundle(cfg: Config, dirs: dict[str, Path], manifest: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame, confidence_intervals: pd.DataFrame, xai: pd.DataFrame, robustness: pd.DataFrame) -> Path:
    calibrations = [json.loads((dirs["configs"] / f"calibration_fold_{fold + 1}.json").read_text()) for fold in range(cfg.N_FOLDS)]
    payload = {
        "architecture": cfg.ARCHITECTURE, "configuration": asdict(cfg),
        "dataset_metadata": json.loads((dirs["root"] / "dataset_manifest.json").read_text()),
        "split_integrity": json.loads((dirs["root"] / "split_integrity_report.json").read_text()),
        "fold_assignments": manifest[["image_id", "relative_path", "label", "group_id", "group_size", "outer_fold"]].to_dict("records"),
        "oof_predictions": predictions.to_dict("records"), "metrics": metrics.to_dict("records"),
        "confidence_intervals": confidence_intervals.to_dict("records"), "calibrations": calibrations,
        "model_registry": model_registry(cfg, dirs).to_dict("records"),
        "xai_metadata": xai.to_dict("records") if not xai.empty else [],
        "robustness": robustness.to_dict("records") if not robustness.empty else [],
        "software_versions": software_versions(), "raw_images_stored": False,
        "warnings": ["trusted-local pickle only", "retrospective research; no deployment claim", "external validation required"],
    }
    path = dirs["root"] / "dfu_polarmorphnet_complete_reproducibility.pkl"
    with path.open("wb") as handle: pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(dirs["root"] / "reproducibility_manifest.json", {"path": str(path), "sha256": sha256_file(path), "raw_images_stored": False})
    return path


def get_github_token() -> Optional[str]:
    token = os.getenv("GITHUB_TOKEN")
    if token: return token.strip()
    try:
        from google.colab import userdata
        value = userdata.get("GITHUB_TOKEN")
        return value.strip() if value else None
    except Exception:
        return None


def copy_github_safe(dirs: dict[str, Path], cfg: Config, destination: Path) -> list[dict[str, Any]]:
    if destination.exists(): shutil.rmtree(destination)
    destination.mkdir(parents=True)
    skipped = []; limit = cfg.GITHUB_MAX_FILE_MB * 1024**2
    allowed = {"tables", "figures", "xai", "configs", "manifests", "logs", "predictions", "robustness"}
    for path in dirs["root"].rglob("*"):
        if not path.is_file(): continue
        relative = path.relative_to(dirs["root"])
        if relative.parts[0] == "models" or path.suffix == ".npy":
            skipped.append({"path": str(path), "sha256": sha256_file(path), "reason": "drive_only_model_or_embedding"}); continue
        if relative.parts[0] not in allowed and len(relative.parts) > 1: continue
        if path.stat().st_size > limit:
            skipped.append({"path": str(path), "sha256": sha256_file(path), "reason": "github_size_limit"}); continue
        target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    write_json(destination / "drive_only_or_oversized_artifacts.json", skipped)
    return skipped


def push_results_to_github(dirs: dict[str, Path], cfg: Config) -> dict[str, Any]:
    status = {"attempted": False, "success": False, "branch": None, "commit_sha": None, "error": None}
    if not cfg.GITHUB_PUSH:
        status["error"] = "disabled"; write_json(dirs["root"] / "github_push_status.json", status); return status
    token = get_github_token()
    if not token:
        status["error"] = "GITHUB_TOKEN unavailable"; write_json(dirs["root"] / "github_push_status.json", status); return status
    status["attempted"] = True; branch = f"polarmorphnet-results/{cfg.RUN_ID}"; status["branch"] = branch
    work = Path("/content/polarmorphnet_git_export") if in_colab() else dirs["root"] / ".git_export"
    try:
        if work.exists(): shutil.rmtree(work)
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{cfg.REPO}.git", str(work)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "checkout", "-B", branch], cwd=work, check=True, capture_output=True, text=True)
        target = work / "results" / "polarmorphnet" / cfg.RUN_ID
        copy_github_safe(dirs, cfg, target)
        latest = work / "results" / "polarmorphnet" / "LATEST_RUN.txt"; latest.parent.mkdir(parents=True, exist_ok=True); latest.write_text(cfg.RUN_ID + "\n")
        subprocess.run(["git", "config", "user.name", "DFU-PolarMorphNet Colab"], cwd=work, check=True)
        subprocess.run(["git", "config", "user.email", "actions@users.noreply.github.com"], cwd=work, check=True)
        subprocess.run(["git", "add", "--", str(target.relative_to(work)), str(latest.relative_to(work))], cwd=work, check=True)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=work).returncode != 0
        if changed: subprocess.run(["git", "commit", "-m", f"Add PolarMorphNet run {cfg.RUN_ID}"], cwd=work, check=True, capture_output=True, text=True)
        authenticated = f"https://x-access-token:{token}@github.com/{cfg.REPO}.git"
        subprocess.run(["git", "push", authenticated, f"HEAD:{branch}", "--force-with-lease"], cwd=work, check=True, capture_output=True, text=True)
        status["commit_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=work, text=True).strip(); status["success"] = True
    except Exception as exc:
        status["error"] = repr(exc)
    finally:
        write_json(dirs["root"] / "github_push_status.json", status)
    return status


def scientific_config(cfg: Config) -> dict[str, Any]:
    ignored = {"MODE", "RUN_ID", "PREFERRED_FOLDS", "FORCE_RETRAIN", "GITHUB_PUSH"}
    return {key: value for key, value in asdict(cfg).items() if key not in ignored}


def lock_run_config(cfg: Config, dirs: dict[str, Path]) -> None:
    path = dirs["configs"] / "resolved_config.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("scientific_config") != scientific_config(cfg):
            raise RuntimeError("Scientific configuration differs from the locked RUN_ID configuration")
    else:
        write_json(path, {"run_id": cfg.RUN_ID, "scientific_config": scientific_config(cfg), "created_at": utcnow()})


def prepare_manifest(cfg: Config, dirs: dict[str, Path]) -> tuple[Path, pd.DataFrame]:
    dataset_root = download_dataset(cfg, dirs)
    locked = dirs["manifests"] / "locked_manifest.csv"
    if locked.exists():
        return dataset_root, remap_paths(pd.read_csv(locked), dataset_root)
    audit_lock = dirs["locks"] / "data_audit.lock"
    owner = False
    try:
        descriptor = os.open(audit_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w") as handle: handle.write(utcnow())
        owner = True
    except FileExistsError:
        pass
    if not owner:
        for _ in range(120):
            if locked.exists(): return dataset_root, remap_paths(pd.read_csv(locked), dataset_root)
            time.sleep(15)
        raise TimeoutError("Another account did not finish the shared data audit")
    try:
        manifest = build_manifest(dataset_root, cfg, dirs)
        manifest = assign_groups(manifest, cfg, dirs)
        manifest = make_outer_folds(manifest, cfg, dirs)
        write_csv(manifest, locked)
        metadata = {
            "dataset": cfg.DATASET_SLUG, "task": "binary Normal-versus-active-DFU patch classification",
            "n_images": len(manifest), "class_counts": manifest.label_name.value_counts().to_dict(),
            "n_groups": int(manifest.group_id.nunique()), "patient_ids_available": False,
            "folds": cfg.N_FOLDS, "manifest_sha256": sha256_file(locked),
            "license": "unknown_or_undeclared", "created_at": utcnow(),
        }
        write_json(dirs["root"] / "dataset_manifest.json", metadata)
        return dataset_root, manifest
    finally:
        audit_lock.unlink(missing_ok=True)


def finalize_run(manifest: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> dict[str, Any]:
    predictions = gather_oof(cfg, dirs)
    metrics, folds = aggregate_metrics(predictions, dirs)
    intervals = group_bootstrap(predictions, cfg, dirs)
    final_figures(manifest, predictions, metrics, folds, dirs)
    xai = run_small_xai(predictions, cfg, dirs) if cfg.RUN_XAI else pd.DataFrame()
    robustness = run_small_robustness(manifest, cfg, dirs) if cfg.RUN_ROBUSTNESS else pd.DataFrame()
    write_json(dirs["root"] / "software_versions.json", software_versions())
    write_documents(cfg, dirs, metrics)
    pkl_path = create_reproducibility_bundle(cfg, dirs, manifest, predictions, metrics, intervals, xai, robustness)
    calibrated = metrics[metrics.state == "fold_calibrated"].iloc[0].to_dict()
    paper_results = {
        "run_id": cfg.RUN_ID, "architecture": cfg.ARCHITECTURE,
        "evaluation": "nested duplicate-group-aware five-fold OOF",
        "primary_internal_result": calibrated, "confidence_intervals": intervals.to_dict("records"),
        "external_validation_performed": False, "multi_seed_performed": False,
        "clinical_deployment_claim": False, "state_of_the_art_claim": False,
        "q1_high_impact_ready": False,
        "remaining_requirements": ["five-seed experiment", "architectural ablation", "frozen external validation", "clinical/reviewer error and XAI review", "license clarification"],
    }
    write_json(dirs["root"] / "paper_results.json", paper_results)
    write_json(dirs["root"] / "artifact_manifest.json", artifact_manifest(dirs["root"]))
    github = push_results_to_github(dirs, cfg)
    registry = model_registry(cfg, dirs)
    verification = {
        "run_id": cfg.RUN_ID, "architecture": cfg.ARCHITECTURE,
        "dataset_images": len(manifest), "class_counts": manifest.label_name.value_counts().to_dict(),
        "valid_fold_count": int(len(registry)), "duplicate_and_leakage_audit": "PASS",
        "checkpoint_count": int(len(registry)), "pkl_path": str(pkl_path), "pkl_sha256": sha256_file(pkl_path),
        "drive_path": str(dirs["root"]), "github_push": github,
        "figures_recreatable_without_training": True, "completed_at": utcnow(),
    }
    write_json(dirs["root"] / "final_verification.json", verification)
    write_json(dirs["root"] / "artifact_manifest.json", artifact_manifest(dirs["root"]))
    if github.get("success"):
        github = push_results_to_github(dirs, cfg); verification["github_push"] = github
        write_json(dirs["root"] / "final_verification.json", verification)
    display_df(metrics, "Final aggregate OOF metrics")
    display_df(intervals, "Duplicate-group bootstrap 95% confidence intervals")
    for name in ["fold_stability.png", "roc_curve.png", "precision_recall_curve.png", "reliability_diagram.png", "risk_coverage.png", "small_robustness.png"]:
        path = dirs["figures"] / name
        if path.exists(): display_image(path)
    print(json.dumps(verification, indent=2, default=str))
    return verification


def find_latest_run(project: Path) -> Path:
    completed = project / "LAST_COMPLETED_POLARMORPHNET_RUN.txt"
    if completed.exists():
        candidate = project / "runs" / completed.read_text().strip()
        if candidate.exists(): return candidate
    candidates = [path for path in (project / "runs").glob("*") if (path / "predictions" / "dfu_polarmorphnet_oof_predictions.csv").exists()]
    if not candidates: raise FileNotFoundError("No complete PolarMorphNet run found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def train_mode(cfg: Config) -> dict[str, Any]:
    project, dirs = resolve_run(cfg); lock_run_config(cfg, dirs); seed_everything(cfg.SEED)
    dataset_root, manifest = prepare_manifest(cfg, dirs)
    print(f"RUN_ID={cfg.RUN_ID} | Drive={dirs['root']} | preferred folds={[x + 1 for x in cfg.preferred_fold_indices]}")
    for fold in cfg.preferred_fold_indices:
        if fold_complete(dirs, fold) and not cfg.FORCE_RETRAIN:
            print(f"Fold {fold + 1}: already complete; no retraining"); continue
        if not claim_fold(dirs, fold, cfg):
            print(f"Fold {fold + 1}: leased by another Colab account; skipped"); continue
        try:
            run_fold(fold, manifest, cfg, dirs)
        except Exception:
            write_json(dirs["logs"] / f"FAILED_FOLD_{fold + 1}.json", {"fold": fold + 1, "timestamp": utcnow(), "traceback": traceback.format_exc()})
            raise
        finally:
            release_fold(dirs, fold)
    complete = [fold_complete(dirs, fold) for fold in range(cfg.N_FOLDS)]
    progress = {"run_id": cfg.RUN_ID, "completed_folds": [i + 1 for i, value in enumerate(complete) if value], "missing_folds": [i + 1 for i, value in enumerate(complete) if not value], "all_complete": all(complete), "drive_path": str(dirs["root"])}
    write_json(dirs["logs"] / "progress.json", progress); print(json.dumps(progress, indent=2))
    if not all(complete):
        print("Resume with the same RUN_ID and shared DRIVE_ROOT. Allocate only missing folds on other accounts.")
        return progress
    verification = finalize_run(manifest, cfg, dirs)
    (project / "LAST_COMPLETED_POLARMORPHNET_RUN.txt").write_text(cfg.RUN_ID + "\n")
    (project / "ACTIVE_POLARMORPHNET_RUN.txt").unlink(missing_ok=True)
    return verification


def load_saved_scientific_config(cfg: Config, dirs: dict[str, Path]) -> None:
    path = dirs["configs"] / "resolved_config.json"
    if not path.exists(): return
    saved = json.loads(path.read_text()).get("scientific_config", {})
    for key, value in saved.items():
        if hasattr(cfg, key): setattr(cfg, key, value)


def artifact_mode(cfg: Config) -> dict[str, Any]:
    mount_drive(); project = Path(cfg.DRIVE_ROOT)
    run = project / "runs" / cfg.RUN_ID if cfg.RUN_ID else find_latest_run(project)
    if not run.exists(): raise FileNotFoundError(run)
    cfg.RUN_ID = run.name; dirs = make_dirs(run); load_saved_scientific_config(cfg, dirs); cfg.MODE = "artifacts"
    seed_everything(cfg.SEED); dataset_root = download_dataset(cfg, dirs)
    manifest_path = dirs["manifests"] / "locked_manifest.csv"
    if not manifest_path.exists(): raise FileNotFoundError(manifest_path)
    manifest = remap_paths(pd.read_csv(manifest_path), dataset_root)
    print("ARTIFACT-ONLY MODE: no model training will occur.")
    return finalize_run(manifest, cfg, dirs)


def upload_mode(cfg: Config) -> dict[str, Any]:
    mount_drive(); project = Path(cfg.DRIVE_ROOT)
    run = project / "runs" / cfg.RUN_ID if cfg.RUN_ID else find_latest_run(project)
    if not run.exists(): raise FileNotFoundError(run)
    cfg.RUN_ID = run.name; dirs = make_dirs(run); load_saved_scientific_config(cfg, dirs); cfg.MODE = "upload"
    print("UPLOAD MODE: no model training or inference will occur.")
    status = push_results_to_github(dirs, cfg); print(json.dumps(status, indent=2)); return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DFU-PolarMorphNet V2 all-in-one research workflow")
    parser.add_argument("--mode", choices=["train", "artifacts", "upload"], default=None)
    parser.add_argument("--run-id", default=None); parser.add_argument("--drive-root", default=None)
    parser.add_argument("--preferred-folds", default=None); parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=None); parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-xai", action="store_true"); parser.add_argument("--no-robustness", action="store_true")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args(); cfg = Config()
    if args.mode: cfg.MODE = args.mode
    if args.run_id: cfg.RUN_ID = args.run_id
    if args.drive_root: cfg.DRIVE_ROOT = args.drive_root
    if args.preferred_folds is not None: cfg.PREFERRED_FOLDS = args.preferred_folds
    if args.force_retrain: cfg.FORCE_RETRAIN = True
    if args.epochs: cfg.MAX_EPOCHS = args.epochs
    if args.batch_size: cfg.BATCH_SIZE = args.batch_size
    if args.no_xai: cfg.RUN_XAI = False
    if args.no_robustness: cfg.RUN_ROBUSTNESS = False
    print("=" * 92)
    print("DFU-PolarMorphNet V2 | lesion-centred polar state-space architecture")
    print(f"Mode={cfg.MODE} | FORCE_RETRAIN={cfg.FORCE_RETRAIN} | Drive={cfg.DRIVE_ROOT}")
    print("No output is prefilled; saved checkpoints and CSVs are reused whenever available.")
    print("=" * 92)
    if cfg.MODE == "train": return train_mode(cfg)
    if cfg.MODE == "artifacts": return artifact_mode(cfg)
    if cfg.MODE == "upload": return upload_mode(cfg)
    raise ValueError(cfg.MODE)


if __name__ == "__main__":
    main()
