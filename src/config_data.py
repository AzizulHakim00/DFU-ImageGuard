from __future__ import annotations

import dataclasses
import datetime as dt
import gc
import hashlib
import json
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class Config:
    PROJECT_NAME: str = "DFU-ImageGuard"
    REPO_FULL_NAME: str = "AzizulHakim00/DFU-ImageGuard"
    DATASET_HANDLE: str = "laithjj/diabetic-foot-ulcer-dfu"
    DATASET_TITLE: str = "diabetic foot ulcer (DFU)"
    DATASET_LICENSE: str = "Unknown / not declared on the Kaggle data card"
    DATASET_CITATIONS: list[str] = field(default_factory=lambda: [
        "Alzubaidi et al. (2020), DFU_QUTNet, Multimedia Tools and Applications 79, 15655–15677.",
        "Alzubaidi et al. (2020), Towards a better understanding of transfer learning for medical imaging, Applied Sciences 10(13), 4523.",
    ])

    DRIVE_ROOT: str = "/content/drive/MyDrive/DFU-ImageGuard"
    LOCAL_FALLBACK_ROOT: str = "/content/DFU-ImageGuard-local"
    LOCAL_REPO: str = "/content/DFU-ImageGuard"
    RUN_ID: Optional[str] = None
    STORAGE_MODE: str = "unresolved"
    MOUNT_DRIVE: bool = True
    ALLOW_LOCAL_FALLBACK: bool = True

    SEED: int = 2026
    N_FOLDS: int = 5
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 24
    NUM_WORKERS: int = 2
    MAX_EPOCHS: int = 30
    PATIENCE: int = 7
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    DROPOUT: float = 0.30
    GRAD_CLIP_NORM: float = 1.0
    USE_AMP: bool = True
    TARGET_SENSITIVITY: float = 0.95
    BOOTSTRAP_REPS: int = 1000

    PHASH_DISTANCE: int = 4
    EMBEDDING_DUPLICATES: bool = True
    EMBEDDING_COSINE_THRESHOLD: float = 0.999

    FORCE_RETRAIN: bool = False
    RUN_BASELINES: bool = True
    RUN_ROBUSTNESS: bool = True
    RUN_XAI: bool = True
    XAI_CASES: int = 6
    XAI_LIME_SAMPLES: int = 150
    ROBUSTNESS_LEVELS: dict[str, list[float]] = field(default_factory=lambda: {
        "gaussian_noise": [0.03, 0.08],
        "gaussian_blur": [1.0, 2.5],
        "brightness": [0.70, 1.30],
        "contrast": [0.65, 1.35],
        "jpeg": [60, 25],
        "rotation": [7, 15],
        "occlusion": [0.10, 0.25],
    })

    PROPOSED_BACKBONE: str = "convnext_tiny"
    PRIMARY_MODEL_NAME: str = "DFU-ImageGuard"
    BASELINE_MODELS: dict[str, str] = field(default_factory=lambda: {
        "ResNet18": "resnet18",
        "DenseNet121": "densenet121",
        "MobileNetV3": "mobilenetv3_large_100",
        "EfficientNet-B0": "efficientnet_b0",
    })
    LINEAR_BASELINE_NAME: str = "Linear-LogReg"
    GITHUB_MAX_BYTES: int = 95 * 1024 * 1024


def seed_everything(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Deterministically seed Python and NumPy inside each DataLoader worker."""
    import torch

    worker_seed = (int(torch.initial_seed()) + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj):
        return asdict(obj)
    raise TypeError(type(obj))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


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
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _proc_mount_contains(path: str) -> bool:
    try:
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == path:
                return True
    except Exception:
        pass
    return False


def google_drive_is_mounted() -> bool:
    return Path("/content/drive/MyDrive").is_dir() and _proc_mount_contains("/content/drive")


def mount_drive(cfg: Optional[Config] = None) -> bool:
    """Mount Google Drive when possible; otherwise configure a clear local fallback."""
    if cfg is not None and not cfg.MOUNT_DRIVE:
        cfg.DRIVE_ROOT = cfg.LOCAL_FALLBACK_ROOT
        cfg.STORAGE_MODE = "local_fallback_mount_disabled"
        Path(cfg.DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
        print(f"Google Drive mounting disabled; using local storage: {cfg.DRIVE_ROOT}")
        return False

    mounted = google_drive_is_mounted()
    errors: list[str] = []
    if not mounted:
        try:
            from google.colab import drive

            try:
                drive.mount("/content/drive", force_remount=False)
            except Exception as exc:
                errors.append(f"initial mount: {exc}")
                try:
                    drive.flush_and_unmount()
                except Exception:
                    pass
                try:
                    drive.mount("/content/drive", force_remount=True)
                except Exception as retry_exc:
                    errors.append(f"forced remount: {retry_exc}")
            mounted = Path("/content/drive/MyDrive").is_dir()
        except Exception as exc:
            errors.append(f"Google Colab Drive unavailable: {exc}")

    if mounted:
        if cfg is not None:
            cfg.STORAGE_MODE = "google_drive"
        print("Google Drive storage ready: /content/drive/MyDrive")
        return True

    if cfg is not None:
        if not cfg.ALLOW_LOCAL_FALLBACK:
            raise RuntimeError("Google Drive could not be mounted and local fallback is disabled: " + " | ".join(errors))
        cfg.DRIVE_ROOT = cfg.LOCAL_FALLBACK_ROOT
        cfg.STORAGE_MODE = "local_fallback"
        Path(cfg.DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
        print("WARNING: Google Drive mount failed. The run will continue in local runtime storage.")
        print(f"Local fallback: {cfg.DRIVE_ROOT}")
        if errors:
            print("Mount details: " + " | ".join(errors))
    else:
        print("Drive mount unavailable: " + " | ".join(errors))
    return False


def prepare_run_dirs(cfg: Config) -> dict[str, Path]:
    cfg.RUN_ID = cfg.RUN_ID or now_run_id()
    root = Path(cfg.DRIVE_ROOT) / "runs" / cfg.RUN_ID
    dirs: dict[str, Path] = {"root": root}
    for name in ["tables", "figures", "models", "xai", "predictions", "logs", "configs", "manifests", "cache"]:
        dirs[name] = root / name
        dirs[name].mkdir(parents=True, exist_ok=True)
    write_json(dirs["configs"] / "resolved_config.json", asdict(cfg))
    return dirs


def download_dataset(cfg: Config, dirs: dict[str, Path]) -> Path:
    import kagglehub

    path = Path(kagglehub.dataset_download(cfg.DATASET_HANDLE))
    if not path.exists():
        raise FileNotFoundError(path)
    write_json(dirs["manifests"] / "dataset_download.json", {
        "handle": cfg.DATASET_HANDLE,
        "resolved_path": str(path),
        "downloaded_at": dt.datetime.now().isoformat(),
        "license": cfg.DATASET_LICENSE,
        "citations": cfg.DATASET_CITATIONS,
    })
    return path


def normalize_folder_name(name: str) -> str:
    return "".join(c.lower() for c in name if c.isalnum())


def locate_strict_patch_root(dataset_root: Path) -> tuple[Path, dict[str, int]]:
    normal = {"normal", "normalhealthyskin", "healthy", "healthyskin", "normalimages"}
    ulcer = {"abnormal", "abnormalulcer", "ulcer", "dfu", "ulcerimages"}
    candidates: list[tuple[Path, dict[str, int]]] = []
    for p in dataset_root.rglob("*"):
        if not p.is_dir() or normalize_folder_name(p.name) != "patches":
            continue
        mapping: dict[str, int] = {}
        for child in p.iterdir():
            if not child.is_dir():
                continue
            normalized = normalize_folder_name(child.name)
            if normalized in normal:
                mapping[child.name] = 0
            elif normalized in ulcer:
                mapping[child.name] = 1
        if set(mapping.values()) == {0, 1}:
            candidates.append((p, mapping))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one explicit Patches class layout; found {candidates}. No labels are guessed."
        )
    return candidates[0]


def _pixel_hash(img: Image.Image) -> str:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h = hashlib.sha256()
    h.update(np.asarray(arr.shape, dtype=np.int32).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def build_manifest(dataset_root: Path, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    import imagehash

    patch_root, mapping = locate_strict_patch_root(dataset_root)
    rows: list[dict[str, Any]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for folder, label in mapping.items():
        for path in sorted((patch_root / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            row: dict[str, Any] = {
                "image_id": hashlib.sha256(str(path.relative_to(dataset_root)).encode()).hexdigest()[:20],
                "image_path": str(path),
                "relative_path": str(path.relative_to(dataset_root)),
                "source_subset": "Patches",
                "source_class_folder": folder,
                "label": int(label),
                "label_name": "DFU" if label else "Normal",
                "patient_id": None,
                "case_id": None,
                "exclude": False,
                "exclusion_reason": "",
            }
            try:
                file_hash = sha256_file(path)
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    row.update(
                        width=rgb.width,
                        height=rgb.height,
                        file_sha256=file_hash,
                        pixel_sha256=_pixel_hash(rgb),
                        phash=str(imagehash.phash(rgb, hash_size=16)),
                    )
            except Exception as exc:
                row.update(
                    exclude=True,
                    exclusion_reason=f"corrupt_or_unreadable:{type(exc).__name__}",
                    width=None,
                    height=None,
                    file_sha256=None,
                    pixel_sha256=None,
                    phash=None,
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No explicitly labelled Patches images found")
    df.to_csv(dirs["manifests"] / "dfu_master_manifest_initial.csv", index=False)
    write_json(dirs["manifests"] / "strict_label_mapping.json", {
        "patch_root": str(patch_root),
        "class_folder_map": mapping,
        "policy": "Only explicit immediate class folders under Patches are accepted.",
    })
    return df


def _hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def compute_embedding_candidates(
    df: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> list[tuple[int, int, float]]:
    if not cfg.EMBEDDING_DUPLICATES or len(df) < 2:
        pd.DataFrame(columns=["i", "j", "cosine_similarity"]).to_csv(
            dirs["tables"] / "embedding_duplicate_candidates.csv", index=False
        )
        return []

    import torch
    from sklearn.neighbors import NearestNeighbors
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    class EmbeddingDataset(Dataset):
        def __init__(self, paths: list[str]):
            self.paths = paths
            self.tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, i: int):
            with Image.open(self.paths[i]) as im:
                return self.tf(im.convert("RGB")), i

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    generator = torch.Generator().manual_seed(cfg.SEED + 17)
    loader_kwargs: dict[str, Any] = {
        "batch_size": 64,
        "shuffle": False,
        "num_workers": max(0, int(cfg.NUM_WORKERS)),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if cfg.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(EmbeddingDataset(df.image_path.tolist()), **loader_kwargs)

    embeddings = np.zeros((len(df), 512), np.float32)
    with torch.inference_mode():
        for xb, idx in loader:
            z = model(xb.to(device, non_blocking=device.type == "cuda")).cpu().numpy()
            z /= np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
            embeddings[idx.numpy()] = z

    np.save(dirs["cache"] / "duplicate_embeddings.npy", embeddings)
    n_neighbors = min(6, len(df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(df)):
        for distance, j in zip(distances[i, 1:], indices[i, 1:]):
            similarity = 1 - float(distance)
            if (
                j > i
                and similarity >= cfg.EMBEDDING_COSINE_THRESHOLD
                and _hamming_hex(str(df.iloc[i].phash), str(df.iloc[j].phash)) <= max(8, cfg.PHASH_DISTANCE)
            ):
                pairs.append((i, int(j), similarity))

    pd.DataFrame(pairs, columns=["i", "j", "cosine_similarity"]).to_csv(
        dirs["tables"] / "embedding_duplicate_candidates.csv", index=False
    )
    del model, loader, embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pairs


def assign_duplicate_groups(df: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    work = df.loc[~df.exclude].reset_index(drop=True).copy()
    if work.empty:
        raise RuntimeError("All images were excluded during the corruption audit")

    union_find = UnionFind(len(work))
    reasons: list[dict[str, Any]] = []
    for col, reason in [("file_sha256", "exact_file"), ("pixel_sha256", "exact_pixel")]:
        for indices in work.groupby(col, dropna=True).groups.values():
            indices = list(indices)
            for j in indices[1:]:
                union_find.union(indices[0], j)
                reasons.append({"i": indices[0], "j": j, "reason": reason, "score": 1.0})

    phashes = work.phash.astype(str).tolist()
    for i in range(len(work)):
        for j in range(i + 1, len(work)):
            distance = _hamming_hex(phashes[i], phashes[j])
            if distance <= cfg.PHASH_DISTANCE:
                union_find.union(i, j)
                reasons.append({"i": i, "j": j, "reason": "phash", "score": distance})

    for i, j, similarity in compute_embedding_candidates(work, cfg, dirs):
        union_find.union(i, j)
        reasons.append({"i": i, "j": j, "reason": "embedding_plus_phash", "score": similarity})

    roots = [union_find.find(i) for i in range(len(work))]
    names = {root: f"DG{n:05d}" for n, root in enumerate(sorted(set(roots)), 1)}
    work["group_id"] = [names[root] for root in roots]

    clusters = work.groupby("group_id").agg(
        n_images=("image_id", "size"),
        n_labels=("label", "nunique"),
        labels=("label_name", lambda s: "|".join(sorted(set(s)))),
        member_ids=("image_id", lambda s: "|".join(s)),
    ).reset_index()
    conflicts = set(clusters.loc[clusters.n_labels > 1, "group_id"])
    work["label_conflict"] = work.group_id.isin(conflicts)
    work.loc[work.label_conflict, "exclude"] = True
    work.loc[work.label_conflict, "exclusion_reason"] = "duplicate_cluster_label_conflict"

    pd.DataFrame(reasons, columns=["i", "j", "reason", "score"]).to_csv(
        dirs["tables"] / "duplicate_pairs.csv", index=False
    )
    clusters.to_csv(dirs["tables"] / "duplicate_clusters.csv", index=False)
    work.to_csv(dirs["manifests"] / "cleaned_manifest_with_exclusions.csv", index=False)
    pd.concat([df.loc[df.exclude], work.loc[work.exclude]], ignore_index=True, sort=False).to_csv(
        dirs["manifests"] / "excluded_images.csv", index=False
    )

    cleaned = work.loc[~work.exclude].reset_index(drop=True)
    if cleaned.label.nunique() != 2:
        raise RuntimeError("Cleaning removed a class")
    if cleaned.group_id.nunique() < cfg.N_FOLDS:
        raise RuntimeError("Too few duplicate groups for the requested outer folds")
    return cleaned


def _class_group_counts(df: pd.DataFrame) -> dict[int, int]:
    return {
        int(label): int(frame.group_id.nunique())
        for label, frame in df.groupby("label")
    }


def make_outer_folds(df: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    group_counts = _class_group_counts(df)
    if min(group_counts.values()) < cfg.N_FOLDS:
        raise RuntimeError(
            f"Each class needs at least {cfg.N_FOLDS} duplicate groups; found {group_counts}"
        )

    selected: Optional[pd.DataFrame] = None
    for attempt in range(50):
        candidate = df.copy()
        candidate["outer_fold"] = -1
        splitter = StratifiedGroupKFold(
            n_splits=cfg.N_FOLDS,
            shuffle=True,
            random_state=cfg.SEED + attempt,
        )
        for fold, (_, test_idx) in enumerate(
            splitter.split(candidate, candidate.label, groups=candidate.group_id)
        ):
            candidate.loc[test_idx, "outer_fold"] = fold
        if all(candidate.loc[candidate.outer_fold == fold, "label"].nunique() == 2 for fold in range(cfg.N_FOLDS)):
            selected = candidate
            break
    if selected is None:
        raise RuntimeError("Could not create five outer folds containing both classes")

    integrity: dict[str, Any] = {
        "valid": True,
        "patient_ids_available": False,
        "split_unit": "duplicate_group",
        "folds": [],
    }
    for fold in range(cfg.N_FOLDS):
        train = selected[selected.outer_fold != fold]
        test = selected[selected.outer_fold == fold]
        overlap = sorted(set(train.group_id) & set(test.group_id))
        integrity["folds"].append({
            "fold": fold + 1,
            "train_n": len(train),
            "test_n": len(test),
            "train_groups": train.group_id.nunique(),
            "test_groups": test.group_id.nunique(),
            "train_normal": int((train.label == 0).sum()),
            "train_dfu": int((train.label == 1).sum()),
            "test_normal": int((test.label == 0).sum()),
            "test_dfu": int((test.label == 1).sum()),
            "group_overlap": overlap,
        })
        integrity["valid"] = bool(integrity["valid"] and not overlap)
    if not integrity["valid"]:
        raise AssertionError("Duplicate-group leakage")

    selected.to_csv(dirs["manifests"] / "locked_outer_fold_assignments.csv", index=False)
    write_json(dirs["root"] / "split_integrity_report.json", integrity)
    pd.DataFrame(integrity["folds"]).to_csv(dirs["tables"] / "fold_distribution.csv", index=False)
    return selected


def make_inner_partition(outer_train: pd.DataFrame, cfg: Config, fold: int) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    base = outer_train.copy().reset_index(drop=True)
    group_counts = _class_group_counts(base)
    n_splits = min(5, min(group_counts.values()))
    if n_splits < 3:
        raise RuntimeError(
            f"At least three duplicate groups per class are required for train/selection/calibration; found {group_counts}"
        )

    for attempt in range(100):
        x = base.copy()
        x["inner_fold"] = -1
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=cfg.SEED + 100 + fold + attempt,
        )
        for inner_fold, (_, idx) in enumerate(splitter.split(x, x.label, groups=x.group_id)):
            x.loc[idx, "inner_fold"] = inner_fold
        x["inner_role"] = "train"
        x.loc[x.inner_fold == fold % n_splits, "inner_role"] = "selection"
        x.loc[x.inner_fold == (fold + 1) % n_splits, "inner_role"] = "calibration"
        roles = {
            role: set(x.loc[x.inner_role == role, "group_id"])
            for role in ["train", "selection", "calibration"]
        }
        disjoint = (
            not roles["train"] & roles["selection"]
            and not roles["train"] & roles["calibration"]
            and not roles["selection"] & roles["calibration"]
        )
        class_valid = all(
            x.loc[x.inner_role == role, "label"].nunique() == 2
            for role in roles
        )
        if disjoint and class_valid:
            x["inner_n_splits"] = n_splits
            return x
    raise RuntimeError("Could not create class-complete, group-disjoint inner partitions")
