from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .config_data import Config, seed_worker
from .evaluation import sigmoid_np
from .models_training import FrameDataset, predict_logits
from .reliability_io import (
    ReliabilitySettings,
    atomic_csv,
    atomic_json,
    mirror_full_run,
)
from .reliability_metrics import metrics_from_decisions
from .reliability_models import create_classifier

REQUIRED_EXTERNAL_COLUMNS = {"image_path", "label"}


def validate_external_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_EXTERNAL_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"External manifest is missing columns: {sorted(missing)}")
    frame = manifest.copy().reset_index(drop=True)
    frame["image_path"] = frame.image_path.astype(str)
    frame["label"] = frame.label.astype(int)
    if not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError("External labels must be binary 0/1")
    if frame.label.nunique() != 2:
        raise ValueError("External evaluation requires both classes")
    if "image_id" not in frame.columns:
        frame["image_id"] = [f"external_{i:06d}" for i in range(len(frame))]
    if "dataset" not in frame.columns:
        frame["dataset"] = "external"
    if "patient_id" not in frame.columns:
        frame["patient_id"] = ""
    for path in frame.image_path:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        with Image.open(candidate) as image:
            image.verify()
    return frame


def _external_loader(frame: pd.DataFrame, cfg: Config, seed: int):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

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
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        FrameDataset(frame, transform),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_frozen_external_validation(
    *,
    run_root: str | Path,
    external_manifest_csv: str | Path,
    output_name: str = "external_validation",
) -> dict[str, Any]:
    import torch

    run_root = Path(run_root)
    manifest_path = Path(external_manifest_csv)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    settings = ReliabilitySettings(
        **json.loads((run_root / "resolved_settings.json").read_text())
    )
    frame = validate_external_manifest(pd.read_csv(manifest_path))
    cfg = Config()
    cfg.IMAGE_SIZE = settings.image_size
    cfg.BATCH_SIZE = settings.batch_size
    cfg.NUM_WORKERS = settings.num_workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU runtime is required for external validation")

    all_model_predictions: list[pd.DataFrame] = []
    model_metrics: list[dict[str, Any]] = []
    for seed in settings.seeds:
        for fold in range(settings.n_folds):
            trial = (
                run_root
                / "trials"
                / settings.primary_model_key
                / f"seed_{seed}"
                / f"fold_{fold + 1}"
            )
            checkpoint = trial / "best_model.pt"
            metrics_path = trial / "metrics.json"
            if not checkpoint.exists() or not metrics_path.exists():
                raise FileNotFoundError(
                    f"Missing frozen primary trial artifacts: {trial}"
                )
            metadata = json.loads(metrics_path.read_text())
            payload = torch.load(
                checkpoint, map_location=device, weights_only=False
            )
            model = create_classifier(
                settings.primary_model_key,
                pretrained=False,
                dropout=settings.dropout,
                resolved_name=str(payload["resolved_name"]),
            ).to(device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.eval()
            loader = _external_loader(frame, cfg, seed + fold * 1000 + 900)
            logits, y, _ = predict_logits(model, loader, device)
            temperature = float(metadata["temperature"])
            threshold = float(metadata["threshold"])
            probability = sigmoid_np(logits / temperature)
            decision = (probability >= threshold).astype(int)
            prediction = frame[
                ["image_id", "image_path", "dataset", "patient_id", "label"]
            ].copy()
            prediction["seed"] = int(seed)
            prediction["outer_fold_model"] = int(fold + 1)
            prediction["temperature_frozen"] = temperature
            prediction["threshold_frozen"] = threshold
            prediction["prob_calibrated"] = probability
            prediction["pred_frozen"] = decision
            all_model_predictions.append(prediction)
            metrics = metrics_from_decisions(y, probability, decision)
            model_metrics.append(
                {
                    "seed": int(seed),
                    "outer_fold_model": int(fold + 1),
                    **metrics,
                }
            )
            del model
            torch.cuda.empty_cache()

    prediction_table = pd.concat(all_model_predictions, ignore_index=True)
    ensemble = prediction_table.groupby(
        ["image_id", "image_path", "dataset", "patient_id", "label"],
        as_index=False,
    ).agg(
        mean_probability=("prob_calibrated", "mean"),
        frozen_positive_vote_rate=("pred_frozen", "mean"),
        probability_sd=("prob_calibrated", "std"),
    )
    ensemble["ensemble_frozen_majority"] = (
        ensemble.frozen_positive_vote_rate >= 0.5
    ).astype(int)
    ensemble_metrics = metrics_from_decisions(
        ensemble.label,
        ensemble.mean_probability,
        ensemble.ensemble_frozen_majority,
    )

    output_dir = run_root / "external" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(
        prediction_table,
        output_dir / "per_frozen_model_predictions.csv",
    )
    atomic_csv(ensemble, output_dir / "ensemble_predictions.csv")
    atomic_csv(
        pd.DataFrame(model_metrics),
        output_dir / "per_model_metrics.csv",
    )
    result = {
        "status": "COMPLETE",
        "external_manifest": str(manifest_path),
        "n_external_images": int(len(frame)),
        "n_frozen_models": int(len(settings.seeds) * settings.n_folds),
        "retraining_performed": False,
        "temperature_refitting_performed": False,
        "threshold_retuning_performed": False,
        "ensemble_rule": (
            "majority vote of each model's frozen calibrated-threshold decision"
        ),
        "ensemble_metrics": ensemble_metrics,
    }
    atomic_json(output_dir / "EXTERNAL_VALIDATION_REPORT.json", result)
    result["secondary_backup"] = mirror_full_run(run_root, settings)
    atomic_json(output_dir / "EXTERNAL_VALIDATION_REPORT.json", result)
    mirror_full_run(run_root, settings)
    return result
