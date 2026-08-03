from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
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
    seed_everything,
    sha256_file,
)
from .evaluation import fit_temperature, select_threshold, sigmoid_np
from .models_training import build_loader, build_transforms
from .reliable_analysis import build_reports, metric_dict
from .reliable_models import build_reliable_model, parameter_summary
from .reliable_storage_v2 import (
    atomic_json,
    backup_active_trial,
    backup_metadata,
    clear_active_trial_backup,
    retire_completed_full_checkpoints,
    verify_portable_checkpoint,
)


@dataclass
class ReliableSettingsV2:
    run_id: str = "RELIABLE_DFU_CV_V2"
    drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard"
    backup_root: str = "/content/drive/MyDrive/DFU-ImageGuard-Backup"
    seeds: tuple[int, ...] = (2026, 2027, 2028)
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    models: tuple[str, ...] = (
        "convnextv2_tiny",
        "mobilenetv3_large",
        "densenet121",
    )
    max_epochs: int = 30
    patience: int = 7
    batch_size: int = 16
    num_workers: int = 2
    target_sensitivity: float = 0.95
    source_commit: str = "unknown"


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def artifact_manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith((".tmp", ".partial"))
    }


def report_backup(status: dict[str, Any]) -> None:
    if status.get("degraded"):
        print(
            "SECONDARY BACKUP: DEGRADED — metadata/resume copy incomplete; "
            "primary Drive remains authoritative and training continues."
        )
    else:
        print(
            f"SECONDARY BACKUP: VERIFIED [{status.get('phase')}] "
            f"{status.get('copied_files', 0)} file(s)."
        )


def infer(model, loader, device):
    import torch

    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for xb, yb, idx in loader:
            output = model(xb.to(device, non_blocking=True)).reshape(-1)
            logits.append(output.float().cpu().numpy())
            labels.append(yb.numpy())
            indices.append(idx.numpy())
    return (
        np.concatenate(logits),
        np.concatenate(labels).astype(int),
        np.concatenate(indices).astype(int),
    )


def completed_trial_is_valid(trial: Path, model_key: str, seed: int, fold: int) -> bool:
    required = [
        trial / "COMPLETE.json",
        trial / "test_predictions.csv",
        trial / "best_model_portable_fp16.pt",
    ]
    if not all(path.is_file() for path in required):
        return False
    verify_portable_checkpoint(
        trial / "best_model_portable_fp16.pt",
        expected_model_key=model_key,
        expected_seed=seed,
        expected_fold=fold,
    )
    return True


def train_trial(
    train_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_key: str,
    seed: int,
    fold: int,
    cfg: Config,
    settings: ReliableSettingsV2,
    trial: Path,
    run: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    complete_path = trial / "COMPLETE.json"
    prediction_path = trial / "test_predictions.csv"
    portable_path = trial / "best_model_portable_fp16.pt"

    if completed_trial_is_valid(trial, model_key, seed, fold):
        retention_path = trial / "CHECKPOINT_RETENTION.json"
        if not retention_path.exists() or (trial / "last_resume.pt").exists() or (trial / "best_model.pt").exists():
            retire_completed_full_checkpoints(trial, model_key, seed, fold)
        print(f"RESUME: completed trial skipped — {model_key} seed={seed} fold={fold + 1}")
        return (
            json.loads(complete_path.read_text(encoding="utf-8")),
            pd.read_csv(prediction_path),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU runtime is required; training was not started.")

    seed_everything(seed)
    model = build_reliable_model(model_key, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.max_epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)

    train_transform, evaluation_transform = build_transforms(cfg)
    train_loader = build_loader(train_df, train_transform, cfg, True, seed)
    selection_loader = build_loader(
        selection_df,
        evaluation_transform,
        cfg,
        False,
        seed + 1,
    )
    positives = float((train_df.label == 1).sum())
    negatives = float((train_df.label == 0).sum())
    pos_weight = torch.tensor(negatives / max(positives, 1.0), device=device)

    last_path = trial / "last_resume.pt"
    best_path = trial / "best_model.pt"
    history_path = trial / "history.csv"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_auc = -1.0
    patience_left = settings.patience

    if last_path.is_file():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        identity_matches = (
            resume.get("model_key") == model_key
            and int(resume.get("seed")) == int(seed)
            and int(resume.get("fold")) == int(fold)
            and resume.get("source_commit") == settings.source_commit
        )
        if not identity_matches:
            raise RuntimeError(
                f"Checkpoint identity mismatch in {last_path}; refusing unsafe resume."
            )
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        best_auc = float(resume["best_auc"])
        patience_left = int(resume["patience_left"])
        if history_path.is_file():
            history = pd.read_csv(history_path).to_dict("records")
        print(
            f"RESUME: {model_key} seed={seed} fold={fold + 1} "
            f"from epoch {start_epoch}"
        )

    for epoch in range(start_epoch, settings.max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_examples = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                logits = model(xb).reshape(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    yb,
                    pos_weight=pos_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss) * len(yb)
            n_examples += len(yb)

        scheduler.step()
        selection_logits, selection_y, _ = infer(model, selection_loader, device)
        selection_auc = float(
            roc_auc_score(selection_y, sigmoid_np(selection_logits))
        )
        history_row = {
            "epoch": int(epoch),
            "train_loss": total_loss / max(n_examples, 1),
            "selection_auc": selection_auc,
        }
        history.append(history_row)
        atomic_csv(history_path, pd.DataFrame(history))
        print(model_key, seed, fold + 1, history_row)

        if selection_auc > best_auc + 1e-5:
            best_auc = selection_auc
            patience_left = settings.patience
            atomic_torch(
                best_path,
                {
                    "format_version": 2,
                    "source_commit": settings.source_commit,
                    "model_key": model_key,
                    "seed": seed,
                    "fold": fold,
                    "epoch": epoch,
                    "best_auc": selection_auc,
                    "model": model.state_dict(),
                    "params": parameter_summary(model),
                },
            )
        else:
            patience_left -= 1

        atomic_torch(
            last_path,
            {
                "format_version": 2,
                "source_commit": settings.source_commit,
                "model_key": model_key,
                "seed": seed,
                "fold": fold,
                "epoch": epoch,
                "best_auc": best_auc,
                "patience_left": patience_left,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
        )
        report_backup(
            backup_active_trial(
                run=run,
                backup_root=settings.backup_root,
                run_id=settings.run_id,
                trial=trial,
            )
        )
        if patience_left <= 0:
            break

    if not best_path.is_file():
        raise RuntimeError(f"Best checkpoint was not created: {best_path}")

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    portable_state = {
        key: (
            value.detach().cpu().half()
            if torch.is_floating_point(value)
            else value.detach().cpu()
        )
        for key, value in model.state_dict().items()
    }
    atomic_torch(
        portable_path,
        {
            "format_version": 2,
            "source_commit": settings.source_commit,
            "model_key": model_key,
            "seed": seed,
            "fold": fold,
            "best_epoch": int(best_payload["epoch"]),
            "best_selection_auc": float(best_payload["best_auc"]),
            "state_dict": portable_state,
            "params": best_payload["params"],
        },
    )
    portable_meta = verify_portable_checkpoint(
        portable_path,
        expected_model_key=model_key,
        expected_seed=seed,
        expected_fold=fold,
    )

    calibration_logits, calibration_y, _ = infer(
        model,
        build_loader(
            calibration_df,
            evaluation_transform,
            cfg,
            False,
            seed + 300,
        ),
        device,
    )
    test_logits, test_y, _ = infer(
        model,
        build_loader(test_df, evaluation_transform, cfg, False, seed + 400),
        device,
    )

    temperature = fit_temperature(calibration_logits, calibration_y)
    calibration_probability = sigmoid_np(calibration_logits / temperature)
    threshold, threshold_rule = select_threshold(
        calibration_y,
        calibration_probability,
        settings.target_sensitivity,
    )
    raw_probability = sigmoid_np(test_logits)
    calibrated_probability = sigmoid_np(test_logits / temperature)
    prediction = (calibrated_probability >= threshold).astype(int)

    prediction_frame = test_df.reset_index(drop=True)[
        ["image_id", "group_id", "label", "label_name", "relative_path"]
    ].copy()
    prediction_frame["model_key"] = model_key
    prediction_frame["seed"] = seed
    prediction_frame["outer_fold"] = fold + 1
    prediction_frame["logit"] = test_logits
    prediction_frame["prob_raw"] = raw_probability
    prediction_frame["prob_calibrated"] = calibrated_probability
    prediction_frame["pred"] = prediction
    prediction_frame["temperature"] = temperature
    prediction_frame["threshold"] = threshold
    atomic_csv(prediction_path, prediction_frame)

    raw_metrics = metric_dict(
        test_y,
        raw_probability,
        (raw_probability >= 0.5).astype(int),
    )
    full_best_sha = sha256_file(best_path)
    full_last_sha = sha256_file(last_path)
    metrics = {
        **metric_dict(test_y, calibrated_probability, prediction),
        "model_key": model_key,
        "seed": seed,
        "outer_fold": fold + 1,
        "source_commit": settings.source_commit,
        "temperature": float(temperature),
        "threshold": float(threshold),
        "threshold_rule": threshold_rule,
        "best_epoch": int(best_payload["epoch"]),
        "best_selection_auc": float(best_payload["best_auc"]),
        "train_n": int(len(train_df)),
        "selection_n": int(len(selection_df)),
        "calibration_n": int(len(calibration_df)),
        "test_n": int(len(test_df)),
        "raw_brier": raw_metrics["brier"],
        "raw_ece": raw_metrics["ece"],
        "raw_log_loss": raw_metrics["log_loss"],
        "portable_model_sha256": portable_meta["sha256"],
        "portable_model_bytes": portable_meta["bytes"],
        "completed_full_best_sha256": full_best_sha,
        "completed_last_resume_sha256": full_last_sha,
    }
    atomic_json(complete_path, metrics)

    retention = retire_completed_full_checkpoints(
        trial=trial,
        model_key=model_key,
        seed=seed,
        fold=fold,
    )
    report_backup(
        clear_active_trial_backup(
            run=run,
            backup_root=settings.backup_root,
            run_id=settings.run_id,
        )
    )
    atomic_json(
        trial / "TRIAL_VERIFICATION.json",
        {
            "status": "PASS",
            "metrics_sha256": sha256_file(complete_path),
            "predictions_sha256": sha256_file(prediction_path),
            "portable_checkpoint": portable_meta,
            "retention": retention,
            "verified_at_ns": time.time_ns(),
        },
    )
    return metrics, prediction_frame


def build_config(settings: ReliableSettingsV2) -> Config:
    cfg = Config()
    cfg.DRIVE_ROOT = settings.drive_root
    cfg.ALLOW_LOCAL_FALLBACK = False
    cfg.N_FOLDS = 5
    cfg.SEED = 2026
    cfg.BATCH_SIZE = settings.batch_size
    cfg.NUM_WORKERS = settings.num_workers
    cfg.MAX_EPOCHS = settings.max_epochs
    cfg.PATIENCE = settings.patience
    cfg.TARGET_SENSITIVITY = settings.target_sensitivity
    return cfg


def run_reliable_framework_v2(
    settings: ReliableSettingsV2 | None = None,
) -> dict[str, Any]:
    settings = settings or ReliableSettingsV2()
    started = time.time()
    run = Path(settings.drive_root) / "runs" / settings.run_id
    run.mkdir(parents=True, exist_ok=True)

    sentinel = run / "DRIVE_SENTINEL.txt"
    sentinel_value = str(time.time_ns())
    sentinel.write_text(sentinel_value, encoding="utf-8")
    if sentinel.read_text(encoding="utf-8") != sentinel_value:
        raise RuntimeError("Primary Google Drive write/read verification failed")

    atomic_json(
        run / "STORAGE_POLICY.json",
        {
            "policy_version": 2,
            "run_id": settings.run_id,
            "source_commit": settings.source_commit,
            "primary_policy": "Only the active trial keeps FP32 optimizer-resume checkpoints. Completed trials retain verified FP16 inference weights, predictions, histories and metrics.",
            "secondary_policy": "One rolling active resume copy plus metadata only; completed model weights are not duplicated within the same Drive account.",
            "expected_trials": len(settings.folds) * len(settings.seeds) * len(settings.models),
            "created_or_verified_at_ns": time.time_ns(),
        },
    )
    report_backup(
        backup_metadata(
            run=run,
            backup_root=settings.backup_root,
            run_id=settings.run_id,
        )
    )

    cfg = build_config(settings)
    dirs = {
        "root": run,
        **{
            name: run / name
            for name in (
                "tables",
                "figures",
                "models",
                "xai",
                "predictions",
                "logs",
                "configs",
                "manifests",
                "cache",
            )
        },
    }
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    atomic_json(run / "configs" / "resolved_settings_v2.json", asdict(settings))

    dataset_root = download_dataset(cfg, dirs)
    manifest = build_manifest(dataset_root, cfg, dirs)
    cleaned = assign_duplicate_groups(manifest, cfg, dirs)
    data = make_outer_folds(cleaned, cfg, dirs)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    expected_trials = len(settings.folds) * len(settings.seeds) * len(settings.models)

    for fold in settings.folds:
        outer_train = data[data.outer_fold != fold].copy()
        test_df = data[data.outer_fold == fold].copy().reset_index(drop=True)
        inner = make_inner_partition(outer_train, cfg, fold)
        train_df = inner[inner.inner_role == "train"].copy()
        selection_df = inner[inner.inner_role == "selection"].copy()
        calibration_df = inner[inner.inner_role == "calibration"].copy()

        for seed in settings.seeds:
            for model_key in settings.models:
                trial = (
                    run
                    / "trials"
                    / model_key
                    / f"seed_{seed}"
                    / f"fold_{fold + 1}"
                )
                trial.mkdir(parents=True, exist_ok=True)
                metrics, predictions = train_trial(
                    train_df=train_df,
                    selection_df=selection_df,
                    calibration_df=calibration_df,
                    test_df=test_df,
                    model_key=model_key,
                    seed=seed,
                    fold=fold,
                    cfg=cfg,
                    settings=settings,
                    trial=trial,
                    run=run,
                )
                metric_rows.append(metrics)
                prediction_frames.append(predictions)

                metrics_frame = pd.DataFrame(metric_rows)
                all_predictions = pd.concat(prediction_frames, ignore_index=True)
                atomic_csv(run / "tables" / "fold_seed_metrics.csv", metrics_frame)
                atomic_csv(
                    run / "tables" / "all_oof_predictions.csv",
                    all_predictions,
                )
                atomic_json(
                    run / "RUN_PROGRESS.json",
                    {
                        "completed_trials": len(metrics_frame),
                        "expected_trials": expected_trials,
                        "last_completed": {
                            "model_key": model_key,
                            "seed": seed,
                            "outer_fold": fold + 1,
                        },
                        "source_commit": settings.source_commit,
                        "updated_at_ns": time.time_ns(),
                    },
                )
                report_backup(
                    backup_metadata(
                        run=run,
                        backup_root=settings.backup_root,
                        run_id=settings.run_id,
                    )
                )

    metrics_frame = pd.DataFrame(metric_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    reports = build_reports(run)
    summary = metrics_frame.groupby("model_key").agg(
        {
            "balanced_accuracy": ["mean", "std"],
            "sensitivity": ["mean", "std"],
            "specificity": ["mean", "std"],
            "roc_auc": ["mean", "std"],
            "pr_auc": ["mean", "std"],
            "brier": ["mean", "std"],
            "ece": ["mean", "std"],
        }
    )
    summary.to_csv(run / "tables" / "model_summary.csv")

    model_index_rows: list[dict[str, Any]] = []
    for path in sorted(run.rglob("best_model_portable_fp16.pt")):
        relative = path.relative_to(run)
        model_index_rows.append(
            {
                "relative_path": str(relative),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    atomic_csv(run / "tables" / "model_checkpoint_index.csv", pd.DataFrame(model_index_rows))

    with (run / "reliable_dfu_reproducibility.pkl").open("wb") as handle:
        pickle.dump(
            {
                "settings": asdict(settings),
                "metrics": metrics_frame.to_dict("records"),
                "predictions": all_predictions.to_dict("records"),
                "reports": reports,
                "model_index": model_index_rows,
            },
            handle,
            pickle.HIGHEST_PROTOCOL,
        )

    atomic_json(run / "ARTIFACT_MANIFEST.json", artifact_manifest(run))
    final_backup = backup_metadata(
        run=run,
        backup_root=settings.backup_root,
        run_id=settings.run_id,
    )
    report_backup(final_backup)
    final = {
        "run_id": settings.run_id,
        "source_commit": settings.source_commit,
        "completed_trials": len(metrics_frame),
        "expected_trials": expected_trials,
        "primary_drive": str(run),
        "secondary_metadata_backup": str(
            Path(settings.backup_root) / "runs" / settings.run_id
        ),
        "portable_models": len(model_index_rows),
        "secondary_backup": final_backup,
        "reports": reports,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    atomic_json(run / "FINAL_VERIFICATION.json", final)
    print(json.dumps(final, indent=2, default=str))
    return final
