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
from .reliable_backup import backup_completed_artifacts, backup_epoch_state
from .reliable_models import build_reliable_model, parameter_summary


@dataclass
class ReliableSettings:
    run_id: str = "RELIABLE_DFU_CV_V1"
    drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard"
    backup_root: str = "/content/drive/MyDrive/DFU-ImageGuard-Backup"
    seeds: tuple[int, ...] = (2026, 2027, 2028)
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    models: tuple[str, ...] = ("convnextv2_tiny", "mobilenetv3_large", "densenet121")
    max_epochs: int = 30
    patience: int = 7
    freeze_epochs: int = 2
    batch_size: int = 16
    num_workers: int = 2
    target_sensitivity: float = 0.95


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _torch(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _manifest(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".tmp")
    }


def _report_backup_status(status: dict[str, Any]) -> None:
    if status.get("degraded"):
        skipped = len(status.get("skipped", []))
        print(
            f"SECONDARY BACKUP: DEGRADED ({skipped} file(s) skipped). "
            "Primary Drive checkpoints remain authoritative; training continues."
        )
    else:
        print(
            f"SECONDARY BACKUP: VERIFIED [{status.get('phase')}] "
            f"{status.get('copied_files', 0)} file(s)."
        )


def _infer(model, loader, device):
    import torch

    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for xb, yb, idx in loader:
            logits.append(
                model(xb.to(device, non_blocking=True))
                .reshape(-1)
                .float()
                .cpu()
                .numpy()
            )
            labels.append(yb.numpy())
            indices.append(idx.numpy())
    return (
        np.concatenate(logits),
        np.concatenate(labels).astype(int),
        np.concatenate(indices).astype(int),
    )


def _train_trial(
    train_df,
    selection_df,
    cal_df,
    test_df,
    model_key,
    seed,
    fold,
    cfg,
    settings,
    trial,
    run,
):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    done = trial / "COMPLETE.json"
    pred_path = trial / "test_predictions.csv"
    if done.exists() and pred_path.exists():
        print(f"RESUME: completed trial skipped — {model_key} seed={seed} fold={fold + 1}")
        return json.loads(done.read_text(encoding="utf-8")), pd.read_csv(pred_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU runtime required")

    seed_everything(seed)
    model = build_reliable_model(model_key, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.max_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)

    train_tf, eval_tf = build_transforms(cfg)
    train_loader = build_loader(train_df, train_tf, cfg, True, seed)
    selection_loader = build_loader(selection_df, eval_tf, cfg, False, seed + 1)
    positives = float((train_df.label == 1).sum())
    negatives = float((train_df.label == 0).sum())
    pos_weight = torch.tensor(negatives / max(positives, 1), device=device)

    last = trial / "last_resume.pt"
    best = trial / "best_model.pt"
    history: list[dict[str, Any]] = []
    start = 1
    best_auc = -1.0
    patience_left = settings.patience

    if last.exists():
        payload = torch.load(last, map_location=device, weights_only=False)
        if (
            payload.get("model_key") == model_key
            and payload.get("seed") == seed
            and payload.get("fold") == fold
        ):
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            scaler.load_state_dict(payload["scaler"])
            start = int(payload["epoch"]) + 1
            best_auc = float(payload["best_auc"])
            patience_left = int(payload["patience_left"])
            if (trial / "history.csv").exists():
                history = pd.read_csv(trial / "history.csv").to_dict("records")
            print(
                f"RESUME: {model_key} seed={seed} fold={fold + 1} "
                f"from epoch {start}"
            )

    for epoch in range(start, settings.max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_examples = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                loss = F.binary_cross_entropy_with_logits(
                    model(xb).reshape(-1), yb, pos_weight=pos_weight
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss) * len(yb)
            n_examples += len(yb)

        scheduler.step()
        val_logits, val_y, _ = _infer(model, selection_loader, device)
        selection_auc = float(roc_auc_score(val_y, sigmoid_np(val_logits)))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_examples, 1),
            "selection_auc": selection_auc,
        }
        history.append(row)
        _csv(trial / "history.csv", pd.DataFrame(history))
        print(model_key, seed, fold + 1, row)

        if selection_auc > best_auc + 1e-5:
            best_auc = selection_auc
            patience_left = settings.patience
            _torch(
                best,
                {
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

        _torch(
            last,
            {
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
        backup_status = backup_epoch_state(
            run=run,
            backup_root=settings.backup_root,
            run_id=settings.run_id,
            trial=trial,
        )
        _report_backup_status(backup_status)
        if patience_left <= 0:
            break

    payload = torch.load(best, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    portable = {
        key: (
            value.detach().cpu().half()
            if torch.is_floating_point(value)
            else value.detach().cpu()
        )
        for key, value in model.state_dict().items()
    }
    _torch(
        trial / "best_model_portable_fp16.pt",
        {
            "model_key": model_key,
            "seed": seed,
            "fold": fold,
            "state_dict": portable,
            "params": payload["params"],
        },
    )

    cal_logits, cal_y, _ = _infer(
        model,
        build_loader(cal_df, eval_tf, cfg, False, seed + 300),
        device,
    )
    test_logits, test_y, _ = _infer(
        model,
        build_loader(test_df, eval_tf, cfg, False, seed + 400),
        device,
    )
    temperature = fit_temperature(cal_logits, cal_y)
    calibrated_calibration_prob = sigmoid_np(cal_logits / temperature)
    threshold, threshold_rule = select_threshold(
        cal_y,
        calibrated_calibration_prob,
        settings.target_sensitivity,
    )
    probability = sigmoid_np(test_logits / temperature)
    prediction = (probability >= threshold).astype(int)

    frame = test_df.reset_index(drop=True)[
        ["image_id", "group_id", "label", "label_name", "relative_path"]
    ].copy()
    frame["model_key"] = model_key
    frame["seed"] = seed
    frame["outer_fold"] = fold + 1
    frame["logit"] = test_logits
    frame["prob_calibrated"] = probability
    frame["pred"] = prediction
    frame["temperature"] = temperature
    frame["threshold"] = threshold
    _csv(pred_path, frame)

    metrics = {
        **metric_dict(test_y, probability, prediction),
        "model_key": model_key,
        "seed": seed,
        "outer_fold": fold + 1,
        "temperature": temperature,
        "threshold": threshold,
        "threshold_rule": threshold_rule,
        "best_model_sha256": sha256_file(best),
        "last_resume_sha256": sha256_file(last),
    }
    _json(done, metrics)
    return metrics, frame


def run_reliable_framework(settings: ReliableSettings | None = None):
    settings = settings or ReliableSettings()
    started = time.time()
    run = Path(settings.drive_root) / "runs" / settings.run_id
    run.mkdir(parents=True, exist_ok=True)
    marker = run / "DRIVE_SENTINEL.txt"
    marker.write_text(str(time.time_ns()), encoding="utf-8")
    if not marker.read_text(encoding="utf-8"):
        raise RuntimeError("Primary Drive write/read verification failed")

    # Migrates and removes the old full-mirror secondary layout once, releasing quota.
    initial_backup = backup_completed_artifacts(
        run=run,
        backup_root=settings.backup_root,
        run_id=settings.run_id,
    )
    _report_backup_status(initial_backup)

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

    dirs = {
        "root": run,
        **{
            name: run / name
            for name in [
                "tables",
                "figures",
                "models",
                "xai",
                "predictions",
                "logs",
                "configs",
                "manifests",
                "cache",
            ]
        },
    }
    for path in dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    data = make_outer_folds(
        assign_duplicate_groups(
            build_manifest(download_dataset(cfg, dirs), cfg, dirs), cfg, dirs
        ),
        cfg,
        dirs,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    expected_trials = len(settings.folds) * len(settings.seeds) * len(settings.models)

    for fold in settings.folds:
        outer_train = data[data.outer_fold != fold].copy()
        test = data[data.outer_fold == fold].copy().reset_index(drop=True)
        inner = make_inner_partition(outer_train, cfg, fold)
        train = inner[inner.inner_role == "train"]
        selection = inner[inner.inner_role == "selection"]
        calibration = inner[inner.inner_role == "calibration"]

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
                metrics, predictions = _train_trial(
                    train,
                    selection,
                    calibration,
                    test,
                    model_key,
                    seed,
                    fold,
                    cfg,
                    settings,
                    trial,
                    run,
                )
                metric_rows.append(metrics)
                prediction_frames.append(predictions)

                metrics_frame = pd.DataFrame(metric_rows)
                all_predictions = pd.concat(prediction_frames, ignore_index=True)
                _csv(run / "tables" / "fold_seed_metrics.csv", metrics_frame)
                _csv(run / "tables" / "all_oof_predictions.csv", all_predictions)
                progress = {
                    "completed_trials": len(metrics_frame),
                    "expected_trials": expected_trials,
                    "last_completed": {
                        "model_key": model_key,
                        "seed": seed,
                        "outer_fold": fold + 1,
                    },
                    "updated_at_ns": time.time_ns(),
                }
                _json(run / "RUN_PROGRESS.json", progress)

                completed_backup = backup_completed_artifacts(
                    run=run,
                    backup_root=settings.backup_root,
                    run_id=settings.run_id,
                )
                _report_backup_status(completed_backup)

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

    with (run / "reliable_dfu_reproducibility.pkl").open("wb") as handle:
        pickle.dump(
            {
                "settings": asdict(settings),
                "metrics": metrics_frame.to_dict("records"),
                "predictions": all_predictions.to_dict("records"),
                "reports": reports,
            },
            handle,
            pickle.HIGHEST_PROTOCOL,
        )

    _json(run / "ARTIFACT_MANIFEST.json", _manifest(run))
    final_backup = backup_completed_artifacts(
        run=run,
        backup_root=settings.backup_root,
        run_id=settings.run_id,
    )
    _report_backup_status(final_backup)
    final = {
        "run_id": settings.run_id,
        "completed_trials": len(metrics_frame),
        "expected_trials": expected_trials,
        "primary_drive": str(run),
        "secondary_drive": str(
            Path(settings.backup_root) / "runs" / settings.run_id
        ),
        "secondary_backup": final_backup,
        "reports": reports,
        "elapsed_minutes": (time.time() - started) / 60,
    }
    _json(run / "FINAL_VERIFICATION.json", final)
    print(json.dumps(final, indent=2, default=str))
    return final
