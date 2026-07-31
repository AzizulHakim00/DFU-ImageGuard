from __future__ import annotations
import base64, dataclasses, datetime as dt, gc, hashlib, json, math, os, pickle, platform, random, shutil, subprocess, sys, time, traceback, warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
warnings.filterwarnings("ignore", category=UserWarning)

from .config_data import *
from .models_training import *
from .evaluation import *
from .statistics_figures import *
from .robustness_xai import *
from .artifacts import *

def run_complete_pipeline(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = Config()
    for key, value in (overrides or {}).items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown configuration key: {key}")
        setattr(cfg, key, value)
    mount_drive()
    seed_everything(cfg.SEED)
    dirs = prepare_run_dirs(cfg)
    started = time.time()
    try:
        dataset_root = download_dataset(cfg, dirs)
        initial = build_manifest(dataset_root, cfg, dirs)
        cleaned = assign_duplicate_groups(initial, cfg, dirs)
        manifest = make_outer_folds(cleaned, cfg, dirs)
        dmeta = dataset_manifest(manifest, cfg, dataset_root)
        write_json(dirs["root"] / "dataset_manifest.json", dmeta)

        prediction_frames: list[pd.DataFrame] = []
        calibrations: list[dict[str, Any]] = []
        for fold in range(cfg.N_FOLDS):
            pred, info = run_torch_fold(cfg.PRIMARY_MODEL_NAME, lambda: build_proposed_model(cfg), fold,
                                        manifest, cfg, dirs, primary=True)
            prediction_frames.append(pred); calibrations.append(info)
            pd.concat([p for p in prediction_frames if p.model.iloc[0] == cfg.PRIMARY_MODEL_NAME], ignore_index=True).to_csv(
                dirs["predictions"] / "proposed_oof_progress.csv", index=False
            )
            write_json(dirs["logs"] / "primary_fit_registry.json", calibrations)

        if cfg.RUN_BASELINES:
            for baseline_name, timm_name in cfg.BASELINE_MODELS.items():
                for fold in range(cfg.N_FOLDS):
                    pred, info = run_torch_fold(baseline_name, lambda n=timm_name: build_baseline_model(n), fold,
                                                manifest, cfg, dirs, primary=False)
                    prediction_frames.append(pred); calibrations.append(info)
            linear_pred, linear_info = run_linear_baseline(manifest, cfg, dirs)
            prediction_frames.append(linear_pred); calibrations.extend(linear_info)

        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_csv(dirs["predictions"] / "all_oof_predictions.csv", index=False)
        predictions[predictions.model == cfg.PRIMARY_MODEL_NAME].to_csv(
            dirs["predictions"] / "dfu_imageguard_oof_predictions.csv", index=False
        )
        metrics = aggregate_metrics(predictions, dirs)
        cis = bootstrap_metric_cis(predictions, cfg, dirs)
        stats = statistical_comparisons(predictions, cfg, dirs)
        uncertainty = uncertainty_summary(predictions, dirs)
        error_df = error_analysis(predictions, cfg, dirs)
        make_core_figures(manifest, predictions, metrics, dirs)
        make_error_gallery(error_df, dirs)
        robustness = run_robustness(manifest, predictions, cfg, dirs) if cfg.RUN_ROBUSTNESS else pd.DataFrame()
        xai = run_xai(manifest, predictions, cfg, dirs) if cfg.RUN_XAI else pd.DataFrame()

        versions = software_hardware_versions()
        write_json(dirs["root"] / "software_versions.json", versions)
        write_model_card(cfg, dirs, metrics)
        write_limitations(cfg, dirs)
        write_external_validation_status(dirs)
        paper_results = {
            "run_id": cfg.RUN_ID,
            "primary_model": cfg.PRIMARY_MODEL_NAME,
            "evaluation": "nested duplicate-group-aware five-fold OOF",
            "metrics": metrics.to_dict("records"),
            "confidence_intervals": cis.to_dict("records"),
            "statistics": stats.to_dict("records"),
            "external_validation": {"performed": False, "reason": "No scientifically compatible independent dataset was supplied."},
            "clinical_deployment_claim": False,
        }
        write_json(dirs["root"] / "paper_results.json", paper_results)
        pkl_path = create_reproducibility_pkl(cfg, dirs, dmeta, manifest, predictions, metrics, cis, stats,
                                              robustness, xai, uncertainty, calibrations)
        write_json(dirs["root"] / "manifest.json", artifact_manifest(dirs["root"]))
        github_status = push_to_github(dirs["root"], cfg, dirs)
        write_json(dirs["root"] / "manifest.json", artifact_manifest(dirs["root"]))
        verification = verify_run(cfg, dirs, manifest, calibrations, pkl_path, github_status)
        verification["elapsed_minutes"] = (time.time() - started) / 60
        return verification
    except Exception as exc:
        failure = {
            "run_id": cfg.RUN_ID, "error": repr(exc), "traceback": traceback.format_exc(),
            "drive_run_preserved": str(dirs["root"]), "timestamp": dt.datetime.now().isoformat(),
        }
        write_json(dirs["logs"] / "FAILED_RUN.json", failure)
        raise


def find_latest_drive_run(drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard") -> Path:
    runs = Path(drive_root) / "runs"
    valid = [p for p in runs.glob("*") if (p / "dfu_imageguard_complete_reproducibility.pkl").exists()]
    if not valid:
        raise FileNotFoundError(f"No completed run found under {runs}")
    return sorted(valid)[-1]


def load_latest_artifacts(drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard") -> dict[str, Any]:
    mount_drive()
    run = find_latest_drive_run(drive_root)
    pkl_path = run / "dfu_imageguard_complete_reproducibility.pkl"
    print(f"Loading trusted local artifact: {pkl_path}")
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    metrics = pd.DataFrame(payload["metrics"])
    predictions = pd.DataFrame(payload["oof_predictions"])
    from IPython.display import display
    display(metrics[metrics.state.isin(["raw", "calibrated"])])
    print(f"Figures: {run / 'figures'}")
    print(f"Tables: {run / 'tables'}")
    return payload


def upload_existing_run(run_id: Optional[str] = None, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = Config()
    for k, v in (overrides or {}).items(): setattr(cfg, k, v)
    mount_drive()
    run = Path(cfg.DRIVE_ROOT) / "runs" / run_id if run_id else find_latest_drive_run(cfg.DRIVE_ROOT)
    cfg.RUN_ID = run.name
    dirs = {"root": run}
    status = push_to_github(run, cfg, dirs)
    print(json.dumps(status, indent=2))
    return status


def future_notebook_guard(kind: str) -> None:
    messages = {
        "ablation": "Ablation requires training modified model variants. It is intentionally disabled to avoid presenting post-hoc or fabricated ablation results.",
        "multiseed": "Genuine multi-seed analysis requires additional training. Suggested seeds are [3, 5, 13]; this is separate from five-fold stability.",
        "external": "External validation requires a scientifically compatible independent dataset and frozen preprocessing, weights, calibration and threshold. A random holdout is not external validation.",
        "federated": "Federated learning is only a simulation unless genuine hospital/site identifiers exist. No hospitals will be fabricated.",
        "multimodal": "Multimodal analysis is disabled unless legitimate structured clinical metadata accompany the images.",
    }
    print(messages.get(kind, "This future analysis is intentionally separated from the primary experiment."))
