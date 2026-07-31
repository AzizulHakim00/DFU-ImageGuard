from __future__ import annotations

import datetime as dt
import json
import pickle
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .artifacts import (
    artifact_manifest,
    create_reproducibility_pkl,
    dataset_manifest,
    push_to_github,
    software_hardware_versions,
    verify_run,
    write_external_validation_status,
    write_limitations,
    write_model_card,
)
from .config_data import (
    Config,
    assign_duplicate_groups,
    build_manifest,
    download_dataset,
    google_drive_is_mounted,
    make_outer_folds,
    mount_drive,
    now_run_id,
    prepare_run_dirs,
    seed_everything,
    write_json,
)
from .evaluation import aggregate_metrics, run_linear_baseline, run_torch_fold
from .models_training import build_baseline_model, build_proposed_model
from .robustness_xai import (
    error_analysis,
    make_error_gallery,
    run_robustness,
    run_xai,
    uncertainty_summary,
)
from .statistics_figures import (
    bootstrap_metric_cis,
    make_core_figures,
    statistical_comparisons,
)


def resolve_resumable_run_id(cfg: Config) -> None:
    project_root = Path(cfg.DRIVE_ROOT)
    runs_root = project_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    active_marker = project_root / "ACTIVE_RUN.txt"
    if cfg.RUN_ID is None:
        candidate = active_marker.read_text(encoding="utf-8").strip() if active_marker.exists() else ""
        candidate_root = runs_root / candidate if candidate else None
        if candidate_root and candidate_root.exists() and not (candidate_root / "final_verification.json").exists():
            cfg.RUN_ID = candidate
            print(f"Resuming interrupted run: {cfg.RUN_ID}")
        else:
            cfg.RUN_ID = now_run_id()
            active_marker.write_text(str(cfg.RUN_ID) + "\n", encoding="utf-8")
            print(f"Starting new run: {cfg.RUN_ID}")
    else:
        active_marker.write_text(str(cfg.RUN_ID) + "\n", encoding="utf-8")


def _primary_factory(cfg: Config):
    return lambda pretrained=True: build_proposed_model(cfg, pretrained=pretrained)


def _baseline_factory(timm_name: str):
    return lambda pretrained=True: build_baseline_model(timm_name, pretrained=pretrained)


def _resolve_storage(cfg: Config) -> None:
    mount_drive(cfg)
    # A failed older run may have created a normal local directory named
    # /content/drive/MyDrive. A directory alone is not proof that Drive is mounted.
    if cfg.STORAGE_MODE == "google_drive" and not google_drive_is_mounted():
        if not cfg.ALLOW_LOCAL_FALLBACK:
            raise RuntimeError("Google Drive is not an actual mounted filesystem")
        cfg.DRIVE_ROOT = cfg.LOCAL_FALLBACK_ROOT
        cfg.STORAGE_MODE = "local_fallback"
        Path(cfg.DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
        print("WARNING: stale /content/drive directory detected; using verified local fallback storage.")
        print(f"Local fallback: {cfg.DRIVE_ROOT}")


def run_complete_pipeline(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = Config()
    for key, value in (overrides or {}).items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown configuration key: {key}")
        setattr(cfg, key, value)

    _resolve_storage(cfg)
    resolve_resumable_run_id(cfg)
    seed_everything(cfg.SEED)
    dirs = prepare_run_dirs(cfg)
    started = time.time()

    try:
        dataset_root = download_dataset(cfg, dirs)
        initial_manifest = build_manifest(dataset_root, cfg, dirs)
        cleaned_manifest = assign_duplicate_groups(initial_manifest, cfg, dirs)
        manifest = make_outer_folds(cleaned_manifest, cfg, dirs)
        dataset_meta = dataset_manifest(manifest, cfg, dataset_root)
        dataset_meta["storage_mode"] = cfg.STORAGE_MODE
        write_json(dirs["root"] / "dataset_manifest.json", dataset_meta)

        prediction_frames: list[pd.DataFrame] = []
        calibrations: list[dict[str, Any]] = []
        primary_factory = _primary_factory(cfg)

        for fold in range(cfg.N_FOLDS):
            prediction, info = run_torch_fold(
                cfg.PRIMARY_MODEL_NAME,
                primary_factory,
                fold,
                manifest,
                cfg,
                dirs,
                primary=True,
            )
            prediction_frames.append(prediction)
            calibrations.append(info)
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                dirs["predictions"] / "proposed_oof_progress.csv",
                index=False,
            )
            write_json(dirs["logs"] / "primary_fit_registry.json", calibrations)
            print(
                f"Primary fold {fold + 1}/{cfg.N_FOLDS} complete | "
                f"trained_now={info['trained_now']} | checkpoint={Path(info['checkpoint']).name}"
            )

        if cfg.RUN_BASELINES:
            for baseline_name, timm_name in cfg.BASELINE_MODELS.items():
                factory = _baseline_factory(timm_name)
                for fold in range(cfg.N_FOLDS):
                    prediction, info = run_torch_fold(
                        baseline_name,
                        factory,
                        fold,
                        manifest,
                        cfg,
                        dirs,
                        primary=False,
                    )
                    prediction_frames.append(prediction)
                    calibrations.append(info)
                    print(
                        f"Baseline {baseline_name} fold {fold + 1}/{cfg.N_FOLDS} complete | "
                        f"trained_now={info['trained_now']}"
                    )
            linear_prediction, linear_info = run_linear_baseline(manifest, cfg, dirs)
            prediction_frames.append(linear_prediction)
            calibrations.extend(linear_info)

        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_csv(dirs["predictions"] / "all_oof_predictions.csv", index=False)
        predictions[predictions.model == cfg.PRIMARY_MODEL_NAME].to_csv(
            dirs["predictions"] / "dfu_imageguard_oof_predictions.csv",
            index=False,
        )

        metrics = aggregate_metrics(predictions, dirs)
        confidence_intervals = bootstrap_metric_cis(predictions, cfg, dirs)
        statistics = statistical_comparisons(predictions, cfg, dirs)
        uncertainty = uncertainty_summary(predictions, dirs)
        error_frame = error_analysis(predictions, cfg, dirs)
        make_core_figures(manifest, predictions, metrics, dirs)
        make_error_gallery(error_frame, dirs)
        robustness = run_robustness(manifest, predictions, cfg, dirs) if cfg.RUN_ROBUSTNESS else pd.DataFrame()
        xai = run_xai(manifest, predictions, cfg, dirs) if cfg.RUN_XAI else pd.DataFrame()

        versions = software_hardware_versions()
        write_json(dirs["root"] / "software_versions.json", versions)
        write_model_card(cfg, dirs, metrics)
        write_limitations(cfg, dirs)
        write_external_validation_status(dirs)

        paper_results = {
            "run_id": cfg.RUN_ID,
            "storage_mode": cfg.STORAGE_MODE,
            "primary_model": cfg.PRIMARY_MODEL_NAME,
            "evaluation": "nested duplicate-group-aware five-fold OOF",
            "metrics": metrics.to_dict("records"),
            "confidence_intervals": confidence_intervals.to_dict("records"),
            "statistics": statistics.to_dict("records"),
            "external_validation": {
                "performed": False,
                "reason": "No scientifically compatible independent dataset was supplied.",
            },
            "clinical_deployment_claim": False,
        }
        write_json(dirs["root"] / "paper_results.json", paper_results)

        pkl_path = create_reproducibility_pkl(
            cfg,
            dirs,
            dataset_meta,
            manifest,
            predictions,
            metrics,
            confidence_intervals,
            statistics,
            robustness,
            xai,
            uncertainty,
            calibrations,
        )
        write_json(dirs["root"] / "manifest.json", artifact_manifest(dirs["root"]))
        github_status = push_to_github(dirs["root"], cfg, dirs)
        write_json(dirs["root"] / "manifest.json", artifact_manifest(dirs["root"]))
        verification = verify_run(
            cfg,
            dirs,
            manifest,
            calibrations,
            pkl_path,
            github_status,
        )
        verification["elapsed_minutes"] = (time.time() - started) / 60
        verification["storage_mode"] = cfg.STORAGE_MODE
        write_json(dirs["root"] / "final_verification.json", verification)

        project_root = Path(cfg.DRIVE_ROOT)
        (project_root / "LAST_COMPLETED_RUN.txt").write_text(str(cfg.RUN_ID) + "\n", encoding="utf-8")
        active_marker = project_root / "ACTIVE_RUN.txt"
        if active_marker.exists():
            active_marker.unlink()
        return verification
    except Exception as exc:
        failure = {
            "run_id": cfg.RUN_ID,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "run_storage_path": str(dirs["root"]),
            "storage_mode": cfg.STORAGE_MODE,
            "timestamp": dt.datetime.now().isoformat(),
            "resolved_config": asdict(cfg),
        }
        write_json(dirs["logs"] / "FAILED_RUN.json", failure)
        print(f"Run failed. Diagnostic saved to: {dirs['logs'] / 'FAILED_RUN.json'}")
        raise


def find_latest_drive_run(drive_root: str = "/content/drive/MyDrive/DFU-ImageGuard") -> Path:
    runs = Path(drive_root) / "runs"
    if not runs.exists():
        raise FileNotFoundError(f"Run directory does not exist: {runs}")
    valid = [path for path in runs.glob("*") if (path / "dfu_imageguard_complete_reproducibility.pkl").exists()]
    if not valid:
        raise FileNotFoundError(f"No completed run found under {runs}")
    return max(valid, key=lambda path: path.stat().st_mtime)


def load_latest_artifacts(drive_root: Optional[str] = None) -> dict[str, Any]:
    cfg = Config()
    if drive_root is not None:
        cfg.DRIVE_ROOT = drive_root
        cfg.MOUNT_DRIVE = False
        cfg.LOCAL_FALLBACK_ROOT = drive_root
    _resolve_storage(cfg)
    run = find_latest_drive_run(cfg.DRIVE_ROOT)
    pkl_path = run / "dfu_imageguard_complete_reproducibility.pkl"
    print(f"Loading trusted local artifact: {pkl_path}")
    with pkl_path.open("rb") as file:
        payload = pickle.load(file)
    metrics = pd.DataFrame(payload["metrics"])
    from IPython.display import display

    display(metrics[metrics.state.isin(["raw", "calibrated"])])
    print(f"Figures: {run / 'figures'}")
    print(f"Tables: {run / 'tables'}")
    return payload


def upload_existing_run(
    run_id: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = Config()
    for key, value in (overrides or {}).items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown configuration key: {key}")
        setattr(cfg, key, value)
    _resolve_storage(cfg)
    run = Path(cfg.DRIVE_ROOT) / "runs" / run_id if run_id else find_latest_drive_run(cfg.DRIVE_ROOT)
    if not run.exists():
        raise FileNotFoundError(run)
    cfg.RUN_ID = run.name
    status = push_to_github(run, cfg, {"root": run})
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
