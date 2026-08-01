"""Q1-corrected entry point for the complete DFU-ImageGuard experiment.

The underlying models are trained exactly as in the locked primary pipeline.
After completion, saved OOF predictions are post-processed without retraining to
replace threshold-clipping errors and legacy confidence analyses.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Optional

from .artifacts import artifact_manifest, push_to_github
from .config_data import Config, write_json
from .dfu_imageguard_pipeline import run_complete_pipeline as _run_locked_pipeline
from .q1_posthoc import run_q1_posthoc_corrections


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _patch_reproducibility_pkl(run_root: Path, corrected: dict[str, Any]) -> None:
    pkl_path = run_root / "dfu_imageguard_complete_reproducibility.pkl"
    if not pkl_path.exists():
        return
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["schema_version"] = "1.1-q1-posthoc"
    payload["oof_predictions"] = corrected["predictions"].drop(columns=["image_path"], errors="ignore").to_dict("records")
    payload["metrics"] = corrected["metrics"].to_dict("records")
    payload["confidence_intervals"] = corrected["confidence_intervals"].to_dict("records")
    payload["statistical_comparisons"] = corrected["comparisons"].to_dict("records")
    payload["uncertainty"] = corrected["uncertainty"].to_dict("records")
    payload["q1_readiness"] = corrected["readiness"]
    payload["reporting_corrections"] = {
        "retraining_performed": False,
        "threshold_clipping_bug_corrected": True,
        "threshold_aware_confidence_added": True,
        "group_sign_flip_permutation_added": True,
        "image_level_mcnemar_is_supplemental": True,
    }
    warnings = list(payload.get("warnings", []))
    warnings.append("Use threshold-aware decision_confidence rather than legacy max-probability confidence for selective prediction.")
    payload["warnings"] = sorted(set(warnings))
    _atomic_pickle(pkl_path, payload)


def run_complete_pipeline(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Run the locked experiment, then apply Q1 corrections without retraining."""
    report = _run_locked_pipeline(overrides)
    run_root = Path(report["storage_path"])
    cfg = Config()
    for key, value in (overrides or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.RUN_ID = report["run_id"]

    corrected = run_q1_posthoc_corrections(
        run_root=run_root,
        primary_model=cfg.PRIMARY_MODEL_NAME,
        bootstrap_repetitions=cfg.BOOTSTRAP_REPS,
        permutation_repetitions=max(10000, int(cfg.BOOTSTRAP_REPS) * 10),
        seed=cfg.SEED,
        overwrite_primary_tables=True,
    )
    _patch_reproducibility_pkl(run_root, corrected)

    report["reporting_version"] = "q1-posthoc-v1"
    report["q1_readiness"] = corrected["readiness"]
    report["q1_corrected_without_retraining"] = True

    corrected_push_status = push_to_github(run_root, cfg, {"root": run_root})
    report["q1_corrected_github_push_status"] = corrected_push_status
    if corrected_push_status.get("success"):
        report["github_push_status"] = True
        report["final_commit_sha"] = corrected_push_status.get("commit_sha")

    write_json(run_root / "manifest.json", artifact_manifest(run_root))
    write_json(run_root / "final_verification.json", report)
    return report


__all__ = ["run_complete_pipeline"]
