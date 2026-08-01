# DFU-RadialAdapter Pilot Repair Package

This package replaces the failed high-complexity architecture workflow with a **performance-first, falsifiable pilot**.

## What it runs

- One locked duplicate-group-aware outer fold (fold 1)
- Two fixed training seeds: `2026`, `2027`
- Same ConvNeXtV2-Tiny feature backbone for both models
- Model A: plain ConvNeXtV2-Tiny binary baseline
- Model B: ConvNeXtV2-Tiny + zero-gated lesion-centred radial/cyclic difference adapter
- Separate inner train, early-stopping selection, and calibration partitions
- Temperature scaling and sensitivity-constrained threshold selection
- Raw-versus-calibrated Brier, log-loss, ECE, and MCE tables
- Threshold-aware risk–coverage and sensitivity–coverage analysis
- Paired duplicate-group bootstrap intervals for adapter-minus-baseline differences
- Live epoch tables, strict-fold metrics, calibration plots, ROC/PR plots, and small model-native weak-coordinate overlays
- Automatic `FAIL`, `CONDITIONAL_PASS`, or `STRONG_PASS` verdict
- **No automatic full five-fold experiment**

## Why this is safer

```text
final_logit = baseline_logit + tanh(alpha) * adapter_logit
alpha = 0
```

The proposed model begins with exactly the baseline output. The pilot tests whether the adapter learns useful evidence without materially degrading balanced accuracy, sensitivity, ROC-AUC, or calibration.

## Main notebook

```text
notebooks/DFU_RadialAdapter_Pilot_OneCell.ipynb
```

It clones branch `radial-adapter-pilot-v1` directly from GitHub. It refuses to train unless the primary Drive, secondary Drive backup, and `GITHUB_TOKEN` secret are available.

## Backup policy

- Primary exact copy: `/content/drive/MyDrive/DFU-ImageGuard/runs/RADIAL_ADAPTER_PILOT_V1`
- Secondary exact mirror: `/content/drive/MyDrive/DFU-ImageGuard-Backup/runs/RADIAL_ADAPTER_PILOT_V1`
- GitHub results branch: `radial-pilot-results`
- Exact full checkpoints are split into `<checkpoint>.chunks/part-XXXX.bin` files below GitHub's normal per-file limit.
- Portable FP16 inference checkpoints, CSV, JSON, PKL, figures, logs, and manifests are also exported.
- GitHub export runs after each completed model/seed trial and again at finalization.
- The run fails if required backup verification fails.

## Saved trial files

- `last_resume.pt`: model, optimizer, scheduler, scaler and completed epoch
- `best_model.pt`: exact full-precision best model
- `best_model_portable_fp16.pt`: portable inference checkpoint
- `history.csv`
- `calibration_predictions.csv`
- `test_predictions.csv`
- `metrics.json`
- `checkpoint_manifest.json` with SHA-256 hashes
- XAI overlays and metadata

## Restore from GitHub

Use:

```text
notebooks/DFU_RadialAdapter_Restore_Checkpoint_OneCell.ipynb
```

The restore code concatenates the checkpoint chunks and verifies the reconstructed file's exact byte count and SHA-256 hash.

## Go/no-go rules

Non-inferiority requires:

- balanced accuracy delta ≥ `-0.005`
- sensitivity delta ≥ `-0.010`
- ROC-AUC delta ≥ `-0.005`
- ECE deterioration ≤ `+0.010`
- active adapter gate and non-zero contribution

Possible verdicts:

- `FAIL_DO_NOT_RUN_FULL_CV`
- `CONDITIONAL_PASS_NEEDS_ONE_MORE_PILOT_FOLD`
- `STRONG_PASS_ELIGIBLE_FOR_FULL_CV`

This pilot is not Q1 publication evidence. It only determines whether the architecture deserves the expensive grouped multi-seed final experiment.
