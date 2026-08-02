# Reliable DFU Framework

## Scientific objective

This branch stops architecture search and evaluates a fixed reliability-oriented DFU screening pipeline.

Primary model: `convnextv2_tiny`

Controlled baselines:

- `mobilenetv3_large`
- `densenet121`

Primary internal experiment:

- five duplicate-group-safe outer folds;
- three fixed seeds: 2026, 2027 and 2028;
- separate training, selection and calibration partitions inside every outer-training split;
- untouched outer-test inference;
- temperature scaling fitted only on calibration data;
- threshold selected only on calibration data to target sensitivity of at least 0.95.

## Persistence policy

### Primary Google Drive

`/content/drive/MyDrive/DFU-ImageGuard/runs/RELIABLE_DFU_CV_V1`

The primary run directory is authoritative and retains every trial's:

- `last_resume.pt` with model, optimizer, scheduler, scaler, epoch and early-stopping state;
- `best_model.pt`;
- `best_model_portable_fp16.pt`;
- history, predictions and metrics.

### Storage-aware secondary backup

`/content/drive/MyDrive/DFU-ImageGuard-Backup/runs/RELIABLE_DFU_CV_V1`

The earlier full-mirror implementation duplicated the complete run after every epoch and could exhaust Google Drive quota. Backup layout version 2 migrates that legacy mirror automatically.

During an active epoch it keeps one rolling copy of the current trial's:

- `last_resume.pt`;
- `best_model.pt`;
- `history.csv`.

After a trial completes, the secondary backup keeps:

- `best_model_portable_fp16.pt`;
- predictions, metrics, histories, tables, figures, manifests and reproducibility files.

Full checkpoints remain in the primary run. Secondary backup shortage is recorded as `DEGRADED` in `SECONDARY_BACKUP_STATUS.json` but no longer terminates training or deletes primary artifacts.

## Resume behavior

Completed trials are detected by `COMPLETE.json` plus `test_predictions.csv` and skipped. An interrupted trial loads `last_resume.pt` and continues from its next epoch. The one-cell notebook deletes cached `src.*` Python modules before importing the freshly cloned branch, ensuring fixes are loaded even when the cell is rerun in the same runtime.

## Reliability outputs

The final no-retraining reporting stage creates:

- fold/seed metrics;
- pooled OOF predictions;
- calibrated model summaries;
- selective-prediction tables at 100%, 95%, 90% and 80% coverage;
- false-negative and false-positive audits;
- paired duplicate-group bootstrap comparisons;
- reproducibility PKL and artifact manifest.

## External validation contract

External evaluation must use a manifest with `image_path`, `image_id` and `label`. It must not retrain models, select models, tune thresholds or refit calibration temperatures on external labels.
