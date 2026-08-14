# DFU Reliability Runbook

## Main notebook

Open `notebooks/DFU_Reliability_Final_OneCell.ipynb` in Colab, select a GPU runtime and run the single cell.

The notebook does not require a GitHub token. It clones the source branch publicly, mounts Google Drive, verifies a real write/read operation and starts the locked experiment. Re-running the same notebook resumes each incomplete model/seed/fold trial from `last_resume.pt`.

## Expected workload

- 45 training trials.
- Additional inference-only robustness evaluations for the 15 primary-model trials.
- Grad-CAM after all OOF predictions are complete.

The code does not automatically start a larger experiment, architecture search or hyperparameter sweep.

## Completion check

The run is complete only when `FINAL_RELIABILITY_VERIFICATION.json` reports:

- `completed_trials = 45`
- 45 best checkpoints
- 45 exact resume checkpoints
- 45 portable FP16 checkpoints
- 45 calibration prediction files
- 45 outer-test prediction files
- 45 metrics files
- verified secondary backup
- `verification_passed = true`

## Regeneration

Run `DFU_Reliability_Regenerate_OneCell.ipynb` to rebuild summary tables and figures from saved CSV/JSON artifacts. It does not train or perform checkpoint inference.

## External validation

Create:

`/content/drive/MyDrive/DFU-ImageGuard/external/external_manifest.csv`

Required columns:

- `image_path`
- `label` where Normal=0 and DFU=1

Recommended columns:

- `image_id`
- `dataset`
- `patient_id`

Then run `DFU_Reliability_External_OneCell.ipynb`. Use only a clinically and semantically compatible binary task.

## GitHub result sync

After final verification, run `DFU_Reliability_Sync_GitHub_OneCell.ipynb`. This does not retrain. It uses an existing Git credential or a valid `GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_PAT` when available. Authentication failure does not damage the two verified Drive copies. The full checkpoint export is intentionally separate because it can be very large; Drive remains the primary recovery source.
