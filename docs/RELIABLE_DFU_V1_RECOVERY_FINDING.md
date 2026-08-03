# Reliable DFU V1 Recovery Finding

## Observed recovery audit

The non-destructive Drive search was executed against the mounted Google Drive account and found two folders named `RELIABLE_DFU_CV_V1`:

- primary: `/content/drive/MyDrive/DFU-ImageGuard/runs/RELIABLE_DFU_CV_V1`
- secondary: `/content/drive/MyDrive/DFU-ImageGuard-Backup/runs/RELIABLE_DFU_CV_V1`

The primary folder contained 16 small files totaling 3,604,699 bytes. The secondary folder was empty. Across both folders the search found:

- `COMPLETE.json`: 0
- `last_resume.pt`: 0
- `test_predictions.csv`: 0
- `FINAL_VERIFICATION.json`: 0
- recognized recoverable artifact hits: 0

## Scientific decision

The earlier notebook output proves that training iterations were executed, but console output is not a substitute for model checkpoints, test predictions, calibration parameters, or trial-completion manifests. Therefore:

- no V1 trial is counted as completed evidence;
- no V1 performance result is reported;
- V1 cannot be resumed from the currently mounted Drive account;
- the six apparently completed console trials must be rerun;
- V2 uses a new run ID and does not overwrite the V1 audit folder.

## V2 prevention

`RELIABLE_DFU_CV_V2` applies bounded checkpoint retention:

1. An incomplete active trial retains exact FP32 model, optimizer, scheduler and AMP resume state.
2. After training, the FP16 portable model, predictions and hashes are written and verified.
3. Only after verification are completed `last_resume.pt` and FP32 `best_model.pt` retired.
4. The secondary directory keeps one rolling active resume copy and metadata; it does not duplicate every completed model weight in the same Drive account.
5. Backup quota failures are recorded as degraded and cannot delete primary evidence or terminate training.

Google Drive Trash and other Google accounts are outside the mounted `MyDrive` search. A manually restored checkpoint may be audited separately, but no such artifact is assumed to exist.
