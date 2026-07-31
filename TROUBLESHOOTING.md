# DFU-ImageGuard Troubleshooting

## Recover from an error in an already-open Colab

An already-open notebook can keep old imported Python modules in memory even after the repository is updated. Use this sequence:

1. In Colab, select **Runtime → Restart session**.
2. Reopen the current notebook from the repository badge or refresh the notebook page.
3. Confirm that the notebook still contains one Markdown cell and one executable code cell.
4. Run the single executable cell.

The updated cell hard-resets `/content/DFU-ImageGuard` to `origin/main` and removes cached `src` modules before importing the pipeline.

## Google Drive mount failure

The pipeline tries a normal mount and then a forced remount. It verifies the actual filesystem mount rather than trusting the existence of `/content/drive/MyDrive`.

When Drive still cannot be mounted, the run continues under:

```text
/content/DFU-ImageGuard-local/runs/<RUN_ID>/
```

This local fallback is not persistent after the Colab runtime ends. Keep the runtime alive until GitHub export finishes, or repair Drive before starting a long publication run. Oversized checkpoints may remain storage-only because GitHub's file-size guard is enforced.

Common Drive recovery actions:

- allow pop-ups and third-party cookies for Colab and Google authentication;
- ensure the Google account has available Drive storage;
- disconnect duplicate Colab sessions using the same Drive mount;
- use **Runtime → Disconnect and delete runtime**, reconnect, and run the updated notebook when a normal restart does not fix authentication.

## Hugging Face unauthenticated warning

The warning about unauthenticated Hugging Face Hub requests is not a model error. Pretrained weights can still download. To remove the warning and improve rate limits, create a Colab Secret named `HF_TOKEN` and enable notebook access. The notebook reads it without printing it.

## Interrupted training

`FORCE_RETRAIN=False` is the default. A valid completed checkpoint is loaded without downloading pretrained weights again. Missing, empty, or invalid checkpoints are quarantined and only that fold is retrained. Completed primary folds are recorded progressively.

## Required final checks

A completed primary run is accepted only when all of the following hold:

- five group-disjoint outer folds are valid;
- exactly five primary calibration records exist;
- all five `dfu_imageguard_fold_*.pt` files contain a non-empty `model_state_dict`;
- the reproducibility PKL exists and is non-empty;
- `final_verification.json` is written.
