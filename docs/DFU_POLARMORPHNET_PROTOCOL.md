# DFU-PolarMorphNet V2 Research Protocol

## Purpose

DFU-PolarMorphNet is a separate V2 project. The previous DFU-ImageGuard checkpoints remain a V1 baseline; they cannot represent the new architecture. V2 therefore requires a new five-fold primary training run.

The architecture is designed around a disease-specific inductive bias rather than stacking standard attention modules:

```text
RGB image
→ morphology-preserving stem
→ weak lesion discovery
→ differentiable lesion-centred polar tokenization
→ bidirectional radial selective state-space encoder
→ cyclic contour mixer
→ radial/contour/global fusion
→ counterfactual background suppression
→ DFU probability
```

### Architectural components

1. **Morphology-preserving stem** — gated local and dilated depthwise responses.
2. **Weak lesion discovery** — learns a soft lesion map from image labels without claiming segmentation ground truth.
3. **Learned polar geometry** — estimates lesion centre and scale, then samples radial rings and angular sectors differentiably.
4. **Bidirectional radial SSM** — models centre-to-periphery and periphery-to-centre transitions.
5. **Cyclic contour mixer** — models a closed lesion-boundary sequence with circular padding.
6. **Adaptive morphology fusion** — combines radial, contour and global evidence per image.
7. **Counterfactual background suppression** — swaps backgrounds within a batch and adversarially discourages background-label shortcuts.

## Primary evaluation

- Strict public `Patches` class folders only; ambiguous folders are excluded.
- Corrupt-image screening.
- SHA-256 file and pixel hashes.
- pHash near-duplicate grouping.
- Optional pretrained embedding-assisted duplicate candidates.
- Duplicate/source-group-disjoint five-fold OOF evaluation.
- Inner training, model-selection and calibration partitions inside every outer-training fold.
- Exactly one primary V2 fit per outer fold unless `FORCE_RETRAIN=True` is explicitly set.
- Fold-specific temperature scaling and development-selected threshold.
- Outer-test folds never affect stopping, calibration or threshold selection.

## Colab notebooks

### Complete experiment

`notebooks/DFU_PolarMorphNet_Complete.ipynb`

The notebook has one Markdown cell and one executable code cell. Edit only the settings at the top:

```python
MODE = "train"
RUN_ID = ""
PREFERRED_FOLDS = ""
DRIVE_ROOT = "/content/drive/MyDrive/DFU-PolarMorphNet"
REPO_REF = "q1-posthoc-corrections-20260801"
FORCE_RETRAIN = False
```

### Multiple Colab accounts

All accounts must use the same accessible Drive folder. A Google Shared Drive is safest. A shared-folder shortcut in each account's MyDrive can also be used, but every account must resolve `DRIVE_ROOT` to the same underlying folder.

Example allocation:

```text
Account A: PREFERRED_FOLDS = "1,2"
Account B: PREFERRED_FOLDS = "3,4"
Account C: PREFERRED_FOLDS = "5"
```

The first account prints a RUN_ID. Copy that exact RUN_ID to the other accounts. Fold lease files prevent two accounts from intentionally training the same fold. Training progress and checkpoints are saved after every epoch/fold.

Do not run the same fold simultaneously from multiple accounts.

### Artifact-only regeneration

`notebooks/DFU_PolarMorphNet_Artifacts_Only.ipynb`

This mode performs no training. It loads the saved OOF predictions, calibrators and five checkpoints to regenerate:

- aggregate/fold metrics;
- group-bootstrap confidence intervals;
- ROC/PR/confusion plots;
- reliability and calibration figures;
- threshold-aware risk–coverage and sensitivity–coverage;
- decision-confidence distributions;
- small fixed-corruption robustness;
- held-out-fold weak-lesion-map and Grad-CAM figures;
- the reproducibility PKL and paper JSON.

### Upload existing run

`notebooks/DFU_PolarMorphNet_Upload_Existing_Run.ipynb`

This mode performs no training and only exports an existing Drive run to GitHub.

## Drive structure

```text
/content/drive/MyDrive/DFU-PolarMorphNet/runs/<RUN_ID>/
├── models/
│   ├── dfu_polarmorphnet_fold_1.pt
│   ├── ...
│   └── dfu_polarmorphnet_fold_5.pt
├── predictions/
│   ├── oof_fold_1.csv
│   ├── ...
│   ├── oof_fold_5.csv
│   ├── oof_embeddings_fold_*.npy
│   └── dfu_polarmorphnet_oof_predictions.csv
├── manifests/
├── configs/
├── tables/
├── figures/
├── xai/
├── robustness/
├── logs/
├── dfu_polarmorphnet_complete_reproducibility.pkl
├── paper_results.json
├── final_verification.json
├── MODEL_CARD.md
└── LIMITATIONS.md
```

## Live output

After every completed fold the notebook displays:

- fold sample counts;
- temperature and selected threshold;
- accuracy, balanced accuracy, sensitivity, specificity, F1, MCC, AUC, Brier and ECE;
- learning curve;
- calibrated confusion matrix.

When all five fold artifacts exist, the last account automatically performs final aggregation and displays the paper-ready figures.

## GitHub export

Add `GITHUB_TOKEN` to Colab Secrets. The token is read but never printed or saved.

The final run is pushed to:

```text
branch: polarmorphnet-results/<RUN_ID>
path: results/polarmorphnet/<RUN_ID>/
```

Checkpoints are retained in Drive and represented in GitHub by their paths, sizes and SHA-256 hashes. This avoids repository-size and GitHub-file-limit problems. GitHub failure does not damage the completed Drive run.

## Reproducibility rules

- `FORCE_RETRAIN=False` by default.
- Completed checkpoints and fold prediction CSVs are reused.
- Figures and tables can be recreated without retraining.
- Every scientific configuration is locked per RUN_ID.
- Fresh Colab accounts automatically remap saved relative dataset paths after programmatic redownload.
- Five-fold stability must not be called multi-seed stability.
- Future five-seed and ablation experiments must be separate runs.

## Required future high-impact evidence

The V2 primary pipeline is not enough by itself for a high-impact claim. The final manuscript still requires:

1. five-seed primary evaluation;
2. complete architectural ablation;
3. frozen scientifically compatible external validation;
4. manual/clinical false-negative and XAI review;
5. dataset licensing clarification;
6. matched comparisons against reproducible strong baselines.
