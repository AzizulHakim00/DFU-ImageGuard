# DFU-PolarMorphNet Project

## Architectural contribution

DFU-PolarMorphNet is a weakly supervised lesion-centred architecture for binary DFU recognition. It replaces a generic rectangular-image attention head with six linked mechanisms:

1. morphology-preserving gated depthwise blocks;
2. weak lesion discovery from feature gradients;
3. differentiable lesion-centred polar tokenization;
4. bidirectional radial sequence modelling from centre to surrounding tissue and back;
5. cyclic contour modelling for closed-boundary topology;
6. radial–contour–global fusion with background-adversarial suppression.

The weak lesion map is not a clinical segmentation mask and must not be described as localization ground truth.

## Primary notebook

Open:

`notebooks/DFU_PolarMorphNet_Complete_OneCell.ipynb`

The notebook contains one Markdown cell and one executable cell. It:

- installs dependencies;
- mounts Google Drive;
- clones the current GitHub branch;
- downloads and audits the public dataset;
- creates duplicate-group-safe five-fold assignments;
- trains each missing outer fold once;
- writes one checkpoint after every improved epoch;
- reuses valid checkpoints when `FORCE_RETRAIN=False`;
- prints live epoch tables;
- displays cumulative fold metrics and figures;
- fits temperature and threshold only on the inner calibration split;
- creates small-scale weak-map XAI for held-out fold images;
- saves OOF predictions, metrics, figures, calibration JSON and PKL;
- pushes GitHub-safe run artifacts.

## Run ID and multiple Colab accounts

Use a fixed ID:

```python
'RUN_ID': 'POLAR_PRIMARY_2026'
```

Every Colab session must use the same repository revision and the same run ID. Resume works only when the session can access the same saved run directory. Different Google accounts do not automatically share `MyDrive`; either:

- use the same Drive account;
- place the project folder in a Drive location shared with the second account and update `DRIVE_ROOT` accordingly; or
- copy the completed run into the identical run path before resuming.

A checkpoint is reused only when it loads successfully and contains the `DFU-PolarMorphNet` architecture tag.

## Saved files

```text
/content/drive/MyDrive/DFU-ImageGuard/runs/<RUN_ID>/
├── models/dfu_polarmorphnet_fold_1.pt ... fold_5.pt
├── predictions/oof_fold_1.csv ... oof_fold_5.csv
├── predictions/all_oof_predictions.csv
├── tables/fold_1_metrics.csv ... fold_5_metrics.csv
├── tables/fold_and_oof_metrics.csv
├── tables/small_xai_metadata.csv
├── logs/history_polarmorph_fold_1.csv ... fold_5.csv
├── configs/fold_1_calibration.json ... fold_5_calibration.json
├── figures/
├── xai/
├── dfu_polarmorphnet_complete_reproducibility.pkl
├── software_versions.json
├── manifest.json
└── final_verification.json
```

The PKL does not store raw images.

## No-retraining regeneration

Use:

`notebooks/DFU_PolarMorphNet_Regenerate_From_Saved.ipynb`

It reads `all_oof_predictions.csv` and `fold_and_oof_metrics.csv` to recreate final figures. It does not train or load the model.

## Current evaluation scope

The primary run is one seed with five duplicate-group-safe outer folds. Fold variability is not multi-seed stability. High-impact journal evidence still requires separately trained seeds, architecture ablations and genuine external validation.

## Required future ablations

- image-centred vs learned lesion-centred coordinates;
- Cartesian pooling vs polar tokenizer;
- unidirectional vs bidirectional radial mixer;
- ordinary vs cyclic contour mixer;
- without background-adversarial branch;
- without global branch;
- without mask regularization;
- full DFU-PolarMorphNet.

Every ablation is a separately trained model variant. Results must never be generated post hoc from the full checkpoint.
