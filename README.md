# DFU-ImageGuard

[![Open complete Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/DFU-ImageGuard/blob/main/notebooks/DFU_ImageGuard_Conference_Complete.ipynb)
[![Load saved artifacts](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/DFU-ImageGuard/blob/main/notebooks/DFU_ImageGuard_Load_Artifacts.ipynb)
[![Upload an existing run](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/DFU-ImageGuard/blob/main/notebooks/DFU_ImageGuard_Upload_Existing_Run.ipynb)

DFU-ImageGuard is a Google Colab–compatible, leakage-safe research pipeline for binary diabetic foot ulcer image classification. It performs nested duplicate-group-aware five-fold out-of-fold evaluation, fold-specific calibration, uncertainty analysis, robustness testing, XAI, group bootstrap confidence intervals, statistical comparisons, progressive Google Drive saving, and size-safe GitHub export.

It does **not** contain precomputed or fabricated results. Running the complete notebook creates the evidence.

## Critical dataset policy

The pipeline automatically downloads the public Kaggle dataset `laithjj/diabetic-foot-ulcer-dfu` with `kagglehub`. It deliberately includes only the `Patches` directory when that directory contains explicit immediate `Normal` and `Abnormal/Ulcer` class folders.

It does not infer labels from root-folder keywords and does not train on ambiguous folders such as `Original Images`, `TestSet`, or transfer-learning folders. Unknown class-folder layouts cause a hard stop instead of guessed labels.

The Kaggle data card does not declare a license. The run records this as a limitation; users must verify redistribution and downstream-use permissions themselves.

## Evaluation design

```text
Strict class-folder audit
→ corrupt-image screening
→ SHA-256 file and pixel hashes
→ pHash near-duplicate detection
→ conservative embedding-assisted duplicate candidates
→ duplicate-group locked StratifiedGroupKFold (5 folds)
→ inner train / selection / calibration partition inside each outer-training fold
→ exactly one DFU-ImageGuard fit per outer fold
→ fold-only temperature scaling and threshold selection
→ untouched outer-fold inference
→ OOF aggregation, confidence intervals, statistics, uncertainty, robustness, XAI
```

Patient/case identifiers are not provided by this dataset. Therefore, the project does not claim patient-level splitting. It reports duplicate-group-aware splitting and preserves this limitation in every completed run.

## Primary model

- ImageNet-pretrained ConvNeXt-Tiny backbone
- lesion-aware channel and spatial attention
- generalized-mean pooling
- dropout and binary classification head
- modular construction for later, separately trained ablations

The complete primary experiment creates exactly:

```text
dfu_imageguard_fold_1.pt
...
dfu_imageguard_fold_5.pt
```

`FORCE_RETRAIN=False` by default. Completed fold checkpoints are reused after interruption. Calibration inspection, plotting, XAI, uncertainty and robustness reuse trained checkpoints and do not fit the primary model again.

## Baselines

The same outer folds and leakage controls are used for:

- ResNet18
- DenseNet121
- MobileNetV3
- EfficientNet-B0
- Logistic Regression on frozen ImageNet ResNet18 embeddings

## Colab use

1. Open `notebooks/DFU_ImageGuard_Conference_Complete.ipynb`.
2. In Colab, open **Secrets** and add `GITHUB_TOKEN` with repository write permission.
3. Run the notebook’s single executable cell.
4. Approve the Google Drive mount.

The public Kaggle dataset is downloaded programmatically. No desktop download or Colab upload is required.

The default complete run is computationally substantial because it trains one proposed model and five baselines across the same five outer folds. Adjusting epochs for a smoke test is possible, but smoke-test results must not be reported as publication results.

## Google Drive layout

```text
/content/drive/MyDrive/DFU-ImageGuard/runs/<RUN_ID>/
├── tables/
├── figures/
├── models/
├── xai/
├── predictions/
├── logs/
├── configs/
├── manifests/
├── dfu_imageguard_complete_reproducibility.pkl
├── paper_results.json
├── dataset_manifest.json
├── split_integrity_report.json
├── software_versions.json
├── MODEL_CARD.md
├── LIMITATIONS.md
├── EXTERNAL_VALIDATION_STATUS.md
├── manifest.json
└── github_push_status.json
```

The PKL contains configuration, seeds, dataset metadata and content hash, exact fold/group assignments, duplicate screening, integrity reports, raw and calibrated OOF predictions, baseline predictions, calibrators, thresholds, metrics, confidence intervals, statistical comparisons, uncertainty, robustness, XAI metadata, paths, versions, limitations and warnings. Raw images are never stored in the PKL.

Only load a PKL that you created or otherwise trust.

## Metrics and outputs

The pipeline reports accuracy, error rate, balanced accuracy, PPV, sensitivity, specificity, NPV, F1, F2, MCC, Cohen’s kappa, ROC-AUC, PR-AUC, log loss, Brier score, ECE, MCE, calibration slope/intercept, FPR, FNR and confusion counts. Raw and calibrated tables are saved separately within the combined metrics file.

Figures include class distribution, duplicate audit, persisted learning curves, ROC and PR curves, raw and normalized confusion matrices, metric and error comparisons, raw and calibrated reliability diagrams, calibration-loss comparisons, confidence distributions, fold stability, risk–coverage, robustness curves, error galleries and XAI overlays. Figures are saved as 300-DPI PNG and PDF.

## Calibration and threshold safeguards

For each outer fold:

- early stopping uses only the inner selection partition;
- temperature scaling uses only the disjoint inner calibration partition;
- the clinical threshold is selected only from the inner calibration partition;
- the outer test fold never controls preprocessing, augmentation, stopping, calibration or thresholding.

The default threshold rule maximizes specificity subject to calibration sensitivity of at least 0.95. When no candidate satisfies that target, the code records a fallback to maximum balanced accuracy. It never silently forces a target unsupported by the data.

## XAI safeguards

Grad-CAM, Grad-CAM++, occlusion sensitivity, SHAP and LIME are attempted for representative OOF cases. Every selected image is explained with the checkpoint from the outer fold that did not train on that image. Method failures are logged rather than replaced with fake outputs.

XAI is not presented as proof of causality, lesion localization accuracy or clinical correctness.

## External validation, ablation, federated and multimodal work

- A random internal holdout is never called external validation.
- External evaluation requires a scientifically compatible independent dataset and frozen preprocessing, weights, calibration and threshold.
- Ablation requires separately training modified variants.
- Multi-seed stability requires additional fits; five-fold variation is not multi-seed stability.
- Federated learning is only a labelled simulation unless genuine hospital/site identities exist.
- Multimodal analysis is disabled unless legitimate structured clinical metadata accompany the images.

Future notebooks document these boundaries without fabricating hospitals, patients, modalities, clinical variables or results.

## GitHub auto-export

At the end of a completed run, the code:

- reads `GITHUB_TOKEN` from Colab Secrets without printing it;
- copies GitHub-safe artifacts to `results/runs/<RUN_ID>/`;
- updates `results/LATEST_RUN.txt`;
- checks every file against a 95-MB safety threshold;
- retains oversized artifacts in Drive and records their paths and SHA-256 values;
- commits and pushes results;
- saves the push result and commit SHA;
- preserves the complete Drive run even if GitHub export fails.

## Reproducibility verification

The final cell prints and saves:

- run ID;
- dataset class counts;
- valid fold count;
- duplicate/leakage audit status;
- proposed fits executed in the current session;
- valid primary checkpoint count;
- PKL and Drive paths;
- GitHub push status;
- final commit SHA when successful.

## Research-use warning

This is retrospective research software, not a medical device. Do not describe internal classification performance as clinical readiness, do not claim state of the art without a directly matched comparison, and do not use predictions to replace qualified clinical assessment.
