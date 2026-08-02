# DFU Reliability Final Protocol

## Scientific position

Architecture search is closed. The failed DFU-RadialAdapter pilot remains a negative development experiment and is not promoted to full cross-validation. The final study evaluates a fixed, pretrained ConvNeXtV2-Tiny primary classifier against MobileNetV3-Large and DenseNet121 under one locked reliability protocol.

The paper claim is reliability-focused rather than architecture-superiority-focused:

> Leakage-aware and calibration-focused diabetic-foot-ulcer screening under uncertainty, image corruption and domain shift.

## Primary experiment

- Dataset source: `laithjj/diabetic-foot-ulcer-dfu`.
- Labels: only explicit immediate class folders under the unique `Patches` directory.
- Exclusions: unreadable files and duplicate clusters containing conflicting labels.
- Split unit: exact/pixel/pHash/embedding-assisted duplicate group.
- Outer validation: five locked `StratifiedGroupKFold` folds.
- Seeds: 2026, 2027 and 2028.
- Models: ConvNeXtV2-Tiny, MobileNetV3-Large and DenseNet121.
- Total trials: 3 models × 3 seeds × 5 folds = 45.
- Patient-level validation is not claimed because patient identifiers are unavailable.

For each outer fold, the outer-training groups are divided into disjoint training, model-selection and calibration groups. The selection partition is used only for early stopping. The calibration partition is used only for temperature scaling and a sensitivity-constrained decision threshold. The outer test fold is used once for final inference.

## Model and optimization

All three models use ImageNet-pretrained timm backbones, global average pooling, dropout and a single binary logit. The backbone is frozen for two epochs, then fine-tuned. BCEWithLogitsLoss uses a positive-class weight computed from that trial's training partition only. AdamW uses separate learning rates for backbone and head parameters. Model selection is lexicographic: higher selection ROC-AUC, then lower selection log-loss when AUC is tied.

Each epoch writes `last_resume.pt` containing model, optimizer, scheduler and AMP-scaler states. The best checkpoint is selected only from the inner selection partition.

## Calibration and decision policy

Temperature is fitted on the calibration partition only. A threshold is selected to maximize specificity subject to calibration sensitivity ≥0.95. If no threshold meets that constraint, the prespecified fallback maximizes calibration balanced accuracy. The frozen temperature and threshold are applied to the outer test fold.

Reported metrics include sensitivity, specificity, balanced accuracy, F1, F2, MCC, ROC-AUC, PR-AUC, Brier score, log-loss, ECE, MCE, calibration slope/intercept and confusion counts.

## Selective prediction

For coverages 100%, 95%, 90% and 80%, cases are retained in descending calibrated confidence. The study reports retained sensitivity, specificity, balanced accuracy, error rate and number referred for expert review. Coverage analysis is descriptive and does not retune the classifier.

## Statistical comparison

Primary-vs-baseline differences are estimated separately within each seed using paired duplicate-group bootstrap resampling. The bootstrap pairs predictions by image and samples duplicate groups with replacement. The study reports 95% intervals and the probability that the primary model is better. Seed-level results remain the primary repeated-experiment unit; pooled repeated-seed rows are not treated as independent patients.

## Robustness

The primary ConvNeXtV2 model is evaluated without retraining under fixed mild/moderate brightness, contrast, blur, JPEG, noise, rotation and central occlusion corruptions. The same fold-specific temperature and threshold are retained. Clean-to-corrupted sensitivity and balanced-accuracy drops are reported.

## Explainability and error review

Grad-CAM is produced for a prespecified small sample of false negatives, false positives, uncertain correct cases and confident correct cases from seed 2026. Grad-CAM is described only as a model-attention visualization, not lesion segmentation or a clinical explanation.

The pipeline creates a two-reviewer error-review template for false negatives and false positives. Categories such as small lesion, low contrast, blur and label uncertainty are intentionally left blank for human review; they are never fabricated automatically.

## External validation

The external notebook requires an explicit compatible binary manifest with `image_path` and `label`. All 15 frozen ConvNeXtV2 models are applied without retraining, temperature refitting or threshold retuning. Each model uses its own frozen calibration temperature and threshold; the external ensemble is a majority vote of those frozen decisions. Task or label incompatibility must be documented rather than silently relabelled.

## Persistence

Primary artifacts are stored under:

`/content/drive/MyDrive/DFU-ImageGuard/runs/DFU_RELIABILITY_FINAL_V1`

Every trial is mirrored and SHA-256 verified under:

`/content/drive/MyDrive/DFU-ImageGuard-Backup/runs/DFU_RELIABILITY_FINAL_V1`

GitHub result export is a separate no-retraining notebook. Full `.pt` files are split into reconstructable chunks with manifests; portable FP16 checkpoints are copied normally. Because the complete 45-trial checkpoint archive is large, the two SHA-256-verified Drive trees remain the primary recovery copies and GitHub synchronization is never allowed to block training.

## Required interpretation

A high internal score alone does not justify a Q1 claim. Q1 suitability requires stable multi-seed results, useful calibration/selective-prediction evidence, a valid frozen external evaluation and transparent limitations. No code path manufactures performance numbers or declares clinical deployment readiness.
