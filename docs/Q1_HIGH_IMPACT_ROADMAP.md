# DFU-ImageGuard: Q1 High-Impact Evidence Roadmap

## Completed without retraining

- Corrected fold metrics when a stored threshold equals 1.0.
- Replaced `max(p, 1-p)` as the primary confidence measure with operating-threshold-aware log-odds distance.
- Added threshold-aware risk–coverage and sensitivity–coverage analysis.
- Added 1,000 duplicate-group bootstrap confidence intervals.
- Added 10,000 group sign-flip permutations for paired accuracy comparisons.
- Treats image-level McNemar as supplemental because members of duplicate clusters are dependent.
- Added automatic claim blocking when the proposed model is not demonstrably superior.
- Added `paper_results_q1_corrected.json`, `Q1_POSTHOC_REPORT.md`, corrected tables and corrected figures.

## Current evidence status

The current run is a strong internal, duplicate-group-aware proof of concept. It is not yet sufficient for a high-impact Q1 clinical or medical-imaging claim because:

1. patient identifiers are unavailable;
2. genuine independent external validation has not been performed;
3. only one training seed was used;
4. MobileNetV3 is nominally the strongest internal model;
5. dataset licensing is undeclared;
6. XAI overlays require visual and preferably clinical review.

## Mandatory next experiments

### Multi-seed final comparison

Freeze the folds, preprocessing, threshold rule and model definitions. Run seeds `2026, 2027, 2028, 2029, 2030` for the proposed model and strongest baseline. Report mean ± SD, duplicate-group bootstrap intervals, paired effect sizes and multiplicity-adjusted tests. This requires new training and must remain separate from five-fold stability.

### Genuine external validation

Use a scientifically compatible independent DFU-versus-normal dataset only after exact-hash, perceptual-hash and embedding-overlap screening. Freeze weights, preprocessing, temperatures and thresholds. Do not retrain, recalibrate or retune on the external test set.

### Error and label review

Review all false negatives, false positives, high decision-confidence errors and ambiguous source images. Record possible label errors, acquisition artefacts, ulcer visibility and clinically relevant failure categories. Do not infer sensitive attributes from images.

### XAI verification

Use Grad-CAM++ and occlusion sensitivity as primary qualitative methods. Inspect overlays for background shortcuts. Without lesion masks, do not claim localization accuracy. Add randomization sanity checks or blinded relevance ratings where feasible.

## Defensible claim strategy

The contribution should be framed as a leakage-aware and reliability-oriented DFU evaluation framework, not unqualified architectural superiority. Until external and multi-seed evidence is complete, use language such as:

> DFU-ImageGuard achieved competitive duplicate-group-aware out-of-fold discrimination with high specificity, while paired uncertainty intervals did not establish superiority over all evaluated baselines.

Blocked claims include state of the art, clinical deployment readiness, patient-level validation, external generalization and XAI-proven clinical correctness.
