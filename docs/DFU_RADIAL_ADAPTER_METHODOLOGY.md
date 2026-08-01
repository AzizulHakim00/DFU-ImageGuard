# Methodology and Decision Rules

## Research objective

Determine whether a minimal lesion-centred radial morphology adapter adds useful DFU-specific inductive bias to a strong pretrained ConvNeXtV2-Tiny model without causing the performance collapse observed with multi-head, multi-loss custom architectures.

## Architecture under test

The shared backbone returns intermediate and final spatial features. The final feature map produces the ordinary global classification logit. The intermediate feature map is projected to a compact representation, then a differentiable spatial evidence distribution estimates a weak image-specific centre and bounded radial scale.

Five rings and sixteen angular sectors are sampled around the learned centre. Adjacent-ring differences encode centre-to-boundary transitions. Circular adjacent-sector differences encode contour variation while preserving wrap-around topology. Small depthwise mixers summarize the two difference families. A scalar adapter logit is added to the global baseline logit through a zero-initialized learnable gate.

The adapter adds no prototype head, transformer, state-space model, adversarial head, quality head, reconstruction loss, or contrastive objective in the pilot.

## Fair comparison

Both models use the same ConvNeXtV2-Tiny pretrained family, locked manifest, duplicate groups, outer test fold, inner partitions, augmentations, optimizer policy, early stopping, calibration policy, and training seeds.

The adapter model creates the shared backbone and global head before adapter initialization. Tests verify that its initial final logits equal its baseline logits exactly.

## Data separation

The existing strict repository pipeline is reused:

1. explicit class folders under `Patches` only;
2. corrupt-image exclusion;
3. exact file hash;
4. exact pixel hash;
5. perceptual hash;
6. embedding-plus-perceptual grouping;
7. stratified duplicate-group outer split;
8. group-disjoint inner train, selection, and calibration partitions.

The outer test fold is never used for early stopping, temperature fitting, or threshold selection.

## Training

- Epochs 1–2: backbone frozen
- Epoch 3 onward: backbone unfrozen at low learning rate
- Backbone LR: `1e-5`
- Baseline head LR: `1e-4`
- Adapter LR: `2e-4`
- Gate LR: `1e-3`
- Maximum epochs: `25`
- Early-stopping patience: `7`
- Loss: weighted binary cross-entropy only

No auxiliary loss is used, isolating whether the architecture itself helps.

## Calibration and threshold

The inner calibration partition fits scalar temperature scaling. The threshold maximizes specificity subject to sensitivity of at least 0.95, with balanced-accuracy fallback. The outer test fold uses the frozen temperature and threshold.

## Automatic decision

Non-inferiority requires:

- balanced-accuracy delta ≥ `-0.005`
- sensitivity delta ≥ `-0.010`
- ROC-AUC delta ≥ `-0.005`
- ECE delta ≤ `+0.010`
- active adapter gate and contribution

Meaningful benefit requires at least one:

- balanced-accuracy delta ≥ `+0.002`
- sensitivity delta ≥ `+0.005`
- ECE delta ≤ `-0.005`

A strong pass only makes the architecture eligible for the final five-fold multi-seed study; it does not prove superiority or Q1 suitability.
