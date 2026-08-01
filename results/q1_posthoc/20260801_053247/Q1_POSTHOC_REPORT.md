# DFU-ImageGuard Q1 Post-hoc Report

Q1 high-impact ready: **False**

Primary rank: **3**; nominal best model: **MobileNetV3**.

Supported claim: Competitive duplicate-group-aware OOF discrimination with high specificity; no statistically supported superiority or deployment claim.

## Corrected metrics

| model           | accuracy | balanced accuracy | sensitivity | specificity | F1 | MCC | ROC-AUC | PR-AUC | Brier | ECE | FN | FP |
|:----------------|---------:|------------------:|------------:|------------:|---:|----:|--------:|-------:|------:|----:|---:|---:|
| MobileNetV3     | 0.9858 | 0.9855 | 0.9746 | 0.9963 | 0.9852 | 0.9717 | 0.9958 | 0.9970 | 0.0205 | 0.0297 | 13 | 2 |
| DFU-ImageGuard  | 0.9744 | 0.9738 | 0.9531 | 0.9945 | 0.9731 | 0.9495 | 0.9929 | 0.9910 | 0.0347 | 0.0476 | 24 | 3 |
| DenseNet121     | 0.9744 | 0.9738 | 0.9531 | 0.9945 | 0.9731 | 0.9495 | 0.9959 | 0.9929 | 0.0167 | 0.0224 | 24 | 3 |
| Linear-LogReg   | 0.9735 | 0.9730 | 0.9590 | 0.9871 | 0.9723 | 0.9472 | 0.9979 | 0.9983 | 0.0132 | 0.0134 | 21 | 7 |
| EfficientNet-B0 | 0.9659 | 0.9653 | 0.9453 | 0.9853 | 0.9641 | 0.9323 | 0.9957 | 0.9963 | 0.0180 | 0.0142 | 28 | 8 |
| ResNet18        | 0.9640 | 0.9631 | 0.9336 | 0.9926 | 0.9618 | 0.9293 | 0.9946 | 0.9946 | 0.0468 | 0.0842 | 34 | 4 |

## Duplicate-group-aware accuracy comparisons

| comparison | primary minus baseline | 95% CI | group permutation p | Holm p |
|:--|--:|:--|--:|--:|
| vs DenseNet121 | 0.0002 | -0.0124 to 0.0125 | 1.0000 | 1.0000 |
| vs EfficientNet-B0 | 0.0088 | -0.0047 to 0.0216 | 0.3054 | 0.9161 |
| vs Linear-LogReg | 0.0013 | -0.0112 to 0.0135 | 0.8542 | 1.0000 |
| vs MobileNetV3 | -0.0114 | -0.0232 to 0.0000 | 0.0843 | 0.3372 |
| vs ResNet18 | 0.0106 | -0.0020 to 0.0238 | 0.0469 | 0.2345 |

## Mandatory remaining evidence

- genuine independent external validation with overlap audit
- pre-registered five-seed final comparison
- visual review of XAI overlays
- independent error/label review
- dataset licensing clarification

## Blocked claims

- state of the art
- significantly outperforms all baselines
- patient-level validation
- external generalisation
- clinical deployment readiness
- XAI proves clinical correctness
