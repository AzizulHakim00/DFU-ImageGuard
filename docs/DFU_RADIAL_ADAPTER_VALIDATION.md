# Package Validation

The following checks were executed before publishing:

```text
Python source compilation                         PASS
One-cell pilot notebook JSON validation           PASS
One-cell pilot notebook Python compilation        PASS
One-cell regeneration notebook JSON validation    PASS
One-cell regeneration notebook Python compilation PASS
One-cell checkpoint restore notebook compilation  PASS
Radial adapter output-shape test                   PASS
Spatial attention normalization test               PASS
Zero-gate contribution test                        PASS
Gate-gradient activation test                      PASS
Baseline-preserving full-model initialization test PASS
Exact checkpoint chunk round-trip test             PASS
```

The full Colab experiment was not executed here because the Kaggle dataset, Google Drive, GPU runtime, pretrained weights, and user GitHub token are external runtime dependencies. No performance result is fabricated.

## Backup validation

- Primary and secondary Drive paths are required.
- Every mirrored file is verified by SHA-256.
- Full `.pt` files are split into GitHub-safe chunks and can be reconstructed byte-for-byte.
- GitHub export is executed after each completed model/seed trial and again at finalization.
- A required backup failure raises an error instead of reporting the experiment as successfully preserved.
