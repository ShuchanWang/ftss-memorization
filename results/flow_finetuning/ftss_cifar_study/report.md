# Pretrained CIFAR-10 RF-UNet: FTSS During Clean Fine-Tuning

State: complete

One public clean class-conditional checkpoint, no watermark or adapters. Fine-tuning and evaluation pools use CIFAR-10 test images, disjoint from the checkpoint's standard training split. The source checkpoint's complete training provenance is not independently audited. Nested class-balanced tuning pools and equal-size untouched reference pools are disjoint from a fixed 1000-image quality reference per subset seed. All evaluated weights are EMA (decay 0.999).

Sizes [50, 200, 1000]; subset seeds [0, 1, 2]; checkpoints [0, 300, 1000, 3000]. Full-parameter AdamW fine-tuning, LR 0.0001, cosine decay over 3000 steps, batch 64. FTSS uses 20 class-balanced trajectories, 20 injection times, 200 Euler steps, relative perturbation 0.1, and the original state/velocity clamps. Each trajectory and all its perturbations retain the same class condition. Generation uses 500 balanced class conditions, 100 Euler steps and the checkpoint's original sampling clamps. FTSS and generation use paired random inputs within each seed. Inception features provide quality, conditional within-class diversity and nearest-neighbor outcomes. Disjoint real-versus-real quality controls use matched reference/sample counts and are saved per seed. Pixel proximity is a secondary check. Preference is attraction, not proof of copying. Flow-loss gaps and membership AUC average eight noise/time draws.

| Images | Steps | Runs | FTSS / Before | Preference Change | Loss Gap Change | Coverage Change | FID / Before |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 300 | 3 | 0.9498 +/- 0.0102 | 0.0127 +/- 0.0137 | 0.0106 +/- 0.0006 | 0.0307 +/- 0.0054 | 0.965 +/- 0.008 |
| 50 | 1000 | 3 | 0.6369 +/- 0.0094 | 0.1193 +/- 0.0116 | 0.0946 +/- 0.0013 | 0.0040 +/- 0.0261 | 1.018 +/- 0.048 |
| 50 | 3000 | 3 | 0.2207 +/- 0.0111 | 0.4520 +/- 0.0854 | 0.4989 +/- 0.0249 | 0.0447 +/- 0.0146 | 1.110 +/- 0.039 |
| 200 | 300 | 3 | 1.0019 +/- 0.0057 | 0.0207 +/- 0.0090 | 0.0031 +/- 0.0001 | 0.0163 +/- 0.0066 | 0.977 +/- 0.005 |
| 200 | 1000 | 3 | 0.9664 +/- 0.0210 | 0.0613 +/- 0.0394 | 0.0179 +/- 0.0006 | 0.0473 +/- 0.0068 | 0.925 +/- 0.012 |
| 200 | 3000 | 3 | 0.5503 +/- 0.0243 | 0.1273 +/- 0.0564 | 0.1985 +/- 0.0088 | 0.0760 +/- 0.0148 | 0.904 +/- 0.024 |
| 1000 | 300 | 3 | 1.0061 +/- 0.0053 | 0.0233 +/- 0.0160 | 0.0005 +/- 0.0000 | 0.0153 +/- 0.0099 | 0.979 +/- 0.007 |
| 1000 | 1000 | 3 | 1.0197 +/- 0.0101 | 0.0120 +/- 0.0123 | 0.0023 +/- 0.0000 | 0.0253 +/- 0.0102 | 0.961 +/- 0.009 |
| 1000 | 3000 | 3 | 1.0096 +/- 0.0142 | 0.0500 +/- 0.0156 | 0.0079 +/- 0.0002 | 0.0543 +/- 0.0103 | 0.909 +/- 0.015 |

Descriptive associations only: repeated checkpoints share one pretrained base. Positive contraction means -log(FTSS / before). Outcome changes subtract the paired baseline. By-size and by-step associations are retained in summary.json; pooled correlations can reflect the shared effect of data size. No independent-observation p-values are reported.

- training_preference: Spearman 0.725690742693907
- training_anchor_advantage: Spearman 0.818070818070818
- loss_gap: Spearman 0.9261294261294261
- membership_auc: Spearman 0.9218559218559219
- coverage: Spearman 0.3655520310247332
- precision: Spearman -0.25668463878561704
- style_diversity_ratio: Spearman -0.3663003663003663
- pixel_training_preference: Spearman 0.884815525976238
