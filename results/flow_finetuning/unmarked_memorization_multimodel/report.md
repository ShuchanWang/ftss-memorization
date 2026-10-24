# Unwatermarked Fine-Tuning: Contraction and Memorization

Ordinary full-parameter fine-tuning of existing clean MNIST MLP and UNet checkpoints. The tuned, untouched reference, and calibration pools are disjoint and class-balanced, drawn from the MNIST test split excluded from original model training. There is no watermark loss or watermark evaluation.

3 seeds; subsets of [10, 50, 1000, 4000] images; checkpoints at [0, 300, 1000, 3000] steps; learning rate 0.0003. Each checkpoint uses 256 generated samples and 15 FTSS probes.

## Operational Definitions

- FTSS ratio: minimum terminal finite-difference RMS gain over probe injection times, divided by its pre-fine-tuning value. A ratio below one indicates increased contraction. Association predictor: minus log of this ratio.
- Calibrated replication: generated sample is closer to the tuned than the equal-size reference pool and its tuned distance is below the 1% quantile of calibration-to-tuned distances. The upstream FTSS rule uses raw endpoints; visible-image replication clips endpoints to the display range.
- Fixed-error replication: visible-image nearest-tuned pixel RMSE in [0,1] is at most 0.05 (or 0.10), and the sample is closer to tuned than reference images. These are supplementary pixel-tolerance definitions; inspect the neighbor figure to judge visual similarity.
- Training attraction: fraction of samples nearer to tuned than reference images, plus normalized nearest-neighbor distance advantage. These measure preference, not exact replication.
- Replication coverage: fraction of distinct tuned examples matched by at least one generated sample under the calibrated replication rule. This distinguishes copying several training examples from repeated output of one example.
- Loss-based membership AUROC: negative flow-matching loss distinguishes tuned from reference images; 0.5 is chance. Loss gap: reference mean loss minus tuned mean loss. Both measure exposure or overfitting rather than image duplication.

## Final Checkpoint

Mean +/- seed standard deviation. Replication and preference are percentages.

| Model | Tuned images | FTSS ratio | Raw calibrated replicas | Visible calibrated replicas | Visible RMSE <= 0.05 | Raw copy coverage | Train preference | Membership AUROC | Loss gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp | 10 | 1.346 +/- 0.387 | 15.625 +/- 21.821 | 36.328 +/- 18.628 | 0.000 +/- 0.000 | 36.667 +/- 44.969 | 97.135 +/- 0.664 | 0.977 +/- 0.021 | 0.415 +/- 0.059 |
| mlp | 50 | 1.267 +/- 0.132 | 0.130 +/- 0.184 | 2.865 +/- 0.737 | 0.000 +/- 0.000 | 0.667 +/- 0.943 | 91.667 +/- 1.208 | 0.913 +/- 0.014 | 0.303 +/- 0.017 |
| mlp | 1000 | 1.114 +/- 0.023 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 57.161 +/- 3.350 | 0.558 +/- 0.017 | 0.026 +/- 0.007 |
| mlp | 4000 | 1.133 +/- 0.041 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 51.302 +/- 5.263 | 0.518 +/- 0.007 | 0.006 +/- 0.004 |
| unet | 10 | 0.159 +/- 0.015 | 97.135 +/- 1.289 | 97.135 +/- 1.289 | 95.052 +/- 0.664 | 100.000 +/- 0.000 | 99.349 +/- 0.664 | 0.997 +/- 0.005 | 0.633 +/- 0.117 |
| unet | 50 | 0.460 +/- 0.076 | 69.922 +/- 1.390 | 69.922 +/- 1.390 | 45.703 +/- 2.923 | 97.333 +/- 2.494 | 96.875 +/- 1.688 | 0.993 +/- 0.002 | 0.391 +/- 0.043 |
| unet | 1000 | 0.971 +/- 0.025 | 0.911 +/- 0.184 | 0.911 +/- 0.184 | 0.000 +/- 0.000 | 0.233 +/- 0.047 | 56.901 +/- 1.757 | 0.569 +/- 0.028 | 0.014 +/- 0.006 |
| unet | 4000 | 1.039 +/- 0.023 | 0.391 +/- 0.319 | 0.521 +/- 0.368 | 0.130 +/- 0.184 | 0.025 +/- 0.020 | 54.297 +/- 1.595 | 0.515 +/- 0.015 | 0.002 +/- 0.002 |

## Association With Increased Contraction

Outcomes are changes from the same trajectory's initial measurement. Positive correlations mean greater contraction accompanies greater replication or exposure. Adjusted correlations remove trajectory intercepts and shared checkpoint-step effects, separately within architecture. Repeated checkpoints stay together in seed-block bootstrap resampling. There are 3 seed blocks; intervals are exploratory, and the checkpoints are not independent experimental repeats.

| Model | Outcome change | Spearman | Adjusted Pearson | Seed-block 95% interval |
|---|---|---:|---:|---|
| mlp | raw_copy_rate_1pct | 0.407 | 0.331 | [-0.547, 0.450] |
| mlp | visible_copy_rate_1pct | 0.164 | -0.118 | [-0.813, 0.472] |
| mlp | visible_copy_rate_5pct | 0.248 | -0.311 | [-0.656, 0.162] |
| mlp | raw_copy_coverage_1pct | 0.339 | 0.187 | [-0.692, 0.539] |
| mlp | visible_copy_coverage_1pct | 0.170 | -0.355 | [-0.470, 0.423] |
| mlp | visible_replica_rmse_005 | undefined (constant or insufficient data) | undefined (constant or insufficient data) | not estimated |
| mlp | visible_replica_rmse_010 | 0.224 | 0.184 | [-0.922, 0.333] |
| mlp | visible_train_preference | 0.296 | -0.074 | [-0.487, -0.022] |
| mlp | visible_neighbor_advantage | 0.231 | -0.307 | [-0.685, 0.103] |
| mlp | membership_auc | 0.260 | -0.271 | [-0.620, -0.223] |
| mlp | loss_gap | 0.191 | -0.461 | [-0.733, -0.204] |
| unet | raw_copy_rate_1pct | 0.857 | 0.711 | [0.661, 0.806] |
| unet | visible_copy_rate_1pct | 0.875 | 0.710 | [0.660, 0.807] |
| unet | visible_copy_rate_5pct | 0.891 | 0.655 | [0.597, 0.759] |
| unet | raw_copy_coverage_1pct | 0.821 | 0.099 | [-0.002, 0.323] |
| unet | visible_copy_coverage_1pct | 0.838 | 0.099 | [-0.002, 0.326] |
| unet | visible_replica_rmse_005 | 0.852 | 0.802 | [0.778, 0.838] |
| unet | visible_replica_rmse_010 | 0.854 | 0.752 | [0.717, 0.818] |
| unet | visible_train_preference | 0.851 | 0.428 | [0.359, 0.494] |
| unet | visible_neighbor_advantage | 0.871 | 0.819 | [0.786, 0.887] |
| unet | membership_auc | 0.868 | 0.229 | [0.072, 0.429] |
| unet | loss_gap | 0.903 | 0.892 | [0.859, 0.964] |

All checkpoint means are in `conditions.csv`; individual measurements and saved audits are in `results.json` and `*_audit.npz`. `analysis.json` contains per-seed and per-step associations. `trajectories.png` shows the full trajectories; `associations.png` plots baseline-relative outcomes against contraction. `neighbors.png` shows selected generated samples and their nearest tuned/reference images at the final checkpoint: selection is by minimum, median, and maximum tuned-image distance in the first seed's 10-image condition. `analysis.json` also reports a secondary sensitivity analysis using the full FTSS curve instead of its minimum; the primary definition remains the upstream minimum-gain statistic.
