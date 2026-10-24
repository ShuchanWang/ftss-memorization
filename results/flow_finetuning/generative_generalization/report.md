# Generative Generalization in Independent Features

An independent MNIST CNN was fitted on 50,000 original training images and selected on 10,000 other original training images. Fine-tuning/evaluation images are in MNIST test; none were used to train or select this evaluator. Its test accuracy is 99.00%.

Primary features are normalized spatial convolutional features; penultimate features provide a secondary sensitivity check. PRDC uses the authors' package, k=3, and 200 real/200 generated examples in the default protocol. Held-out examples are disjoint from fine-tuning. Within-digit style diversity is feature variance relative to real digits; digit counts and missing-class information are saved. Precision measures feature-manifold fidelity, not human quality judgments. Classifier confidence alone is not treated as quality.

## Main Measurements

Mean over three seeds. Each cell shows initial -> final. Coverage, recall, precision, and training preference are percentages; style diversity is a ratio to real-data diversity.

| Model | Tuned images | Precision | Held-out coverage | Held-out recall | Within-digit diversity | Effective digits | Training preference |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp | 10 | 68.333 -> 58.667 | 66.333 -> 20.000 | 71.833 -> 4.333 | 0.997 -> 0.510 | 9.302 -> 8.295 | 41.536 -> 89.844 |
| mlp | 50 | 71.000 -> 69.333 | 65.667 -> 51.833 | 72.500 -> 24.333 | 0.996 -> 0.886 | 9.302 -> 8.694 | 54.297 -> 80.729 |
| mlp | 1000 | 72.333 -> 60.167 | 65.500 -> 50.500 | 70.333 -> 62.833 | 1.002 -> 0.950 | 9.302 -> 8.969 | 46.875 -> 52.865 |
| mlp | 4000 | 74.500 -> 60.000 | 65.833 -> 51.167 | 72.500 -> 62.500 | 0.986 -> 0.936 | 9.302 -> 8.543 | 54.557 -> 53.255 |
| unet | 10 | 86.667 -> 98.667 | 85.167 -> 20.000 | 92.500 -> 1.667 | 1.067 -> 0.105 | 9.758 -> 9.114 | 46.745 -> 99.219 |
| unet | 50 | 86.833 -> 96.500 | 85.000 -> 68.000 | 91.167 -> 40.500 | 1.067 -> 0.771 | 9.758 -> 9.738 | 50.000 -> 95.703 |
| unet | 1000 | 84.333 -> 86.000 | 84.833 -> 86.000 | 91.333 -> 91.000 | 1.072 -> 1.004 | 9.758 -> 9.759 | 51.302 -> 57.943 |
| unet | 4000 | 89.667 -> 87.833 | 84.000 -> 82.667 | 91.000 -> 90.500 | 1.056 -> 1.021 | 9.758 -> 9.696 | 48.828 -> 52.474 |

## Association With FTSS Contraction

Baseline-relative changes are analyzed separately by architecture. Coverage, recall, style, and category outcomes are oriented as losses; precision and training attraction as gains. Adjusted correlations remove trajectory intercepts and common step effects. Three seed blocks support exploratory associations, not independent repetitions of every checkpoint.

| Model | External outcome | Spearman | Adjusted Pearson |
|---|---|---:|---:|
| mlp | spatial_coverage_oriented_change | 0.076 | -0.217 |
| mlp | spatial_recall_oriented_change | 0.177 | -0.298 |
| mlp | spatial_precision_oriented_change | -0.158 | -0.249 |
| mlp | spatial_style_diversity_ratio_oriented_change | 0.106 | -0.267 |
| mlp | spatial_effective_digit_modes_oriented_change | -0.105 | -0.121 |
| mlp | spatial_training_preference_oriented_change | 0.294 | -0.304 |
| mlp | spatial_training_anchor_advantage_oriented_change | 0.248 | -0.294 |
| mlp | embedding_coverage_oriented_change | 0.317 | -0.338 |
| mlp | embedding_recall_oriented_change | 0.203 | -0.259 |
| mlp | embedding_precision_oriented_change | -0.244 | -0.198 |
| mlp | embedding_style_diversity_ratio_oriented_change | 0.162 | -0.437 |
| mlp | embedding_effective_digit_modes_oriented_change | -0.105 | -0.121 |
| mlp | embedding_training_preference_oriented_change | 0.144 | -0.242 |
| mlp | embedding_training_anchor_advantage_oriented_change | 0.219 | -0.345 |
| unet | spatial_coverage_oriented_change | 0.809 | 0.826 |
| unet | spatial_recall_oriented_change | 0.837 | 0.938 |
| unet | spatial_precision_oriented_change | 0.789 | 0.663 |
| unet | spatial_style_diversity_ratio_oriented_change | 0.872 | 0.824 |
| unet | spatial_effective_digit_modes_oriented_change | 0.295 | 0.159 |
| unet | spatial_training_preference_oriented_change | 0.891 | 0.502 |
| unet | spatial_training_anchor_advantage_oriented_change | 0.893 | 0.939 |
| unet | embedding_coverage_oriented_change | 0.791 | 0.825 |
| unet | embedding_recall_oriented_change | 0.859 | 0.898 |
| unet | embedding_precision_oriented_change | 0.338 | -0.298 |
| unet | embedding_style_diversity_ratio_oriented_change | 0.873 | 0.816 |
| unet | embedding_effective_digit_modes_oriented_change | 0.295 | 0.159 |
| unet | embedding_training_preference_oriented_change | 0.878 | 0.590 |
| unet | embedding_training_anchor_advantage_oriented_change | 0.879 | 0.928 |

## Real-Versus-Real Controls

Two disjoint balanced held-out pools with the same sample count establish finite-sample metric levels. These controls are not used to train features or tune metric parameters.

- spatial: precision=0.909, recall=0.904, density=0.999, coverage=0.898
- embedding: precision=0.870, recall=0.895, density=0.980, coverage=0.901

[Metric definitions and authors' implementation](https://github.com/clovaai/generative-evaluation-prdc). All measurements, feature tensors, evaluator weights, provenance, and per-seed associations are saved alongside this report.
