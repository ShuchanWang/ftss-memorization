# Matched FTSS MLP Reproduction and Fine-Tuning

State: complete. Phase: complete. Tables update as measurements finish.

Original training/model/evaluator function bodies are extracted without edits from the sibling repository. The selected from-scratch data sizes are endpoint checks, not the full published sweep. Subset RNG states preserve the original selected-size/run order on restart.

Training: four 2048-wide hidden layers, LayerNorm, dropout 0.1, Xavier initialization, AdamW, OneCycle peak LR 1e-4, batch 64, gradient clipping 5, EMA 0.999, CUDA AMP. Paper estimator: 15 trajectories and 200 Euler steps by default, 20 injection times through 0.95, fresh directions at each time. A validated batched backend preserves those probes; backend and execution-change provenance are recorded. The denominator below is the mean minimum gain of the largest from-scratch subset across runs, matching the original evaluation script.

## From Scratch

| Images | Run | Minimum FTSS | Paper ratio |
|---:|---:|---:|---:|
| 50 | 0 | 0.38677 | 0.7315 |
| 50 | 1 | 0.40950 | 0.7744 |
| 12000 | 0 | 0.51646 | 0.9767 |
| 12000 | 1 | 0.54106 | 1.0233 |

## Matched Fine-Tuning

Starts from the largest-subset EMA checkpoint. New class-balanced MNIST test examples are disjoint from original training; reference/calibration pools are disjoint from tuning. Fine-tuning uses the same architecture, dropout, optimizer family, batch, peak LR, clipping and EMA, with OneCycle compressed to the requested fine-tuning duration. Ordinary and EMA weights are evaluated separately with the same paper estimator protocol. This is an extension to fine-tuning, not a fine-tuning experiment reported in the original paper.

| Images | Run | Steps | Weights | FTSS / Before | Precision | Held-out Coverage | Style Diversity | Training Preference |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 50 | 0 | 0 | ema | 1.0000 | 0.175 | 0.145 | 0.743 | 0.590 |
| 50 | 0 | 1000 | raw | 0.9228 | 0.200 | 0.110 | 0.696 | 0.699 |
| 50 | 0 | 1000 | ema | 0.9346 | 0.170 | 0.105 | 0.695 | 0.633 |
| 50 | 0 | 3000 | raw | 0.8988 | 0.230 | 0.120 | 0.684 | 0.727 |
| 50 | 0 | 3000 | ema | 0.8892 | 0.215 | 0.120 | 0.679 | 0.715 |
| 1000 | 0 | 0 | ema | 1.0000 | 0.210 | 0.100 | 0.754 | 0.512 |
| 1000 | 0 | 1000 | raw | 1.0276 | 0.175 | 0.100 | 0.740 | 0.582 |
| 1000 | 0 | 1000 | ema | 0.9872 | 0.205 | 0.080 | 0.716 | 0.570 |
| 1000 | 0 | 3000 | raw | 0.9998 | 0.225 | 0.120 | 0.721 | 0.527 |
| 1000 | 0 | 3000 | ema | 0.9923 | 0.205 | 0.110 | 0.731 | 0.551 |
| 50 | 1 | 0 | ema | 1.0000 | 0.215 | 0.105 | 0.776 | 0.637 |
| 50 | 1 | 1000 | raw | 0.9433 | 0.335 | 0.110 | 0.759 | 0.746 |
| 50 | 1 | 1000 | ema | 0.9408 | 0.265 | 0.095 | 0.751 | 0.668 |
| 50 | 1 | 3000 | raw | 0.8933 | 0.405 | 0.125 | 0.747 | 0.742 |
| 50 | 1 | 3000 | ema | 0.8891 | 0.375 | 0.105 | 0.768 | 0.730 |
| 1000 | 1 | 0 | ema | 1.0000 | 0.185 | 0.095 | 0.774 | 0.559 |
| 1000 | 1 | 1000 | raw | 1.0384 | 0.145 | 0.075 | 0.821 | 0.562 |
| 1000 | 1 | 1000 | ema | 0.9850 | 0.160 | 0.085 | 0.802 | 0.559 |
| 1000 | 1 | 3000 | raw | 0.9894 | 0.230 | 0.120 | 0.799 | 0.566 |
| 1000 | 1 | 3000 | ema | 0.9855 | 0.190 | 0.105 | 0.788 | 0.578 |

External outcomes use the existing independent MNIST classifier and authors' PRDC implementation, with fixed spatial features and 200 real/200 generated samples. Scratch training-set preference is omitted because its evaluator pools are held out from scratch training, not members. All configurations, source hashes, and audits are saved.
