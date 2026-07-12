# ftss-memorization

Code and paper for **"The Geometry of Memorization: Finite-Time Spectral Sensitivity as a Diagnostic for Flow Matching Models"**.

This repository studies memorization in continuous-time generative models through **Finite-Time Spectral Sensitivity (FTSS)**, a gradient-free diagnostic for the geometry of learned flow matching trajectories. FTSS measures the root-mean-square directional sensitivity of the terminal generated state to small perturbations injected at intermediate times. In low-data regimes, overfitted models exhibit a sharp spectral collapse: the minimum FTSS along the trajectory drops relative to a full-data baseline.

## Contents

- `paper/`: LaTeX source, bibliography, and figures for the paper.
- `train_memorization_mlp.py`: Train MLP flow matching models across dataset sizes.
- `train_memorization_unet.py`: Train UNet flow matching models across dataset sizes.
- `train_memorization_dit.py`: Train DiT flow matching models across dataset sizes.
- `test_memorization_trend_mlp.py`: Estimate FTSS curves and Spectral Collapse Ratios for MLP checkpoints.
- `test_memorization_trend_unet.py`: Estimate FTSS curves and Spectral Collapse Ratios for UNet checkpoints.
- `test_memorization_trend_dit.py`: Estimate FTSS curves and Spectral Collapse Ratios for DiT checkpoints.

## Method

The paper introduces FTSS, denoted `g(t)`, as a forward-pass finite-difference estimator of the RMS singular value of the state-transition matrix from time `t` to the terminal time. The scalar diagnostic statistic is the **Spectral Collapse Ratio**

```text
M = g_min / g_min_full
```

where `g_min` is the bottleneck minimum of the FTSS curve and `g_min_full` is the corresponding full-data baseline. Lower values indicate stronger internal spectral collapse and therefore stronger evidence of memorization.

## Experiments

The experiments sweep training set sizes across MNIST, CIFAR-10, CIFAR-100, and imagenette/ImageNet-style data. The code supports three architectures:

- MLP for MNIST-scale flattened inputs.
- UNet for image flow matching.
- DiT for patch-based transformer flow matching.

Example training commands:

```bash
python train_memorization_mlp.py --dataset mnist
python train_memorization_unet.py --dataset cifar10
python train_memorization_dit.py --dataset imagenet --imagenet_path ./data/imagenette2
```

Example FTSS evaluation commands:

```bash
python test_memorization_trend_mlp.py --dataset mnist
python test_memorization_trend_unet.py --dataset cifar10
python test_memorization_trend_dit.py --dataset imagenet
```

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The scripts write checkpoints to `checkpoints/` and figures to `figures/`.
