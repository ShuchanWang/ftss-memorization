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

## Pretrained Fine-Tuning Studies

The FTSS follow-up code formerly in the watermark repository now lives in
`studies/flow_finetuning/`, with its tests beside the runners. The original
training and FTSS implementations above are unchanged.

```bash
python run_study.py --list
python run_study.py run_ftss_cifar_study.py --help
python run_study.py test
python export_study_summaries.py
```

Install `requirements-finetuning.txt` for these studies. They reuse model
loaders and existing public clean checkpoints from the sibling
`flow-matching-dynamics-watermarking` checkout. Set `WATERMARK_REPO` when that
checkout is elsewhere. No private watermark key is required by the clean studies.

New FTSS outputs default to this repository's `table_runs/`; datasets and
public base checkpoints are still resolved in the watermark checkout. The
launcher sets that working directory explicitly for legacy relative paths.
On the current machine, old FTSS artifact paths are compatibility symlinks
to the moved files. These local symlinks and full checkpoints are not synced.

`results/flow_finetuning/` contains compact, source-fingerprinted evidence,
including negative results. Full local logs, images and checkpoints remain in
`table_runs/`, which is ignored by Git. Saved manifests retain their historical
paths and hashes: source relocation does not authorize overwriting a frozen
experiment configuration. Use a new output directory when a runner rejects
an old source fingerprint.
