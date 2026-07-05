"""
Run contraction (transport gain) tests on all trained DiT models.
Computes memorization score M for each data size.
Supports MNIST, CIFAR-10, CIFAR-100, and ImageNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
# ============================================================================
# ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                    help='Dataset to test on')
args = parser.parse_args()

DATASET_NAME = args.dataset

# ============================================================================
# DATASET CONFIG
# ============================================================================
DATASET_CONFIGS = {
    'mnist': {
        'in_channels': 1,
        'img_size': 28,
        'D': 784,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000],
    },
    'cifar10': {
        'in_channels': 3,
        'img_size': 32,
        'D': 3072,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000],
    },
    'cifar100': {
        'in_channels': 3,
        'img_size': 32,
        'D': 3072,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000],
    },
    'imagenet': {
        'in_channels': 3,
        'img_size': 64,
        'D': 12288,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000],
    },
}

# Try to load config from any checkpoint to get actual image size
def get_config_from_checkpoint():
    """Try to load config from first available checkpoint."""
    cfg = DATASET_CONFIGS[DATASET_NAME]
    for n in cfg['data_sizes']:
        for run in [0, 1]:
            path = f"checkpoints/dit_{DATASET_NAME}_n{n}_run{run}.pt"
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu')
                if 'config' in ckpt:
                    return ckpt['config']
    return {
        'img_size': cfg['img_size'],
        'in_channels': cfg['in_channels'],
        'patch_size': 4,
        'hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 4,
        'dropout': 0.1,
    }

CONFIG = get_config_from_checkpoint()
IN_CHANNELS = CONFIG['in_channels']
IMG_SIZE = CONFIG['img_size']
D = IN_CHANNELS * IMG_SIZE * IMG_SIZE
DATA_SIZES = DATASET_CONFIGS[DATASET_NAME]['data_sizes']
N_RUNS = 2

print(f"Dataset: {DATASET_NAME}, Image size: {IMG_SIZE}, D: {D}")
print(f"Data sizes: {DATA_SIZES}")

# ============================================================================
# DiT MODEL (must match training script exactly)
# ============================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=1)

        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class SimpleDiT(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1,
                 hidden_dim=256, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.hidden_dim = hidden_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, hidden_dim) * 0.02)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_linear = nn.Linear(hidden_dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        B = x.shape[0]
        x_img = x.view(B, self.in_channels, self.img_size, self.img_size)

        x = self.patch_embed(x_img)
        x = x + self.pos_embed

        t_emb = self.time_embed(t)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_norm(x)
        x = self.final_linear(x)

        n_patches_per_side = self.img_size // self.patch_size
        x = x.reshape(B, self.n_patches, self.patch_size, self.patch_size, self.in_channels)
        x = x.reshape(B, n_patches_per_side, n_patches_per_side,
                      self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.in_channels, self.img_size, self.img_size)

        return x.reshape(B, -1)


# ============================================================================
# TRANSPORT GAIN MEASUREMENT
# ============================================================================
def measure_transport_gain(model, device, n_samples=15, n_steps=200):
    """Measure transport gain g(t) via isotropic finite differences."""
    model.eval()

    n_test_times = 20
    test_times = np.linspace(0, 0.95, n_test_times)
    delta_scale = 0.1

    all_ratios = []

    with torch.no_grad():
        for s in range(n_samples):
            x0 = torch.randn(1, D, device=device)

            # Generate clean trajectory
            x = x0
            trajectory = [x.clone()]
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t_val = torch.ones(1, 1, device=device) * (i * dt)
                v = model(x, t_val)
                v = torch.clamp(v, -10, 10)
                x = x + v * dt
                x = torch.clamp(x, -5, 5)
                trajectory.append(x.clone())
            x1_clean = x

            # Perturb at each test time
            ratios = []
            for t_perturb in test_times:
                step = int(t_perturb * n_steps)
                x_t = trajectory[step]

                # Isotropic perturbation
                delta = torch.randn(1, D, device=device)
                delta = delta / torch.norm(delta) * delta_scale * torch.norm(x_t)

                x_pert = x_t + delta
                for i in range(step, n_steps):
                    t_val = torch.ones(1, 1, device=device) * (i * dt)
                    v = model(x_pert, t_val)
                    v = torch.clamp(v, -10, 10)
                    x_pert = x_pert + v * dt
                    x_pert = torch.clamp(x_pert, -5, 5)

                dist = torch.norm(x_pert - x1_clean).item()
                delta_norm = torch.norm(delta).item()
                ratios.append(dist / (delta_norm + 1e-8))

            all_ratios.append(ratios)

    all_ratios = np.array(all_ratios)

    # RMS averaging (consistent with g(t) definition)
    g_rms = np.sqrt(np.mean(all_ratios**2, axis=0))

    early = np.mean(g_rms[:5])
    late = np.mean(g_rms[-5:])
    g_min = np.min(g_rms)
    overall = np.mean(g_rms)

    return {
        'mean_curve': g_rms,
        'test_times': test_times,
        'early': early,
        'late': late,
        'min': g_min,
        'overall': overall,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    all_metrics = {}

    for n in DATA_SIZES:
        print(f"\n{'='*60}")
        print(f"Testing DiT on {DATASET_NAME} N={n}...", flush=True)
        all_metrics[n] = []

        for run in range(N_RUNS):
            path = f"checkpoints/dit_{DATASET_NAME}_n{n}_run{run}.pt"
            print(f"  Looking for: {path}", flush=True)
            
            if not os.path.exists(path):
                print(f"  MISSING: {path}", flush=True)
                continue

            ckpt = torch.load(path, map_location=device)

            if 'config' in ckpt:
                cfg = ckpt['config']
            else:
                cfg = CONFIG

            model = SimpleDiT(
                img_size=cfg['img_size'],
                patch_size=cfg['patch_size'],
                in_channels=cfg['in_channels'],
                hidden_dim=cfg['hidden_dim'],
                num_heads=cfg['num_heads'],
                num_layers=cfg['num_layers'],
                dropout=cfg.get('dropout', 0.1),
            ).to(device)

            model.load_state_dict(ckpt['model'])
            model.eval()

            print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
            t0 = time.time()
            metrics = measure_transport_gain(model, device, n_samples=15)
            elapsed = time.time() - t0
            all_metrics[n].append(metrics)

            print(f"done in {elapsed:.0f}s, early={metrics['early']:.3f}, "
                  f"late={metrics['late']:.3f}, g_min={metrics['min']:.3f}, "
                  f"overall={metrics['overall']:.3f}", flush=True)

     # ================================================================
    # PLOTS
    # ================================================================
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    largest_n = max(DATA_SIZES)
    baseline_min = np.mean([m['min'] for m in all_metrics[largest_n]]) if all_metrics.get(largest_n) else None

    # ================================================================
    # PLOT 1: Transport gain curves g(t) for each data size
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(DATA_SIZES)))
    for n, color in zip(DATA_SIZES, colors):
        if all_metrics[n]:
            curves = [m['mean_curve'] for m in all_metrics[n]]
            mean_curve = np.mean(curves, axis=0)
            times = all_metrics[n][0]['test_times']
            ax.plot(times, mean_curve, color=color, lw=2, label=f'$N={n}$')
    ax.axhline(1.0, color='gray', ls='--', lw=1.5, label='$g=1$ (identity)')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Transport Gain $g(t)$')
    ax.set_title(f'Transport Gain Profiles (DiT, {DATASET_NAME})')
    ax.legend(ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'figures/dit_{DATASET_NAME}_gain_curves.png', dpi=200, bbox_inches='tight')
    print(f"Saved figures/dit_{DATASET_NAME}_gain_curves.png")
    plt.close()

    # ================================================================
    # PLOT 2: Memorization score M vs data size
    # ================================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    if baseline_min:
        sizes = []
        M_scores = []
        M_std = []
        for n in DATA_SIZES:
            if all_metrics[n]:
                vals = [m['min'] / baseline_min for m in all_metrics[n]]
                sizes.append(n)
                M_scores.append(np.mean(vals))
                M_std.append(np.std(vals))
        ax.errorbar(sizes, M_scores, yerr=M_std, marker='D', capsize=5, lw=2,
                    markersize=9, color='#2196F3')
        ax.axhline(1.0, color='gray', ls='--', lw=1.5, 
                   label=f'Baseline ($N={largest_n}$)')

    ax.set_xlabel('Number of Training Samples $N$')
    ax.set_ylabel('Memorization Score $M$')
    ax.set_title(f'Memorization Score vs Data Size (DiT, {DATASET_NAME})')
    ax.set_xscale('log')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'figures/dit_{DATASET_NAME}_memorization_score.png', dpi=200, bbox_inches='tight')
    print(f"Saved figures/dit_{DATASET_NAME}_memorization_score.png")
    plt.close()

    # ================================================================
    # PRINT SUMMARY TABLE
    # ================================================================
    print(f"\n{'='*80}")
    print(f"DiT MEMORIZATION ANALYSIS SUMMARY - {DATASET_NAME}")
    print(f"{'='*80}")
    if baseline_min:
        print(f"  Baseline g_min (N={largest_n}): {baseline_min:.4f}")
    print(f"  {'N':>8s}  {'Early':>8s}  {'Late':>8s}  {'g_min':>10s}  {'Overall':>8s}  {'M':>8s}")
    print(f"  {'-'*60}")

    for n in DATA_SIZES:
        if all_metrics[n]:
            e = np.mean([m['early'] for m in all_metrics[n]])
            l = np.mean([m['late'] for m in all_metrics[n]])
            m_val = np.mean([m['min'] for m in all_metrics[n]])
            o = np.mean([m['overall'] for m in all_metrics[n]])
            if baseline_min:
                M = m_val / baseline_min
                flag = "  <-- MEM" if M < 0.7 else ""
                print(f"  {n:>8d}  {e:>8.3f}  {l:>8.3f}  {m_val:>10.4f}  {o:>8.3f}  {M:>8.3f}{flag}")
            else:
                print(f"  {n:>8d}  {e:>8.3f}  {l:>8.3f}  {m_val:>10.4f}  {o:>8.3f}  {'N/A':>8s}")

if __name__ == "__main__":
    main()