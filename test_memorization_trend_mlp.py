"""
Test FTSS for MLP-based flow matching model.
Measures g(t) and Spectral Collapse Ratio M across data sizes.
Supports MNIST, CIFAR-10, CIFAR-100.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
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
                    choices=['mnist', 'cifar10', 'cifar100'],
                    help='Dataset to test on')
args = parser.parse_args()

DATASET_NAME = args.dataset

# ============================================================================
# DATASET CONFIG
# ============================================================================
DATASET_CONFIGS = {
    'mnist': {'img_size': 28, 'in_channels': 1, 'D': 784, 'hidden_dim': 2048,
              'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000]},
    'cifar10': {'img_size': 32, 'in_channels': 3, 'D': 3072, 'hidden_dim': 2048,
                'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000]},
    'cifar100': {'img_size': 32, 'in_channels': 3, 'D': 3072, 'hidden_dim': 2048,
                 'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000]},
}

# Try to load actual config from checkpoint
def get_config_from_checkpoint():
    cfg = DATASET_CONFIGS[DATASET_NAME]
    for n in cfg['data_sizes']:
        for run in [0, 1]:
            path = f"checkpoints/mlp_{DATASET_NAME}_n{n}_run{run}.pt"
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu')
                if 'config' in ckpt:
                    saved = ckpt['config']
                    return {
                        'D': saved.get('D', cfg['D']),
                        'hidden_dim': saved.get('hidden_dim', cfg['hidden_dim']),
                        'img_size': saved.get('img_size', cfg['img_size']),
                        'in_channels': saved.get('in_channels', cfg['in_channels']),
                        'data_sizes': cfg['data_sizes'],
                    }
    return cfg

cfg = get_config_from_checkpoint()
D = cfg['D']
HIDDEN_DIM = cfg['hidden_dim']
IMG_SIZE = cfg.get('img_size', 28)
IN_CHANNELS = cfg.get('in_channels', 1)
DATA_SIZES = cfg['data_sizes']
N_RUNS = 2

print(f"Dataset: {DATASET_NAME}, D: {D}, Hidden dim: {HIDDEN_DIM}")
print(f"Data sizes: {DATA_SIZES}")

# ============================================================================
# MLP MODEL (must match training script exactly)
# ============================================================================
class MLPFlow(nn.Module):
    def __init__(self, D=784, hidden_dim=2048):
        super().__init__()
        self.D = D
        self.net = nn.Sequential(
            nn.Linear(D + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, D)
        )
    
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ============================================================================
# FTSS ESTIMATION
# ============================================================================
def estimate_ftss(model, device, n_samples=15, n_steps=200):
    """Measure FTSS g(t) via isotropic finite differences."""
    model.eval()
    
    n_test_times = 20
    test_times = np.linspace(0, 0.95, n_test_times)
    delta_scale = 0.1
    
    all_ratios = []
    
    with torch.no_grad():
        for s in range(n_samples):
            x0 = torch.randn(1, D, device=device)
            
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
            
            ratios = []
            for t_perturb in test_times:
                step = int(t_perturb * n_steps)
                x_t = trajectory[step]
                
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
        print(f"Testing MLP on {DATASET_NAME} N={n}...", flush=True)
        all_metrics[n] = []
        
        for run in range(N_RUNS):
            path = f"checkpoints/mlp_{DATASET_NAME}_n{n}_run{run}.pt"
            print(f"  Looking for: {path}", flush=True)
            
            if not os.path.exists(path):
                print(f"  MISSING: {path}", flush=True)
                continue
            
            ckpt = torch.load(path, map_location=device)
            
            # Read config from checkpoint
            if 'config' in ckpt:
                d_ckpt = ckpt['config'].get('D', D)
                hd_ckpt = ckpt['config'].get('hidden_dim', HIDDEN_DIM)
            else:
                d_ckpt, hd_ckpt = D, HIDDEN_DIM
            
            model = MLPFlow(D=d_ckpt, hidden_dim=hd_ckpt).to(device)
            
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            
            print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
            metrics = estimate_ftss(model, device, n_samples=15)
            all_metrics[n].append(metrics)
            
            print(f"early={metrics['early']:.3f}, late={metrics['late']:.3f}, "
                  f"g_min={metrics['min']:.3f}, overall={metrics['overall']:.3f}", flush=True)
    
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
    
    # PLOT 1: FTSS curves
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
    ax.set_ylabel('FTSS $g(t)$')
    ax.set_title(f'FTSS Profiles (MLP, {DATASET_NAME})')
    ax.legend(ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'figures/mlp_{DATASET_NAME}_gain_curves.png', dpi=200, bbox_inches='tight')
    print(f"Saved figures/mlp_{DATASET_NAME}_gain_curves.png")
    plt.close()
    
    # PLOT 2: Spectral Collapse Ratio
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
    ax.set_ylabel('Spectral Collapse Ratio $M$')
    ax.set_title(f'Spectral Collapse Ratio vs Data Size (MLP, {DATASET_NAME})')
    ax.set_xscale('log')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'figures/mlp_{DATASET_NAME}_memorization_score.png', dpi=200, bbox_inches='tight')
    print(f"Saved figures/mlp_{DATASET_NAME}_memorization_score.png")
    plt.close()


if __name__ == "__main__":
    main()