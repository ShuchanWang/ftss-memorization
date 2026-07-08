"""
Test FTSS for UNet-based flow matching model.
Measures g(t) and memorization score M across data sizes.
Supports MNIST, CIFAR-10, CIFAR-100, and ImageNet.
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
                    choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                    help='Dataset to test on')
args = parser.parse_args()

DATASET_NAME = args.dataset

# ============================================================================
# DATASET CONFIG
# ============================================================================
DATASET_CONFIGS = {
    'mnist': {'img_size': 28, 'in_channels': 1, 'D': 784,
              'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000]},
    'cifar10': {'img_size': 32, 'in_channels': 3, 'D': 3072,
                'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000]},
    'cifar100': {'img_size': 32, 'in_channels': 3, 'D': 3072,
                 'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000]},
    'imagenet': {'img_size': 64, 'in_channels': 3, 'D': 12288,
                 'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000]},
}

# Try to load actual config from checkpoint if available
def get_config_from_checkpoint():
    cfg = DATASET_CONFIGS[DATASET_NAME]
    for n in cfg['data_sizes']:
        for run in [0, 1]:
            path = f"checkpoints/unet_{DATASET_NAME}_n{n}_run{run}.pt"
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu')
                if 'config' in ckpt:
                    return {
                        'img_size': ckpt['config'].get('img_size', cfg['img_size']),
                        'in_channels': ckpt['config'].get('in_channels', cfg['in_channels']),
                        'D': ckpt['config'].get('img_size', cfg['img_size']) ** 2 * 
                             ckpt['config'].get('in_channels', cfg['in_channels']),
                        'data_sizes': cfg['data_sizes'],
                    }
    return cfg

cfg = get_config_from_checkpoint()
IMG_SIZE = cfg['img_size']
IN_CHANNELS = cfg['in_channels']
D = cfg['D']
DATA_SIZES = cfg['data_sizes']
N_RUNS = 2

print(f"Dataset: {DATASET_NAME}, Image size: {IMG_SIZE}, D: {D}")
print(f"Data sizes: {DATA_SIZES}")

# ============================================================================
# UNET MODEL (must match training script)
# ============================================================================
class SimpleUNet(nn.Module):
    def __init__(self, img_size, in_channels, base_ch=64):
        super().__init__()
        self.img_size, self.in_channels = img_size, in_channels
        ch1, ch2, ch3 = base_ch, base_ch * 2, base_ch * 4
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1, 3, padding=1),
            nn.InstanceNorm2d(ch1, affine=True), nn.SiLU()
        )
        self.down1 = nn.Conv2d(ch1, ch1, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch1, ch2, 3, padding=1),
            nn.InstanceNorm2d(ch2, affine=True), nn.SiLU()
        )
        self.down2 = nn.Conv2d(ch2, ch2, 4, stride=2, padding=1)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch2, ch3, 3, padding=1),
            nn.InstanceNorm2d(ch3, affine=True), nn.SiLU(),
            nn.Conv2d(ch3, ch3, 3, padding=1),
            nn.InstanceNorm2d(ch3, affine=True), nn.SiLU(),
        )
        
        self.up1 = nn.ConvTranspose2d(ch3, ch2, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch2 + ch2, ch2, 3, padding=1),
            nn.InstanceNorm2d(ch2, affine=True), nn.SiLU()
        )
        self.up2 = nn.ConvTranspose2d(ch2, ch1, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(ch1 + ch1, ch1, 3, padding=1),
            nn.InstanceNorm2d(ch1, affine=True), nn.SiLU()
        )
        self.final = nn.Conv2d(ch1, in_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, ch3), nn.SiLU(), nn.Linear(ch3, ch3)
        )
    
    def forward(self, x, t):
        B = x.shape[0]
        x_img = x.view(B, self.in_channels, self.img_size, self.img_size)
        t_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        
        h1 = self.enc1(x_img)
        h1_d = self.down1(h1)
        h2 = self.enc2(h1_d)
        h2_d = self.down2(h2)
        
        h = self.bottleneck[0](h2_d)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h)
        h = h + t_emb
        h = self.bottleneck[3](h)
        h = self.bottleneck[4](h)
        
        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec1[0](h); h = self.dec1[1](h); h = self.dec1[2](h)
        
        h = self.up2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec2[0](h); h = self.dec2[1](h); h = self.dec2[2](h)
        
        out = self.final(h)
        return out.reshape(B, -1)


# ============================================================================
# TRANSPORT GAIN MEASUREMENT
# ============================================================================
def measure_transport_gain(model, device, n_samples=15, n_steps=200):
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
        print(f"Testing UNet on {DATASET_NAME} N={n}...", flush=True)
        all_metrics[n] = []
        
        for run in range(N_RUNS):
            path = f"checkpoints/unet_{DATASET_NAME}_n{n}_run{run}.pt"
            print(f"  Looking for: {path}", flush=True)
            
            if not os.path.exists(path):
                print(f"  MISSING: {path}", flush=True)
                continue
            
            ckpt = torch.load(path, map_location=device)
            
            # Read config from checkpoint if available
            if 'config' in ckpt:
                img_sz = ckpt['config'].get('img_size', IMG_SIZE)
                in_ch = ckpt['config'].get('in_channels', IN_CHANNELS)
            else:
                img_sz, in_ch = IMG_SIZE, IN_CHANNELS
            
            model = SimpleUNet(img_sz, in_ch, base_ch=64).to(device)
            
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            
            print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
            metrics = measure_transport_gain(model, device, n_samples=15)
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
    
    # PLOT 1: Transport gain curves
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
    ax.set_title(f'FTSS Profiles (UNet, {DATASET_NAME})')
    ax.legend(ncol=2, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'figures/unet_{DATASET_NAME}_gain_curves.png', dpi=200, bbox_inches='tight')
    print(f"Saved figures/unet_{DATASET_NAME}_gain_curves.png")
    plt.close()
    
    # PLOT 2: Memorization score
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
    ax.set_title(f'Memorization Score vs Data Size (UNet, {DATASET_NAME})')
    ax.set_xscale('log')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f'figures/unet_{DATASET_NAME}_memorization_score.png', dpi=200, bbox_inches='tight')
    print(f"Saved figures/unet_{DATASET_NAME}_memorization_score.png")
    plt.close()


if __name__ == "__main__":
    main()