"""
Train MLP models at multiple data sizes to measure memorization vs FTSS.
Supports MNIST, CIFAR-10, CIFAR-100.
Automatically skips training if checkpoint exists.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import time
import os
import argparse
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'cifar10', 'cifar100'],
                    help='Dataset to train on')
parser.add_argument('--sizes', type=int, nargs='+',
                    help='Train only these subset sizes (default: full sweep)')
parser.add_argument('--runs', type=int, default=2,
                    help='Number of independent runs (default: 2)')
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
        'mean': (0.5,),
        'std': (0.5,),
        'steps': 20000,
        'lr': 1e-4,
    },
    'cifar10': {
        'in_channels': 3,
        'img_size': 32,
        'D': 3072,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000],
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'steps': 30000,
        'lr': 5e-5,
    },
    'cifar100': {
        'in_channels': 3,
        'img_size': 32,
        'D': 3072,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000],
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'steps': 30000,
        'lr': 5e-5,
    },
}

cfg = DATASET_CONFIGS[DATASET_NAME]
IN_CHANNELS = cfg['in_channels']
IMG_SIZE = cfg['img_size']
D = cfg['D']
DATA_SIZES = cfg['data_sizes']
if args.sizes is not None:
    if any(n not in DATA_SIZES for n in args.sizes):
        parser.error(f'--sizes must be chosen from {DATA_SIZES}')
    DATA_SIZES = sorted(set(args.sizes))
STEPS = cfg['steps']
LR = cfg['lr']

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 64
HIDDEN_DIM = 2048  # Larger hidden dim for image data
N_RUNS = args.runs
if N_RUNS < 1:
    parser.error('--runs must be positive')
EMA_DECAY = 0.999
USE_AMP = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Dataset: {DATASET_NAME}, Image size: {IMG_SIZE}, D: {D}")
print(f"Data sizes: {DATA_SIZES}")
print(f"Steps: {STEPS}, Batch size: {BATCH_SIZE}, LR: {LR}")
print(f"Hidden dim: {HIDDEN_DIM}")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
        return datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
        return datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
        return datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)

print(f"\nLoading {DATASET_NAME} dataset...")
train_ds = get_dataset(DATASET_NAME)
print(f"Training samples available: {len(train_ds)}")

DATA_SIZES = [n for n in DATA_SIZES if n <= len(train_ds)]
print(f"Effective data sizes: {DATA_SIZES}")

# ============================================================================
# MODEL
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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ============================================================================
# EMA
# ============================================================================
@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_samples(model, save_path, n_samples=4):
    model.eval()
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
    if n_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i in range(n_samples):
            x = torch.randn(1, D, device=device)
            dt = 1.0 / 200
            for step in range(200):
                t_val = torch.ones(1, 1, device=device) * (step * dt)
                v = model(x, t_val)
                v = torch.clamp(v, -10, 10)
                x = x + v * dt
                x = torch.clamp(x, -5, 5)
            
            sample = x.view(IN_CHANNELS, IMG_SIZE, IMG_SIZE).cpu().numpy()
            if IN_CHANNELS == 1:
                axes[i].imshow(sample[0], cmap='gray', vmin=-1, vmax=1)
            else:
                sample_rgb = np.transpose(sample, (1, 2, 0))
                sample_rgb = np.clip((sample_rgb + 1) / 2, 0, 1)
                axes[i].imshow(sample_rgb)
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
    
    plt.suptitle(f'Generated Samples - {os.path.basename(save_path)}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved sample images to {save_path}")


# ============================================================================
# TRAINING
# ============================================================================
def train_with_tracking(n_samples, run_id):
    ckpt_path = f"checkpoints/mlp_{DATASET_NAME}_n{n_samples}_run{run_id}.pt"
    
    if os.path.exists(ckpt_path):
        print(f"  Checkpoint exists, loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg_saved = ckpt.get('config', {})
        model = MLPFlow(D=cfg_saved.get('D', D), hidden_dim=cfg_saved.get('hidden_dim', HIDDEN_DIM)).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        vis_path = f"figures/mlp_{DATASET_NAME}_n{n_samples}_run{run_id}_samples.png"
        if not os.path.exists(vis_path):
            visualize_samples(model, vis_path)
        
        loss_history = ckpt.get('loss_history', [])
        final_loss = ckpt['final_loss']
        return model, loss_history, final_loss
    
    torch.manual_seed(SEED + run_id * 1000)
    model = MLPFlow(D=D, hidden_dim=HIDDEN_DIM).to(device)
    
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=STEPS, pct_start=0.1
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    indices = random.sample(range(len(train_ds)), n_samples)
    subset = Subset(train_ds, indices)
    loader = DataLoader(subset, batch_size=min(BATCH_SIZE, n_samples), shuffle=True)
    
    all_x1 = []
    for x, _ in loader:
        all_x1.append(x.view(x.size(0), -1))
    x1_pool = torch.cat(all_x1, dim=0).to(device)
    
    loss_history = []
    log_every = max(1, STEPS // 50)
    best_loss = float('inf')
    
    for step in range(STEPS):
        model.train()
        idx = torch.randperm(len(x1_pool))[:min(BATCH_SIZE, len(x1_pool))]
        x1 = x1_pool[idx]
        bs = x1.shape[0]
        
        x0 = torch.randn(bs, D, device=device)
        t_val = torch.rand(bs, 1, device=device)
        x_t = (1 - t_val) * x0 + t_val * x1
        u_true = x1 - x0
        
        opt.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            pred = model(x_t, t_val)
            loss = F.mse_loss(pred, u_true)
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        
        update_ema(ema_model, model, EMA_DECAY)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if step % log_every == 0:
            loss_history.append((step, loss.item()))
    
    final_loss = loss.item()
    
    vis_path = f"figures/mlp_{DATASET_NAME}_n{n_samples}_run{run_id}_samples.png"
    visualize_samples(ema_model, vis_path)
    
    torch.save({
        'model': ema_model.state_dict(),
        'n_samples': n_samples,
        'dataset': DATASET_NAME,
        'final_loss': final_loss,
        'best_loss': best_loss,
        'loss_history': loss_history,
        'config': {
            'D': D,
            'hidden_dim': HIDDEN_DIM,
            'img_size': IMG_SIZE,
            'in_channels': IN_CHANNELS,
        }
    }, ckpt_path)
    
    return ema_model, loss_history, final_loss


# ============================================================================
# TRAIN ALL
# ============================================================================
all_results = {}

for n in DATA_SIZES:
    print(f"\n{'='*60}")
    print(f"Training MLP on {DATASET_NAME} with N={n} samples")
    print(f"{'='*60}")
    
    all_results[n] = []
    
    for run in range(N_RUNS):
        print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
        t0 = time.time()
        model, loss_hist, final_loss = train_with_tracking(n, run)
        elapsed = time.time() - t0
        
        if loss_hist:
            print(f"done in {elapsed:.0f}s, final_loss={final_loss:.6f}", flush=True)
        else:
            print(f"loaded in {elapsed:.0f}s, final_loss={final_loss:.6f}", flush=True)
        
        all_results[n].append({
            'model': model,
            'loss_history': loss_hist,
            'final_loss': final_loss
        })

# ============================================================================
# PLOTS
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
for n in DATA_SIZES:
    all_losses = []
    all_steps = []
    for run_data in all_results[n]:
        if run_data['loss_history']:
            steps = [s for s, _ in run_data['loss_history']]
            losses = [l for _, l in run_data['loss_history']]
            all_losses.append(losses)
            all_steps.append(steps)
    if all_losses:
        mean_loss = np.mean(all_losses, axis=0)
        ax.plot(all_steps[0], mean_loss, lw=2, label=f'N={n}')
ax.set_xlabel('Training Step')
ax.set_ylabel('MSE Loss')
ax.set_title(f'MLP Training Loss vs Data Size ({DATASET_NAME})')
ax.legend(fontsize=7, ncol=2)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.savefig(f'figures/mlp_{DATASET_NAME}_training_loss.png', dpi=150, bbox_inches='tight')
print(f"\nSaved figures/mlp_{DATASET_NAME}_training_loss.png")

fig, ax = plt.subplots(figsize=(8, 5))
means = []
stds = []
for n in DATA_SIZES:
    losses = [r['final_loss'] for r in all_results[n]]
    means.append(np.mean(losses))
    stds.append(np.std(losses))
ax.errorbar(DATA_SIZES, means, yerr=stds, marker='o', capsize=5, lw=2)
ax.set_xlabel('Number of Training Samples')
ax.set_ylabel('Final MSE Loss')
ax.set_title(f'MLP Convergence vs Data Size ({DATASET_NAME})')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.savefig(f'figures/mlp_{DATASET_NAME}_final_loss_vs_size.png', dpi=150, bbox_inches='tight')
print(f"Saved figures/mlp_{DATASET_NAME}_final_loss_vs_size.png")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print(f"MLP TRAINING COMPLETE - {DATASET_NAME}")
print(f"{'='*60}")
print(f"  {'N':>8s}  {'Final Loss':>12s}  {'Converged?':>12s}")
print(f"  {'-'*38}")
for n in DATA_SIZES:
    losses = [r['final_loss'] for r in all_results[n]]
    mean_l = np.mean(losses)
    converged = np.std(losses) < 0.1 * mean_l
    print(f"  {n:>8d}  {mean_l:>12.6f}  {'YES' if converged else 'MAYBE':>12s}")

print(f"\nCheckpoints saved to checkpoints/mlp_{DATASET_NAME}_n*_run*.pt")
print(f"Sample images saved to figures/mlp_{DATASET_NAME}_n*_run*_samples.png")
print(f"Now run FTSS tests with: python test_memorization_trend_mlp.py --dataset {DATASET_NAME}")
