"""
Train DiT models at multiple data sizes to measure memorization vs FTSS.
Supports MNIST, CIFAR-10, CIFAR-100, and ImageNet (imagenette/tiny).
Automatically skips training if checkpoint exists.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import math
import time
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                    help='Dataset to train on')
parser.add_argument('--imagenet_path', type=str, default='./data/imagenette2',
                    help='Path to imagenette/imagenet dataset')
parser.add_argument('--img_size', type=int, default=None,
                    help='Image size (auto-detected from dataset)')
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
    },
    'cifar10': {
        'in_channels': 3,
        'img_size': 32,
        'D': 3072,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000],
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
    },
    'cifar100': {
        'in_channels': 3,
        'img_size': 32,
        'D': 3072,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000, 12000],
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
    },
    'imagenet': {
        'in_channels': 3,
        'img_size': 64,
        'D': 12288,
        'data_sizes': [50, 100, 200, 500, 1000, 2000, 5000],
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
    },
}

if args.img_size is not None:
    DATASET_CONFIGS[DATASET_NAME]['img_size'] = args.img_size
    DATASET_CONFIGS[DATASET_NAME]['D'] = args.img_size * args.img_size * DATASET_CONFIGS[DATASET_NAME]['in_channels']

cfg = DATASET_CONFIGS[DATASET_NAME]
IN_CHANNELS = cfg['in_channels']
IMG_SIZE = cfg['img_size']
D = cfg['D']
DATA_SIZES = cfg['data_sizes']

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STEPS = 8000
BATCH_SIZE = 64
LR = 5e-4
N_RUNS = 2

PATCH_SIZE = 4
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Dataset: {DATASET_NAME}, Image size: {IMG_SIZE}, D: {D}")
print(f"Data sizes: {DATA_SIZES}")
print(f"Architecture: DiT (patch_size={PATCH_SIZE}, dim={HIDDEN_DIM}, heads={NUM_HEADS}, layers={NUM_LAYERS})")
print(f"Steps: {STEPS}, Batch size: {BATCH_SIZE}, LR: {LR}")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def get_dataset(dataset_name, imagenet_path='./data/imagenette2'):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
        return datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
        return datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
        return datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
        if os.path.exists(imagenet_path):
            return datasets.ImageFolder(root=os.path.join(imagenet_path, 'train'), transform=transform)
        elif os.path.exists('./data/tiny-imagenet-200/train'):
            return datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        else:
            print("Downloading imagenette (10-class ImageNet subset)...")
            import subprocess
            subprocess.run(['wget', 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz', '-P', './data/'])
            subprocess.run(['tar', '-xzf', './data/imagenette2.tgz', '-C', './data/'])
            return datasets.ImageFolder(root='./data/imagenette2/train', transform=transform)

print(f"\nLoading {DATASET_NAME} dataset...")
train_ds = get_dataset(DATASET_NAME, args.imagenet_path)
print(f"Training samples available: {len(train_ds)}")

DATA_SIZES = [n for n in DATA_SIZES if n <= len(train_ds)]
print(f"Effective data sizes: {DATA_SIZES}")

# ============================================================================
# MODEL
# ============================================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
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
        x = x.reshape(B, n_patches_per_side, n_patches_per_side, 
                      self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.in_channels, self.img_size, self.img_size)
        
        return x.reshape(B, -1)


# ============================================================================
# DIAGNOSTIC
# ============================================================================
def test_model_output(model):
    model.eval()
    with torch.no_grad():
        B = 4
        x = torch.randn(B, D, device=device)
        t = torch.rand(B, 1, device=device)
        out = model(x, t)
        
        print(f"  Input shape: {x.shape}, Output shape: {out.shape}")
        print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
        print(f"  Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
        
        assert out.shape == (B, D), f"Shape mismatch: expected ({B}, {D}), got {out.shape}"
        assert out.std() > 0.01, f"Output std too small: {out.std().item():.6f}"
        print(f"  OK: Model output test passed")


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
    ckpt_path = f"checkpoints/dit_{DATASET_NAME}_n{n_samples}_run{run_id}.pt"
    
    if os.path.exists(ckpt_path):
        print(f"  Checkpoint exists, loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg_saved = ckpt['config']
        model = SimpleDiT(
            img_size=cfg_saved['img_size'],
            patch_size=cfg_saved['patch_size'],
            in_channels=cfg_saved['in_channels'],
            hidden_dim=cfg_saved['hidden_dim'],
            num_heads=cfg_saved['num_heads'],
            num_layers=cfg_saved['num_layers'],
            dropout=cfg_saved.get('dropout', 0.1)
        ).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        
        vis_path = f"figures/dit_{DATASET_NAME}_n{n_samples}_run{run_id}_samples.png"
        if not os.path.exists(vis_path):
            visualize_samples(model, vis_path)
        
        loss_history = ckpt.get('loss_history', [])
        final_loss = ckpt['final_loss']
        return model, loss_history, final_loss
    
    torch.manual_seed(SEED + run_id * 1000)
    model = SimpleDiT(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
        hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, 
        num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    if run_id == 0 and n_samples == DATA_SIZES[0]:
        print(f"  Running model diagnostic...")
        test_model_output(model)
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    
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
        idx = torch.randperm(len(x1_pool))[:min(BATCH_SIZE, len(x1_pool))]
        x1 = x1_pool[idx]
        bs = x1.shape[0]
        
        x0 = torch.randn(bs, D, device=device)
        t_val = torch.rand(bs, 1, device=device)
        x_t = (1 - t_val) * x0 + t_val * x1
        u_true = x1 - x0
        
        pred = model(x_t, t_val)
        loss = F.mse_loss(pred, u_true)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if step % log_every == 0:
            loss_history.append((step, loss.item()))
    
    final_loss = loss.item()
    
    vis_path = f"figures/dit_{DATASET_NAME}_n{n_samples}_run{run_id}_samples.png"
    visualize_samples(model, vis_path)
    
    torch.save({
        'model': model.state_dict(), 
        'n_samples': n_samples,
        'dataset': DATASET_NAME,
        'final_loss': final_loss,
        'best_loss': best_loss,
        'loss_history': loss_history,
        'config': {
            'img_size': IMG_SIZE,
            'patch_size': PATCH_SIZE,
            'in_channels': IN_CHANNELS,
            'hidden_dim': HIDDEN_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
        }
    }, ckpt_path)
    
    return model, loss_history, final_loss


# ============================================================================
# TRAIN ALL
# ============================================================================
all_results = {}

for n in DATA_SIZES:
    print(f"\n{'='*60}")
    print(f"Training DiT on {DATASET_NAME} with N={n} samples")
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
ax.set_title(f'DiT Training Loss vs Data Size ({DATASET_NAME})')
ax.legend(fontsize=7, ncol=2)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.savefig(f'figures/dit_{DATASET_NAME}_training_loss.png', dpi=150, bbox_inches='tight')
print(f"\nSaved figures/dit_{DATASET_NAME}_training_loss.png")

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
ax.set_title(f'DiT Convergence vs Data Size ({DATASET_NAME})')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.savefig(f'figures/dit_{DATASET_NAME}_final_loss_vs_size.png', dpi=150, bbox_inches='tight')
print(f"Saved figures/dit_{DATASET_NAME}_final_loss_vs_size.png")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*60}")
print(f"DiT TRAINING COMPLETE - {DATASET_NAME}")
print(f"{'='*60}")
print(f"  {'N':>8s}  {'Final Loss':>12s}  {'Converged?':>12s}")
print(f"  {'-'*38}")
for n in DATA_SIZES:
    losses = [r['final_loss'] for r in all_results[n]]
    mean_l = np.mean(losses)
    converged = np.std(losses) < 0.1 * mean_l
    print(f"  {n:>8d}  {mean_l:>12.6f}  {'YES' if converged else 'MAYBE':>12s}")

print(f"\nCheckpoints saved to checkpoints/dit_{DATASET_NAME}_n*_run*.pt")
print(f"Sample images saved to figures/dit_{DATASET_NAME}_n*_run*_samples.png")
print(f"Now run FTSS tests with: python test_memorization_trend_dit.py --dataset {DATASET_NAME}")