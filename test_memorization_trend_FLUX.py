"""
FTSS + Divergence measurement for SD3.5 Medium (flow matching model).
"""

import torch
import numpy as np
import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='a photo of a cat')
parser.add_argument('--model', type=str, default='stabilityai/stable-diffusion-3.5-medium')
parser.add_argument('--n_samples', type=int, default=3)
parser.add_argument('--n_times', type=int, default=8)
parser.add_argument('--n_steps', type=int, default=20)
parser.add_argument('--delta_scale', type=float, default=0.1)
parser.add_argument('--n_pert', type=int, default=2)
parser.add_argument('--n_probes', type=int, default=5, help='Probes for divergence estimator')
parser.add_argument('--output_dir', type=str, default='figures')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
print(f"Loading {args.model}...", flush=True)
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
print("Model loaded.", flush=True)

transformer = pipe.transformer

LATENT_C = 16
LATENT_H = 128
LATENT_W = 128

# ============================================================================
# ENCODE PROMPT
# ============================================================================
print(f"Encoding prompt: '{args.prompt}'", flush=True)

with torch.no_grad():
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt,
        prompt_2=args.prompt,
        prompt_3=args.prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)
pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.bfloat16)

# ============================================================================
# VELOCITY (with gradient support for divergence)
# ============================================================================
def velocity_no_grad(x, t_scalar):
    """Standard forward — no gradients."""
    B = x.shape[0]
    timestep = torch.full((B,), t_scalar * 1000.0, device=x.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = transformer(
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
    return out


def velocity_with_grad(x, t_scalar):
    """Forward with gradients enabled — for divergence computation."""
    B = x.shape[0]
    timestep = torch.full((B,), t_scalar * 1000.0, device=x.device, dtype=torch.bfloat16)
    out = transformer(
        hidden_states=x,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]
    return out


# ============================================================================
# DIVERGENCE ESTIMATOR (Hutchinson)
# ============================================================================
def estimate_divergence(x, t_scalar, n_probes=5):
    """
    Estimate div v(x, t) = tr(∂v/∂x) via Hutchinson: E_u[u^T J u].
    Uses VJP: J^T u = grad of (v · u) w.r.t. x.
    Returns mean over probes.
    """
    div_estimates = []
    
    for _ in range(n_probes):
        # Random probe
        u = torch.randn_like(x)
        u = u / (torch.norm(u) + 1e-8)
        
        # Need gradient w.r.t. input
        x_req = x.detach().requires_grad_(True)
        
        v = velocity_with_grad(x_req, t_scalar)  # (B, 16, 128, 128)
        
        # Compute v · u (scalar)
        s = (v * u).sum()
        
        # Compute gradient: g = ∂(v·u)/∂x = J^T u
        try:
            g = torch.autograd.grad(s, x_req, create_graph=False, retain_graph=False)[0]
        except RuntimeError:
            # Fallback: some ops might not support grad
            div_estimates.append(0.0)
            continue
        
        # tr(J) ≈ u^T J u = u^T (J^T u) = (g * u).sum()
        div = (g * u).sum().item() / x.shape[0]  # average over batch
        div_estimates.append(div)
    
    return float(np.mean(div_estimates)), float(np.std(div_estimates))


# ============================================================================
# GENERATE TRAJECTORY (with divergence tracked)
# ============================================================================
def generate_trajectory(n_steps, seed=None, track_div=False, n_probes=5):
    if seed is not None:
        torch.manual_seed(seed)
    
    z = torch.randn(1, LATENT_C, LATENT_H, LATENT_W, device='cuda', dtype=torch.bfloat16)
    timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device='cuda')
    
    trajectory = [z.clone()]
    divergences = []
    
    if track_div:
        d0, _ = estimate_divergence(z, timesteps[0].item(), n_probes)
        divergences.append(d0)
    
    x = z
    for i in range(n_steps):
        t = timesteps[i].item()
        t_next = timesteps[i+1].item()
        dt = t_next - t
        v = velocity_no_grad(x, t)
        x = x + v * dt
        trajectory.append(x.clone())
        
        if track_div:
            d, _ = estimate_divergence(x, t_next, n_probes)
            divergences.append(d)
    
    return x, trajectory, timesteps, divergences


# ============================================================================
# FTSS MEASUREMENT
# ============================================================================
def measure_ftss(n_samples, n_times, n_steps, delta_scale, n_pert):
    test_times = np.linspace(0.05, 0.95, n_times)
    all_ratios = []
    
    for s in tqdm(range(n_samples), desc="Trajectories"):
        x1_clean, trajectory, timesteps, _ = generate_trajectory(n_steps, seed=s)
        ratios = []
        
        for t_perturb in test_times:
            step = int(t_perturb * n_steps)
            step = min(step, n_steps - 1)
            x_t = trajectory[step]
            
            sum_sq = 0.0
            delta_norm = 0.0
            for _ in range(n_pert):
                delta = torch.randn_like(x_t)
                delta = delta / (torch.norm(delta) + 1e-8)
                delta = delta * delta_scale * torch.norm(x_t)
                delta_norm = torch.norm(delta).item()
                
                x_pert = x_t + delta
                for i in range(step, n_steps):
                    t = timesteps[i].item()
                    t_next = timesteps[i+1].item()
                    dt = t_next - t
                    v = velocity_no_grad(x_pert, t)
                    x_pert = x_pert + v * dt
                
                diff = x_pert - x1_clean
                sum_sq += torch.sum(diff ** 2).item()
            
            rms = np.sqrt(sum_sq / n_pert)
            ratios.append(rms / (delta_norm + 1e-8))
        
        all_ratios.append(ratios)
    
    all_ratios = np.array(all_ratios)
    g_t = all_ratios.mean(axis=0)
    return test_times, g_t, all_ratios


# ============================================================================
# DIVERGENCE PROFILE
# ============================================================================
def measure_divergence_profile(n_samples, n_steps, n_probes):
    """Measure div v along trajectories at each time step."""
    all_divs = []
    
    for s in tqdm(range(n_samples), desc="Divergence"):
        _, _, timesteps, divs = generate_trajectory(
            n_steps, seed=s, track_div=True, n_probes=n_probes
        )
        all_divs.append(divs)
    
    all_divs = np.array(all_divs)  # (n_samples, n_steps + 1)
    div_t = all_divs.mean(axis=0)
    return timesteps.cpu().numpy(), div_t, all_divs


# ============================================================================
# RUN
# ============================================================================
print(f"\nFTSS: {args.n_samples} samples × {args.n_times} times × {args.n_pert} perturb")
test_times, g_t, all_g = measure_ftss(
    args.n_samples, args.n_times, args.n_steps, args.delta_scale, args.n_pert
)

print(f"\nDivergence: {args.n_samples} samples × {args.n_steps} steps × {args.n_probes} probes")
div_times, div_t, all_div = measure_divergence_profile(
    args.n_samples, args.n_steps, args.n_probes
)

# Print FTSS
print(f"\n{'='*60}")
print(f"FTSS RESULTS (SD3.5-Medium)")
print(f"{'='*60}")
for i, t in enumerate(test_times):
    print(f"  t={t:.3f}  g(t)={g_t[i]:.4f} ± {all_g[:, i].std():.4f}")

early = g_t[:len(g_t)//4].mean()
late = g_t[-len(g_t)//4:].mean()
print(f"\n  Early: {early:.4f}, Late: {late:.4f}, Ratio: {early/(late+1e-8):.3f}")

# Print divergence
print(f"\n{'='*60}")
print(f"DIVERGENCE RESULTS (SD3.5-Medium)")
print(f"{'='*60}")
for i, t in enumerate(div_times):
    print(f"  t={t:.3f}  div v={div_t[i]:+.4f} ± {all_div[:, i].std():.4f}")

div_early = div_t[:len(div_t)//4].mean()
div_late = div_t[-len(div_t)//4:].mean()
div_overall = div_t.mean()
cumdiv = np.trapz(div_t, div_times)  # ∫ div v dt

print(f"\n  Early: {div_early:+.4f}")
print(f"  Late: {div_late:+.4f}")
print(f"  Overall: {div_overall:+.4f}")
print(f"  Cumulative ∫div v dt: {cumdiv:+.4f}")
print(f"  ⇒ Expected log volume change: {cumdiv:+.4f}")
print(f"  ⇒ Volume ratio = exp({cumdiv:+.4f}) = {np.exp(cumdiv):.4f}")

# ============================================================================
# PLOT
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
for i in range(min(args.n_samples, 5)):
    ax.plot(test_times, all_g[i], alpha=0.4, lw=1)
ax.plot(test_times, g_t, 'k-', lw=3, label='Mean')
ax.fill_between(test_times, g_t - all_g.std(axis=0), g_t + all_g.std(axis=0),
                alpha=0.2, color='k')
ax.axhline(1.0, color='gray', ls='--', label='g=1')
ax.set_xlabel('Time $t$')
ax.set_ylabel('FTSS $g(t)$')
ax.set_title(f'FTSS Profile (SD3.5-Medium)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for i in range(min(args.n_samples, 5)):
    ax.plot(div_times, all_div[i], alpha=0.4, lw=1)
ax.plot(div_times, div_t, 'r-', lw=3, label='Mean')
ax.fill_between(div_times, div_t - all_div.std(axis=0), div_t + all_div.std(axis=0),
                alpha=0.2, color='r')
ax.axhline(0.0, color='gray', ls='--', label='div v = 0')
ax.set_xlabel('Time $t$')
ax.set_ylabel(r'$\nabla \cdot v$')
ax.set_title(f'Divergence Profile (∫ = {cumdiv:+.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(args.output_dir, 'ftss_div_sd35.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved {out}")

np.savez(os.path.join(args.output_dir, 'ftss_div_sd35_data.npz'),
         test_times=test_times, g_t=g_t, all_g=all_g,
         div_times=div_times, div_t=div_t, all_div=all_div,
         prompt=args.prompt, cumdiv=cumdiv)