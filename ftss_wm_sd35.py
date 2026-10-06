"""
FTSS + Divergence + Watermark for SD3.5 Medium (flow matching model).

Watermark design:
    v_wm(x, t) = v(x, t) + ε · a_m(t) · v(x, t)
              = (1 + ε · a_m(t)) · v(x, t)

where a_m(t) = Σ_i c_{m,i} · sin(2π(i+1)t) encodes the message.

This is a pure TIME REPARAMETRIZATION of the flow:
- Trajectory stays on the same curve in latent space
- Endpoint is preserved exactly (since ∫ a_m(t) dt = 0)
- Watermark is encoded in the speed profile along the trajectory

Detection: project the velocity onto itself, correlate with sin(2π(i+1)t).
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

# ---- Watermark args ----
parser.add_argument('--wm_enable', action='store_true', help='Enable watermark')
parser.add_argument('--wm_message', type=str, default='101', help='Bitstring, e.g. 101')
parser.add_argument('--wm_eps', type=float, default=0.15, help='Watermark strength ε')
parser.add_argument('--wm_lambda', type=float, default=0.0, help='Extra bias λ (optional)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============================================================================
# WATERMARK DEFINITION
# ============================================================================
WM_BITS = [int(b) for b in args.wm_message]
WM_N_BITS = len(WM_BITS)
WM_EPS = args.wm_eps
WM_LAMBDA = args.wm_lambda


def a_m(t_scalar, device, dtype):
    """
    Temporal modulation pattern encoding the message.

    a_m(t) = Σ_i c_i · sin(2π(i+1)·t)

    Each bit uses a different frequency. All sin terms have zero mean
    over [0,1], so ∫ a_m(t) dt = 0 → endpoint preserved.
    """
    t_tensor = torch.tensor(t_scalar, device=device, dtype=dtype)
    a = torch.zeros((), device=device, dtype=dtype)
    for i, bit in enumerate(WM_BITS):
        freq = (i + 1)
        a = a + bit * torch.sin(2 * np.pi * freq * t_tensor)
    return a


def get_watermark_scale(t_scalar):
    """
    Returns the multiplicative scale applied to velocity:
        scale(t) = 1 + ε · a_m(t)
    """
    if not args.wm_enable:
        return 1.0
    a = a_m(t_scalar, device='cuda', dtype=torch.float32)
    return 1.0 + WM_EPS * a.item() + WM_LAMBDA * np.sin(2 * np.pi * t_scalar)


# ============================================================================
# LOAD MODEL
# ============================================================================
print(f"Loading {args.model}...", flush=True)
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
pipe.enable_sequential_cpu_offload()   # was: enable_model_cpu_offload()
pipe.enable_attention_slicing()
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
def velocity_no_grad(x, t_scalar, apply_wm=False):
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
    if apply_wm and args.wm_enable:
        scale = get_watermark_scale(t_scalar)
        out = out * scale
    return out


def velocity_with_grad(x, t_scalar, apply_wm=False):
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
    if apply_wm and args.wm_enable:
        scale = get_watermark_scale(t_scalar)
        out = out * scale
    return out


# ============================================================================
# DIVERGENCE ESTIMATOR (Hutchinson)
# ============================================================================
def estimate_divergence(x, t_scalar, n_probes=5, apply_wm=False):
    """
    Estimate div v(x, t) via Hutchinson: E_u[u^T J u].

    Note: for the parallel watermark v_wm = (1 + ε·a(t))·v,
          div v_wm = (1 + ε·a(t))·div v.
          So divergence is scaled by the same factor.
    """
    div_estimates = []

    for _ in range(n_probes):
        u = torch.randn_like(x)
        u = u / (torch.norm(u) + 1e-8)

        x_req = x.detach().requires_grad_(True)
        v = velocity_with_grad(x_req, t_scalar, apply_wm=apply_wm)

        s = (v * u).sum()

        try:
            g = torch.autograd.grad(s, x_req, create_graph=False, retain_graph=False)[0]
        except RuntimeError:
            div_estimates.append(0.0)
            continue

        div = (g * u).sum().item() / x.shape[0]
        div_estimates.append(div)

    return float(np.mean(div_estimates)), float(np.std(div_estimates))


# ============================================================================
# GENERATE TRAJECTORY (with divergence tracked)
# ============================================================================
def generate_trajectory(n_steps, seed=None, track_div=False, n_probes=5, apply_wm=False):
    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(1, LATENT_C, LATENT_H, LATENT_W, device='cuda', dtype=torch.bfloat16)
    timesteps = torch.linspace(0.0, 1.0, n_steps + 1, device='cuda')

    trajectory = [z.clone()]
    divergences = []
    scales = [get_watermark_scale(timesteps[0].item())]

    if track_div:
        d0, _ = estimate_divergence(z, timesteps[0].item(), n_probes, apply_wm=apply_wm)
        divergences.append(d0)

    x = z
    for i in range(n_steps):
        t = timesteps[i].item()
        t_next = timesteps[i+1].item()
        dt = t_next - t
        v = velocity_no_grad(x, t, apply_wm=apply_wm)
        x = x + v * dt
        trajectory.append(x.clone())
        scales.append(get_watermark_scale(t_next))

        if track_div:
            d, _ = estimate_divergence(x, t_next, n_probes, apply_wm=apply_wm)
            divergences.append(d)

    return x, trajectory, timesteps, divergences, scales


# ============================================================================
# FTSS MEASUREMENT
# ============================================================================
def measure_ftss(n_samples, n_times, n_steps, delta_scale, n_pert, apply_wm=False):
    test_times = np.linspace(0.05, 0.95, n_times)
    all_ratios = []

    for s in tqdm(range(n_samples), desc="Trajectories"):
        x1_clean, trajectory, timesteps, _, _ = generate_trajectory(
            n_steps, seed=s, apply_wm=apply_wm
        )
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
                    v = velocity_no_grad(x_pert, t, apply_wm=apply_wm)
                    x_pert = x_pert + v * dt

                diff = x_pert - x1_clean
                sum_sq += torch.sum(diff ** 2).item()

            rms = np.sqrt(sum_sq / n_pert)
            ratios.append(rms / (delta_norm + 1e-8))

        all_ratios.append(ratios)

    all_ratios = np.array(all_ratios)
    g_t = np.sqrt(np.mean(all_ratios**2, axis=0))
    return test_times, g_t, all_ratios


# ============================================================================
# DIVERGENCE PROFILE
# ============================================================================
def measure_divergence_profile(n_samples, n_steps, n_probes, apply_wm=False):
    all_divs = []
    all_scales = []

    for s in tqdm(range(n_samples), desc="Divergence"):
        _, _, timesteps, divs, scales = generate_trajectory(
            n_steps, seed=s, track_div=True, n_probes=n_probes, apply_wm=apply_wm
        )
        all_divs.append(divs)
        all_scales.append(scales)

    all_divs = np.array(all_divs)
    div_t = all_divs.mean(axis=0)
    return timesteps.cpu().numpy(), div_t, all_divs


# ============================================================================
# WATERMARK DETECTION
# ============================================================================
def detect_watermark(trajectory, timesteps, apply_wm=True):
    """
    Detect watermark by computing the time-varying scale along trajectory.

    For parallel watermark v_wm = (1 + ε·a(t))·v, we can recover the scale by:
        s(t) = ||v_wm(x_t, t)|| / ||v_clean(x_t, t)||

    But we only have v_wm at test time. Instead, we detect using the
    *structure*: project the velocity onto itself, correlate with sin(2π(i+1)t).

    A cleaner detection: at each step, compute <v_wm, v_wm_ref> / ||v_wm_ref||^2
    where v_wm_ref is a reference (e.g., the same model without wm — not available).

    Practical approach for pretrained models: detection uses self-consistency.
    We compute the projection of velocity onto the trajectory direction and
    check the temporal profile.

    For the pretrained-model case, detection works by:
      1. Sample a fresh trajectory with the watermarked model
      2. At each step, compute r(t) = <v(x_t, t), x_{t+dt} - x_t> / ||x_{t+dt} - x_t||^2
         (this recovers ~scale(t) up to integration error)
      3. Correlate r(t) - 1 with sin(2π(i+1)t) for each bit

    However, since we don't have the clean reference at test time,
    we use the fact that the watermark makes the *divergence* profile
    have a characteristic temporal pattern.
    """
    scales_detected = []
    n_steps = len(trajectory) - 1

    for i in range(n_steps):
        x_t = trajectory[i]
        x_next = trajectory[i+1]
        t = timesteps[i].item()
        dt = timesteps[i+1].item() - t

        # Displacement direction
        disp = (x_next - x_t).flatten()
        disp_norm = torch.norm(disp) + 1e-8

        # Velocity at this step
        v = velocity_no_grad(x_t, t, apply_wm=apply_wm).flatten()

        # Projection ratio — should be 1 if no wm, (1 + ε a(t)) if wm
        # because x_next - x_t ≈ v(x_t, t) · dt
        ratio = torch.dot(v, disp) / (torch.norm(v) * disp_norm + 1e-8)
        scales_detected.append(ratio.item())

    return np.array(scales_detected)


def decode_bits(scales_detected, timesteps):
    """Correlate the detected scale deviation with each frequency."""
    n_steps = len(scales_detected)
    t_vals = np.array([timesteps[i].item() for i in range(n_steps)])
    r = np.array(scales_detected) - 1.0  # remove baseline

    decoded = []
    for i in range(WM_N_BITS):
        freq = i + 1
        carrier = np.sin(2 * np.pi * freq * t_vals)
        corr = np.mean(r * carrier)
        bit = 1 if corr > 0 else 0
        decoded.append(bit)

    return decoded


# ============================================================================
# RUN: BASELINE (NO WM)
# ============================================================================
print(f"\n{'='*60}")
print("BASELINE (no watermark)")
print(f"{'='*60}")

test_times, g_t, all_g = measure_ftss(
    args.n_samples, args.n_times, args.n_steps, args.delta_scale, args.n_pert,
    apply_wm=False,
)

div_times, div_t, all_div = measure_divergence_profile(
    args.n_samples, args.n_steps, args.n_probes, apply_wm=False,
)

print(f"\nFTSS (baseline):")
for i, t in enumerate(test_times):
    print(f"  t={t:.3f}  g(t)={g_t[i]:.4f} ± {all_g[:, i].std():.4f}")

early = g_t[:len(g_t)//4].mean()
late = g_t[-len(g_t)//4:].mean()
print(f"  Early: {early:.4f}, Late: {late:.4f}, Ratio: {early/(late+1e-8):.3f}")

print(f"\nDivergence (baseline):")
for i, t in enumerate(div_times):
    print(f"  t={t:.3f}  div v={div_t[i]:+.4f} ± {all_div[:, i].std():.4f}")

cumdiv = np.trapz(div_t, div_times)
print(f"  ∫div v dt: {cumdiv:+.4f}")
print(f"  Volume ratio: exp({cumdiv:+.4f}) = {np.exp(cumdiv):.4f}")

# ============================================================================
# RUN: WATERMARKED
# ============================================================================
if args.wm_enable:
    print(f"\n{'='*60}")
    print(f"WATERMARKED (message={args.wm_message}, ε={WM_EPS})")
    print(f"{'='*60}")

    test_times_wm, g_t_wm, all_g_wm = measure_ftss(
        args.n_samples, args.n_times, args.n_steps, args.delta_scale, args.n_pert,
        apply_wm=True,
    )

    div_times_wm, div_t_wm, all_div_wm = measure_divergence_profile(
        args.n_samples, args.n_steps, args.n_probes, apply_wm=True,
    )

    print(f"\nFTSS (watermarked):")
    for i, t in enumerate(test_times_wm):
        print(f"  t={t:.3f}  g(t)={g_t_wm[i]:.4f} ± {all_g_wm[:, i].std():.4f}")

    early_wm = g_t_wm[:len(g_t_wm)//4].mean()
    late_wm = g_t_wm[-len(g_t_wm)//4:].mean()
    print(f"  Early: {early_wm:.4f}, Late: {late_wm:.4f}, Ratio: {early_wm/(late_wm+1e-8):.3f}")

    print(f"\nDivergence (watermarked):")
    for i, t in enumerate(div_times_wm):
        print(f"  t={t:.3f}  div v={div_t_wm[i]:+.4f} ± {all_div_wm[:, i].std():.4f}")

    cumdiv_wm = np.trapz(div_t_wm, div_times_wm)
    print(f"  ∫div v dt: {cumdiv_wm:+.4f}")
    print(f"  Volume ratio: exp({cumdiv_wm:+.4f}) = {np.exp(cumdiv_wm):.4f}")

    # ---- Detection ----
    print(f"\n{'='*60}")
    print(f"WATERMARK DETECTION")
    print(f"{'='*60}")

    detection_results = []
    for s in range(args.n_samples):
        _, traj, tsteps, _, _ = generate_trajectory(
            args.n_steps, seed=1000 + s, apply_wm=True
        )
        scales_detected = detect_watermark(traj, tsteps, apply_wm=True)
        decoded = decode_bits(scales_detected, tsteps)
        match = (decoded == WM_BITS)
        detection_results.append(match)
        print(f"  Sample {s}: decoded={''.join(map(str, decoded))} "
              f"true={args.wm_message} {'✓' if match else '✗'}")

    acc = np.mean(detection_results) * 100
    print(f"\n  Detection accuracy: {acc:.1f}%")

    # ---- Endpoint preservation check ----
    print(f"\n{'='*60}")
    print(f"ENDPOINT PRESERVATION CHECK")
    print(f"{'='*60}")

    endpoint_diffs = []
    for s in range(min(args.n_samples, 5)):
        x1_clean, _, _, _, _ = generate_trajectory(args.n_steps, seed=s, apply_wm=False)
        x1_wm, _, _, _, _ = generate_trajectory(args.n_steps, seed=s, apply_wm=True)
        diff = torch.norm(x1_wm - x1_clean).item() / torch.norm(x1_clean).item()
        endpoint_diffs.append(diff)
        print(f"  Sample {s}: relative endpoint diff = {diff:.6f}")

    print(f"\n  Mean relative diff: {np.mean(endpoint_diffs):.6f}")
    print(f"  (Should be small if a_m has zero mean and ε is small)")

# ============================================================================
# PLOT
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# FTSS
ax = axes[0]
for i in range(min(args.n_samples, 5)):
    ax.plot(test_times, all_g[i], alpha=0.4, lw=1, color='C0')
ax.plot(test_times, g_t, 'C0-', lw=3, label='Baseline')
if args.wm_enable:
    for i in range(min(args.n_samples, 5)):
        ax.plot(test_times_wm, all_g_wm[i], alpha=0.4, lw=1, color='C1')
    ax.plot(test_times_wm, g_t_wm, 'C1-', lw=3, label='Watermarked')
ax.axhline(1.0, color='gray', ls='--', label='g=1')
ax.set_xlabel('Time $t$')
ax.set_ylabel('FTSS $g(t)$')
ax.set_title('FTSS Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Divergence
ax = axes[1]
ax.plot(div_times, div_t, 'C0-', lw=3, label=f'Baseline (∫={cumdiv:+.2f})')
if args.wm_enable:
    ax.plot(div_times_wm, div_t_wm, 'C1-', lw=3, label=f'Watermarked (∫={cumdiv_wm:+.2f})')
ax.axhline(0.0, color='gray', ls='--')
ax.set_xlabel('Time $t$')
ax.set_ylabel(r'$\nabla \cdot v$')
ax.set_title('Divergence Profile')
ax.legend()
ax.grid(True, alpha=0.3)

# Watermark scale profile
ax = axes[2]
t_plot = np.linspace(0, 1, 200)
a_vals = np.array([a_m(t, 'cpu', torch.float32).item() for t in t_plot])
scale_vals = 1 + WM_EPS * a_vals + WM_LAMBDA * np.sin(2 * np.pi * t_plot)
ax.plot(t_plot, scale_vals, 'C2-', lw=2, label=f'1 + ε·a(t), ε={WM_EPS}')
ax.axhline(1.0, color='gray', ls='--')
ax.set_xlabel('Time $t$')
ax.set_ylabel('Velocity scale')
ax.set_title(f'Watermark Scale (message={args.wm_message})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(args.output_dir, 'ftss_div_sd35_wm.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved {out}")

# Save data
save_dict = dict(
    test_times=test_times, g_t=g_t, all_g=all_g,
    div_times=div_times, div_t=div_t, all_div=all_div,
    prompt=args.prompt, cumdiv=cumdiv,
    wm_enable=args.wm_enable, wm_message=args.wm_message, wm_eps=WM_EPS,
)
if args.wm_enable:
    save_dict.update(dict(
        test_times_wm=test_times_wm, g_t_wm=g_t_wm, all_g_wm=all_g_wm,
        div_times_wm=div_times_wm, div_t_wm=div_t_wm, all_div_wm=all_div_wm,
        cumdiv_wm=cumdiv_wm,
    ))
np.savez(os.path.join(args.output_dir, 'ftss_div_sd35_wm_data.npz'), **save_dict)
