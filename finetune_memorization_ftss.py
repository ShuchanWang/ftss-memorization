"""Fine-tuning study: generated-image replication versus FTSS on MNIST.

The replication rule is an operational measure inspired by nearest-neighbor
training-image retrieval in Somepalli et al. (CVPR 2023) and Carlini et al.
(USENIX Security 2023). Pixel L2 is suitable for this aligned MNIST pilot;
it should be replaced by a validated copy detector for natural images.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms


class MLPFlow(nn.Module):
    def __init__(self, dimension=784, hidden_dim=2048):
        super().__init__()
        layers = []
        width = dimension + 1
        for _ in range(5):
            layers.extend([nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim),
                           nn.SiLU(), nn.Dropout(0.1)])
            width = hidden_dim
        layers.append(nn.Linear(width, dimension))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


@torch.no_grad()
def integrate(model, x, start_step, steps):
    for step in range(start_step, steps):
        t = x.new_full((len(x), 1), step / steps)
        velocity = model(x, t).clamp(-10, 10)
        x = (x + velocity / steps).clamp(-5, 5)
    return x


@torch.no_grad()
def generate(model, noise, steps, batch_size):
    return torch.cat([
        integrate(model, noise[i:i + batch_size], 0, steps).cpu()
        for i in range(0, len(noise), batch_size)
    ])


@torch.no_grad()
def ftss(model, noise, directions, steps, time_points, epsilon_scale):
    """Return RMS finite-difference gain at each time point and its minimum."""
    trajectory = [noise]
    x = noise
    for step in range(steps):
        t = x.new_full((len(x), 1), step / steps)
        x = (x + model(x, t).clamp(-10, 10) / steps).clamp(-5, 5)
        trajectory.append(x)
    terminal = trajectory[-1]

    curve = []
    for step in np.linspace(0, steps - 1, time_points, dtype=int):
        state = trajectory[step]
        unit = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        epsilon = epsilon_scale * state.norm(dim=1, keepdim=True)
        perturbed = integrate(model, state + epsilon * unit, int(step), steps)
        gain = (perturbed - terminal).norm(dim=1) / epsilon.squeeze(1).clamp_min(1e-12)
        curve.append(gain.square().mean().sqrt().item())
    return curve, float(min(curve))


@torch.no_grad()
def nearest_distances(queries, pool, batch_size=256):
    values = []
    for i in range(0, len(queries), batch_size):
        values.append(torch.cdist(queries[i:i + batch_size], pool).min(dim=1).values)
    return torch.cat(values).numpy()


@torch.no_grad()
def flow_loss(model, images, noise, times, device):
    target = images.to(device)
    noise = noise[:len(target)]
    times = times[:len(target)]
    state = (1 - times) * noise + times * target
    return float((model(state, times) - (target - noise)).square().mean().item())


def evaluate(model, generation_noise, probe_noise, directions, tuned, reference,
             calibration, loss_noise, loss_times, args):
    model.eval()
    samples = generate(model, generation_noise, args.ode_steps, args.batch_size)
    train_distance = nearest_distances(samples, tuned)
    reference_distance = nearest_distances(samples, reference)

    # Calibrate a 1% natural-neighbor rate using examples never used to tune.
    threshold = float(np.quantile(nearest_distances(calibration, tuned), 0.01))
    reference_threshold = float(np.quantile(nearest_distances(calibration, reference), 0.01))
    train_matches = (train_distance < threshold) & (train_distance < reference_distance)
    reference_matches = (reference_distance < reference_threshold) & (reference_distance < train_distance)
    pairwise = torch.cdist(samples, samples)
    pairwise.fill_diagonal_(float("inf"))
    curve, g_min = ftss(model, probe_noise, directions, args.ode_steps,
                        args.time_points, args.epsilon_scale)
    return {
        "replication_rate": float(train_matches.mean()),
        "reference_match_rate": float(reference_matches.mean()),
        "mean_train_distance": float(train_distance.mean()),
        "mean_reference_distance": float(reference_distance.mean()),
        "generated_nn_distance": float(pairwise.min(dim=1).values.mean().item()),
        "tuned_flow_loss": flow_loss(model, tuned, loss_noise, loss_times, generation_noise.device),
        "reference_flow_loss": flow_loss(model, reference, loss_noise, loss_times,
                                          generation_noise.device),
        "copy_threshold": threshold,
        "g_min": g_min,
        "ftss_curve": curve,
    }


def load_base(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if checkpoint.get("dataset") != "mnist" or config.get("D", 784) != 784:
        raise ValueError("The base checkpoint must be an MNIST MLP checkpoint")
    model = MLPFlow(784, config.get("hidden_dim", 2048)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def correlations(rows):
    """Associate within-arm FTSS changes with within-arm replication changes."""
    results = {}
    for size in sorted({row["size"] for row in rows}):
        pairs = []
        for run in sorted({row["run"] for row in rows}):
            arm = sorted((row for row in rows if row["size"] == size and row["run"] == run),
                         key=lambda row: row["step"])
            if not arm or arm[0]["step"] != 0:
                continue
            baseline = arm[0]
            for row in arm[1:]:
                pairs.append((row["g_min"] / baseline["g_min"] - 1,
                              row["replication_rate"] - baseline["replication_rate"]))
        if len(pairs) < 3:
            results[str(size)] = {"n_checkpoints": len(pairs), "pearson_r": None}
            continue
        values = np.asarray(pairs)
        x, y = values[:, 0], values[:, 1]
        denominator = np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2))
        results[str(size)] = {
            "n_checkpoints": len(pairs),
            "pearson_r": float(np.sum((x - x.mean()) * (y - y.mean())) / denominator)
            if denominator > 0 else None,
            "mean_ftss_relative_change": float(x.mean()),
            "mean_replication_rate_change": float(y.mean()),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path,
                        default=Path("checkpoints/mlp_mnist_n12000_run0.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("finetune_ftss_results"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-generated", type=int, default=256)
    parser.add_argument("--n-probes", type=int, default=15)
    parser.add_argument("--time-points", type=int, default=20)
    parser.add_argument("--ode-steps", type=int, default=200)
    parser.add_argument("--epsilon-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.runs < 1 or args.steps < 1 or args.eval_every < 1:
        parser.error("sizes must be >= 2 and runs, steps, eval-every must be positive")
    if max(args.sizes) * 2 + 1000 > 10000:
        parser.error("MNIST test set must contain two equal pools plus 1000 calibration images")
    if args.epsilon_scale <= 0 or args.ode_steps < 2 or args.time_points < 2:
        parser.error("epsilon-scale must be positive; ode-steps and time-points must be >= 2")
    if not args.base_checkpoint.is_file():
        parser.error(f"base checkpoint not found: {args.base_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)
    images = torch.stack([dataset[i][0].flatten() for i in range(len(dataset))])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {**vars(args), "base_checkpoint": str(args.base_checkpoint),
                "data_root": str(args.data_root), "output_dir": str(args.output_dir),
                "operational_definition": "Generated image is nearer to a fine-tuning image than to an equal-size untouched pool and below the 1% held-out calibration distance."}
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2))

    rows = []
    for run in range(args.runs):
        seed = args.seed + 1000 * run
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(dataset))
        generator = torch.Generator(device=device).manual_seed(seed)
        generation_noise = torch.randn(args.n_generated, 784, device=device, generator=generator)
        probe_noise = torch.randn(args.n_probes, 784, device=device, generator=generator)
        directions = torch.randn(args.n_probes, 784, device=device, generator=generator)
        loss_noise = torch.randn(max(args.sizes), 784, device=device, generator=generator)
        loss_times = torch.rand(max(args.sizes), 1, device=device, generator=generator)

        for size in args.sizes:
            tuned_indices = order[:size]
            reference_indices = order[size:2 * size]
            calibration_indices = order[2 * size:2 * size + 1000]
            tuned = images[tuned_indices]
            reference = images[reference_indices]
            calibration = images[calibration_indices]
            (args.output_dir / f"indices_n{size}_run{run}.json").write_text(json.dumps({
                "tuned": tuned_indices.tolist(), "reference": reference_indices.tolist(),
                "calibration": calibration_indices.tolist(),
            }))
            pool = tuned.to(device)
            model = load_base(args.base_checkpoint, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            baseline_min = None

            def record(step):
                nonlocal baseline_min
                metrics = evaluate(model, generation_noise, probe_noise, directions,
                                   tuned, reference, calibration, loss_noise, loss_times, args)
                if baseline_min is None:
                    baseline_min = metrics["g_min"]
                metrics["ftss_ratio_to_base"] = metrics["g_min"] / baseline_min
                row = {"run": run, "size": size, "step": step, **metrics}
                rows.append(row)
                print(f"run={run} N={size} step={step} copy={metrics['replication_rate']:.3f} "
                      f"control={metrics['reference_match_rate']:.3f} g_min={metrics['g_min']:.3f}",
                      flush=True)

            record(0)
            for step in range(1, args.steps + 1):
                model.train()
                selected = torch.randint(size, (min(args.batch_size, size),),
                                         generator=generator, device=device)
                x1 = pool[selected]
                x0 = torch.randn(x1.shape, device=device, generator=generator)
                t = torch.rand(len(x1), 1, device=device, generator=generator)
                prediction = model((1 - t) * x0 + t * x1, t)
                loss = (prediction - (x1 - x0)).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if step % args.eval_every == 0 or step == args.steps:
                    record(step)

            # Persist each completed arm so interrupted studies retain results.
            with (args.output_dir / "measurements.jsonl").open("w") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")

    with (args.output_dir / "measurements.csv").open("w", newline="") as output:
        fields = [name for name in rows[0] if name != "ftss_curve"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = correlations(rows)
    (args.output_dir / "correlations.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(rows)} measurements to {args.output_dir}")


if __name__ == "__main__":
    main()
