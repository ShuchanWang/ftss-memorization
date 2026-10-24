# FTSS During Fine-Tuning of the Existing MNIST MLP

The study completed 54 checkpoint evaluations over three seeds. It starts
from this project's saved clean MNIST MLP, rather than training a new base.
Both arms use the same fine-tuning examples, minibatches, noise, optimizer,
and learning rate. The watermark arm adds the existing five-bit orthogonal
target and carrier-correlation objective. Original checkpoints are unchanged.

## Protocol

- Fine-tuning sets: 50 or 1,000 images from the MNIST test split, which was
  excluded from original base training. Each has an equal-size untouched
  reference pool and 1,000 disjoint calibration images.
- AdamW, learning rate 1e-4, 1,000 steps; evaluations at 0, 100, 300, and 1,000.
- The sibling repo's `finetune_memorization_ftss.py` supplies the actual
  FTSS, generation, nearest-neighbor, and copying-rule implementations.
- FTSS uses 15 trajectories, 20 time points, 200 Euler steps, relative
  perturbation 0.1, and the upstream velocity/state clamps.
- Ratios divide by the original pretrained clean checkpoint's FTSS minimum
  using paired probes. This is a pre-fine-tuning baseline comparison.
- Image copying uses the upstream rule: the generated image must be nearer
  to a tuned example than to the untouched reference pool, and within the
  calibrated 1% held-out-neighbor distance.
- Each checkpoint generates 256 samples per seed. Watermark recovery uses
  20 trials of 4,096 raw velocity queries, with the original saved key.
- The previously saved watermarked checkpoint is also measured at step zero.
  Its original training examples are not the copying targets in this study.

## Results After 1,000 Steps

Mean and population standard deviation across three seeds:

| Fine-tuning images | Arm | FTSS / pretrained baseline | Copies | WM recovery |
|---|---|---|---|---|
| 50 | Clean continuation | 1.027 +/- 0.061 | 0/768 | 0% |
| 50 | Watermark embedding | 1.101 +/- 0.092 | 0/768 | 100% |
| 1,000 | Clean continuation | 1.039 +/- 0.012 | 0/768 | 0% |
| 1,000 | Watermark embedding | 1.034 +/- 0.003 | 0/768 | 100% |

Watermark recovery reaches 100% in all seeds by step 300. The FTSS ratios
remain close to or above their pre-fine-tuning baselines, and the image-copying
rule detects no replicas. The 50-image runs nevertheless develop a larger
training/reference flow-loss gap: after watermark embedding, their mean
losses are 0.326 and 0.515, versus 0.413 and 0.430 for the 1,000-image runs.
Thus ordinary fine-tuning overfitting is visible, while the tested checkpoints
show no newly replicated fine-tuning images or marked spectral collapse.
The upstream evaluator provides no universal binary FTSS cutoff; none is
introduced here.

## Artifacts

- `results.json`: all checkpoint measurements and FTSS curves.
- `summary.json`, `summary.csv`: seed-aggregated measurements.
- `comparison.png`, `comparison.pdf`: FTSS, copying, and recovery trajectories.
- `config.json`: source commit/hash, checkpoint hashes, and full parameters.
- `indices_*.json`: exact data partitions.
- `*_final.pt`: new fine-tuned model states.

Commands from the watermarking repository:

```bash
python run_ftss_finetuning.py --output-dir table_runs/ftss_finetuning_study \
  --runs 3 --steps 0 100 300 1000
python summarize_ftss_finetuning.py table_runs/ftss_finetuning_study
```

The runner deliberately rejects an existing study directory; use a fresh
output directory to repeat the experiment.
