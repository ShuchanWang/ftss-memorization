# Fine-Tuning Study Protocols

These follow-up studies migrated from the watermark repository. Run commands
from the FTSS checkout through `run_study.py`; see its README for dependencies.
The descriptions below record the experimental protocols. Full FTSS artifacts
now live in this repository's `table_runs/`; compact evidence is in
`results/flow_finetuning/`. Historical manifests retain their original hashes.

## Clean FTSS Outcome Study

`run_ftss_outcome_study.py` extends the matched FTSS MLP experiment using
its two saved clean 12,000-image base checkpoints. It fine-tunes on nested,
class-balanced sets of 50, 200, and 1,000 new images, with three subset
seeds per base, and evaluates EMA weights at 0, 1,000, and 3,000 steps.
No watermark is trained. The original FTSS implementations in this repository supply the estimator.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u run_study.py run_ftss_outcome_study.py
```

The original FTSS probe definition is evaluated with the validated batched
backend. Paired generation uses 1,000 samples per checkpoint. Outcomes include
equal-pool training-example preference, training/held-out flow-loss gap,
loss-based membership AUC, feature precision and coverage, and style diversity.
Results, per-sample audits, descriptive associations, and figures are saved in
`table_runs/ftss_outcome_study/`. Re-running the identical command resumes
completed measurements and saved optimizer states; changed protocols require a
new output directory. Shared base models and repeated checkpoints are recorded
as dependent observations, not independent model replications. This study does
not modify the watermark paper.

### Pretrained CIFAR-10 Extension

`python -u run_study.py run_ftss_cifar_study.py` evaluates the public clean CIFAR-10
class-conditional RF-UNet during full-parameter fine-tuning. Defaults are
50/200/1,000-image nested test-split subsets, three subset seeds, and
0/300/1,000/3,000-step EMA evaluations. FTSS uses 20 balanced trajectories
and keeps the requested class fixed across each trajectory and its
perturbations. Generation uses 500 balanced class conditions and 100 Euler
steps. Quality is measured with Inception FID and PRDC, with disjoint
real-versus-real controls at matched sample counts. Training-example
preference is class-matched in feature and pixel spaces; flow-loss
membership and within-class diversity provide separate outcomes.

Results, optimizer snapshots, per-sample audits, and sample grids are in
`table_runs/ftss_cifar_study/`. The identical command resumes saved work.
The final report and association figures are generated automatically.
This uses one pretrained base, not independently trained model seeds, and
does not change either paper or the original checkpoint. Subset seeds
repeat fine-tuning, not base pretraining.


### Held-Out FTSS Diagnostic Validation

`python -u run_study.py run_ftss_diagnostic_validation.py` calibrates an FTSS ratio
threshold on the earlier UNet endpoint runs, then freezes it before
measuring new tuning seeds. Both clean MNIST MLP and UNet models are
tested. Each condition uses 400 distinct images: uniform sampling,
80% sampling mass on ten selected examples without changing the data,
and diverse digit-1 specialization. All 400 members are audited against
400 untouched same-domain examples; 200 more examples supply quality
references. A fixed pixel-RMSE near-copy rule is applied symmetrically
to members and nonmembers. The primary outcome is at least one percentage
point of training-minus-reference near-copy excess, not the condition's
name. All checkpoints generate 1,000 samples.

Primary sensitivity and false-alarm counts use the final 3,000-step
checkpoints; intermediate evaluations are secondary. Calibration FTSS is
remeasured with the same original-paper estimator used in validation.
Previously inspected calibration data are explicitly distinguished from
the new seeds, and the MLP results test transfer of the UNet-calibrated
rule. Checkpoints share one pretrained base per architecture, so seed
repeats are not independent base-model replications. Reports, the frozen
rule, source hashes, near-neighbor audits, numerical-clamp counters and
figures are saved in `table_runs/ftss_diagnostic_validation/`. The identical
command resumes work. This runner does not modify either paper.


# Clean SD3.5 Fine-Tuning Sensitivity

`run_ftss_sd35_finetuning.py` tests whether ordinary, unwatermarked LoRA
fine-tuning changes SD3.5's remaining-flow sensitivity. Defaults use 10 and
200 real CIFAR-10 cat images, three seeds, and checkpoints at 0, 100, 300,
and 1,000 optimizer steps. The cached clean backbone is used without any
watermark adapter. Training uses AdamW at `5e-5`, rank-16 attention and
feed-forward LoRA, batch 8, and the standard noise-minus-data target.

This is a 256px, cat-conditioned, CFG=1 pilot, not a native-resolution or
full-parameter experiment. Euler integration runs from noise sigma 1 to
data sigma 0, using FP32 latent states and BF16 network evaluation. Paired
probes compare RMS finite-difference gains at relative perturbations 0.02
and 0.1. Reports include complete temporal curves and train/reference
flow losses; latent nearest-neighbor distances are not labeled as copies.

```bash
python -u run_ftss_sd35_finetuning.py
python run_study.py test test_ftss_sd35_finetuning
```

Results, checkpoints, the report, and the final curve figure are written to
`table_runs/ftss_sd35_finetuning/`. Existing output directories are preserved;
use `--output` for an independent run.
