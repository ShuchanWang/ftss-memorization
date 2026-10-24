# Clean Fine-Tuning: FTSS and Generative Outcomes

State: complete

2 saved matched-recipe clean base models; 3 subset seeds per base. Class-balanced, nested [50, 200, 1000]-image tuning subsets use MNIST test data excluded from base training. Equal-size untouched reference pools control nearest-neighbor set size. A fixed disjoint 1000-image bank supplies PRDC and diversity references. All outcomes use EMA weights, paired generation noise and FTSS probes within each run. Training uses the matched 3000-step OneCycle recipe.

FTSS retains the upstream estimator definition: 15 trajectories, 20 injection times, 200 Euler steps and relative perturbation 0.1. Generated outcomes use 1000 samples; flow losses average eight noise/time draws per real example. Precision and coverage assess generation quality; training preference and loss gap assess training-set attraction and overfitting; diversity assesses concentration. These are distinct outcomes, not interchangeable definitions of memorization.

| Images | Steps | Runs | FTSS / Before | Preference Change | Loss Gap Change | Coverage Change | Style Diversity Change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1000 | 6 | 0.9314 +/- 0.0065 | 0.0533 +/- 0.0236 | 0.1143 +/- 0.0052 | -0.0035 +/- 0.0019 | -0.0321 +/- 0.0105 |
| 50 | 3000 | 6 | 0.8890 +/- 0.0071 | 0.0977 +/- 0.0405 | 0.3129 +/- 0.0147 | -0.0020 +/- 0.0031 | -0.0414 +/- 0.0219 |
| 200 | 1000 | 6 | 0.9694 +/- 0.0024 | 0.0225 +/- 0.0055 | 0.0471 +/- 0.0018 | -0.0025 +/- 0.0019 | -0.0293 +/- 0.0041 |
| 200 | 3000 | 6 | 0.9499 +/- 0.0060 | 0.0523 +/- 0.0205 | 0.1303 +/- 0.0045 | -0.0018 +/- 0.0033 | -0.0255 +/- 0.0136 |
| 1000 | 1000 | 6 | 0.9908 +/- 0.0052 | 0.0060 +/- 0.0132 | 0.0099 +/- 0.0003 | -0.0038 +/- 0.0013 | -0.0120 +/- 0.0047 |
| 1000 | 3000 | 6 | 0.9878 +/- 0.0035 | 0.0067 +/- 0.0093 | 0.0237 +/- 0.0003 | -0.0002 +/- 0.0013 | -0.0109 +/- 0.0065 |

Means +/- population standard deviations across completed fine-tuning realizations.

## Associations

Descriptive associations; repeated checkpoints and shared bases are dependent. Subset seeds are repeated fine-tuning realizations, not independent pretrained models.

Contraction = -log(FTSS / baseline); positive values mean lower FTSS. Changes subtract the same run's step-zero outcome. Spearman associations are also stratified by checkpoint step and pretrained base in summary.json. No independence-based p-values are used.

- training_preference_change: Spearman 0.854255245903635
- training_anchor_advantage_change: Spearman 0.9413127413127413
- loss_gap_change: Spearman 0.905019305019305
- membership_auc_change: Spearman 0.9142857142857144
- coverage_change: Spearman -0.08945858476693329
- precision_change: Spearman -0.031018044220189136
- style_diversity_ratio_change: Spearman -0.5992277992277991
- effective_digit_modes_change: Spearman -0.7688545688545689
