# Held-Out FTSS Diagnostic Validation

State: complete

Primary outcome: generated images within fixed visible-pixel RMSE <= 0.05 of a prespecified audited training image and closer to it than to an equal-size, class-matched reference pool. The same proximity rule is applied symmetrically to untouched reference examples. A checkpoint is outcome-positive if training-minus-reference replication exceeds or equals 1 percentage point(s). These fixed operational tolerances define this study, not a universal definition of memorization.

The FTSS warning threshold is fitted only on old UNet endpoint runs and frozen before new validation measurements. Calibration FTSS is remeasured using the same original-paper batched estimator as validation (15 trajectories, 20 times through 0.95, 200 Euler steps). Old generation audits are reused for calibration labels; their sample and anchor counts differ from validation and this is a distribution-shift test. Calibration data were previously viewed and are not described as blinded. New seeds are held out; MLP is an architecture-transfer test.

All three arms have 400 distinct training images and the same update schedule. Uniform and repeat arms share the exact data; the repeat arm assigns 80% of draws to ten prespecified images. Specialization uses 400 distinct digit-1 images. Each arm audits 400 members, 400 untouched same-domain reference images, and 200 quality-reference images. Specialization is not automatically labeled negative: its actual replication outcome determines the label. There is no watermark.

Frozen FTSS ratio threshold: 0.482010.

| Model | Arm | Checkpoints | TP | FN | FP | TN | Sensitivity | False alarm rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mlp | all | 9 | 0 | 0 | 0 | 9 | not estimable | 0.000 |
| mlp | uniform | 3 | 0 | 0 | 0 | 3 | not estimable | 0.000 |
| mlp | repeat | 3 | 0 | 0 | 0 | 3 | not estimable | 0.000 |
| mlp | specialize | 3 | 0 | 0 | 0 | 3 | not estimable | 0.000 |
| unet | all | 9 | 0 | 4 | 0 | 5 | 0.000 | 0.000 |
| unet | uniform | 3 | 0 | 0 | 0 | 3 | not estimable | 0.000 |
| unet | repeat | 3 | 0 | 3 | 0 | 0 | 0.000 | not estimable |
| unet | specialize | 3 | 0 | 1 | 0 | 2 | 0.000 | 0.000 |

Primary confusion counts use only the final 3000-step checkpoint. Earlier checkpoints are secondary monitoring measurements. All draws use the same fixed 1000-sample generation budget per checkpoint. Repeated checkpoints share pretrained bases and tuning realizations. Report counts by seed/arm, not independence-based significance tests. Absence of positive validation cases makes sensitivity unidentifiable, not perfect. The pixel reproduction definition is aligned-MNIST specific and does not establish natural-image perceptual copying.

| Model | Seed | Arm | Step | FTSS ratio | Training copies | Reference copies | Copy excess | Flagged |
|---|---:|---|---:|---:|---:|---:|---:|---|
| unet | 23101 | uniform | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23101 | uniform | 1000 | 0.9534 | 0.0010 | 0.0000 | 0.0010 | False |
| unet | 23101 | uniform | 3000 | 0.9032 | 0.0010 | 0.0000 | 0.0010 | False |
| unet | 23101 | repeat | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23101 | repeat | 1000 | 0.7553 | 0.3080 | 0.0000 | 0.3080 | False |
| unet | 23101 | repeat | 3000 | 0.5688 | 0.5670 | 0.0000 | 0.5670 | False |
| unet | 23101 | specialize | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23101 | specialize | 1000 | 0.7364 | 0.0080 | 0.0020 | 0.0060 | False |
| unet | 23101 | specialize | 3000 | 0.7097 | 0.0130 | 0.0070 | 0.0060 | False |
| unet | 23102 | uniform | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23102 | uniform | 1000 | 0.9490 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23102 | uniform | 3000 | 0.8836 | 0.0010 | 0.0010 | 0.0000 | False |
| unet | 23102 | repeat | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23102 | repeat | 1000 | 0.7752 | 0.2900 | 0.0000 | 0.2900 | False |
| unet | 23102 | repeat | 3000 | 0.5157 | 0.6210 | 0.0000 | 0.6210 | False |
| unet | 23102 | specialize | 0 | 1.0000 | 0.0010 | 0.0030 | -0.0020 | False |
| unet | 23102 | specialize | 1000 | 0.7501 | 0.0090 | 0.0080 | 0.0010 | False |
| unet | 23102 | specialize | 3000 | 0.7151 | 0.0290 | 0.0130 | 0.0160 | False |
| unet | 23103 | uniform | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23103 | uniform | 1000 | 0.9737 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23103 | uniform | 3000 | 0.8884 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23103 | repeat | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23103 | repeat | 1000 | 0.7917 | 0.2810 | 0.0000 | 0.2810 | False |
| unet | 23103 | repeat | 3000 | 0.6998 | 0.6060 | 0.0000 | 0.6060 | False |
| unet | 23103 | specialize | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| unet | 23103 | specialize | 1000 | 0.7068 | 0.0090 | 0.0060 | 0.0030 | False |
| unet | 23103 | specialize | 3000 | 0.6606 | 0.0130 | 0.0050 | 0.0080 | False |
| mlp | 23101 | uniform | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23101 | uniform | 1000 | 1.0982 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23101 | uniform | 3000 | 1.1359 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23101 | repeat | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23101 | repeat | 1000 | 0.9739 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23101 | repeat | 3000 | 0.9577 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23101 | specialize | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23101 | specialize | 1000 | 0.7734 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23101 | specialize | 3000 | 0.7809 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | uniform | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | uniform | 1000 | 1.3222 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | uniform | 3000 | 1.1339 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | repeat | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | repeat | 1000 | 1.2486 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | repeat | 3000 | 1.5869 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | specialize | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | specialize | 1000 | 1.1028 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23102 | specialize | 3000 | 1.1983 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | uniform | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | uniform | 1000 | 1.1017 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | uniform | 3000 | 1.0478 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | repeat | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | repeat | 1000 | 0.9317 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | repeat | 3000 | 0.8196 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | specialize | 0 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | specialize | 1000 | 1.1686 | 0.0000 | 0.0000 | 0.0000 | False |
| mlp | 23103 | specialize | 3000 | 0.7249 | 0.0000 | 0.0000 | 0.0000 | False |
