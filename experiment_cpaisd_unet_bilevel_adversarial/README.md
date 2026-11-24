# CPAISD U-Net: bilevel adversarial training vs. the retrained 30-epoch baseline

This folder holds the scripts and results for comparing two tamper-localization U-Nets on CPAISD:

- **Retrained 30-epoch baseline:** width 8, 512 training slices, batch 8, fixed loss, no adversarial training.
- **Upgraded bilevel adversarial U-Net:** width 24, all 8,376 training slices, batch 18, online 7-step PGD adversarial training (L∞ 0.02, step 0.004), positive-pixel BCE weight 3, and a WBCE/Dice weight learned by hypergradient.

The scripts are wrappers around `AdityaGupta-debug/dip-coma-cn` at commit `f895971`, which they leave unmodified. Running them needs a checkout of that repository next to this folder (`work/dip-coma-cn`), as `scripts/run_bilevel_cached.py` shows. The absolute data paths in the scripts belong to the machine the runs were made on; change them before re-running.

## Setup

| | |
|---|---|
| Train | 8,376 slices (official CPAISD train fold) |
| Validation | 980 slices, 10 studies (checkpoint selection, all threshold calibration) |
| Test | 809 slices, 10 studies (official test fold) |
| Edited test images | 527: localized patch 236, sparse low-amplitude 152, copy-move (never seen in training) 139 |
| Attacks | FGSM; 10-step PGD with the evaluator's mean-BCE objective; 10-step PGD on the max logit (`scripts/run_max_logit_pgd.py`). All at L∞ ≤ 0.02 in normalized units, seed 42 |
| Image score | Max pre-sigmoid logit over the slice |
| Hardware | RTX 4070 Laptop GPU (8 GB); peak 5.82 GiB allocated / 7.16 GiB reserved |

**Batch size change.** A batch of 16 was requested; 18 was used, after a pilot run showed it fit in 8 GB.

**Training.** All 30 epochs completed in 23,169 s (about 6 h 26 min). Epoch 25 was selected, with validation macro IoU 0.8667 at a 0.5 pixel threshold. The learned weights at that epoch were BCE 0.197 / Dice 0.803. The BCE weight was still falling at epoch 30 (0.139), so the learned weight had not converged.

## 1. Evaluator protocol (thresholds from the checkpoint)

The image alarm threshold is the clean-validation q99 of the max logit. The pixel threshold is the clean-pixel q99.5 stored in the checkpoint.

| Family | Attack | Recall, baseline | Recall, upgraded | IoU, baseline | IoU, upgraded |
|---|---|---:|---:|---:|---:|
| Patch | Unattacked | 0.996 | 0.996 | 0.610 | 0.170 |
| Patch | FGSM | 0.992 | 1.000 | 0.022 | 0.011 |
| Patch | PGD mean BCE | 0.144 | 0.996 | 0.023 | 0.012 |
| Patch | PGD max logit | 0.470 | 1.000 | 0.023 | 0.011 |
| Sparse | Unattacked | 1.000 | 1.000 | 0.404 | 0.070 |
| Sparse | FGSM | 0.941 | 1.000 | 0.014 | 0.028 |
| Sparse | PGD mean BCE | 0.059 | 1.000 | 0.082 | 0.033 |
| Sparse | PGD max logit | 1.000 | 1.000 | 0.081 | 0.029 |
| Copy-move | Unattacked | 0.892 | 0.863 | 0.537 | 0.287 |
| Copy-move | FGSM | 1.000 | 1.000 | 0.020 | 0.024 |
| Copy-move | PGD mean BCE | 0.137 | 0.993 | 0.024 | 0.024 |
| Copy-move | PGD max logit | 0.309 | 1.000 | 0.025 | 0.024 |

Clean-slice FPR: 4.82% (39/809) for the baseline and 2.10% (17/809) for the upgraded model. Clean-study FPR: 7/10 and 5/10.

**This table overstates the upgraded model's robustness and understates its localization.**

- **Recall.** At this threshold (−11.77 logits), the upgraded model also alarms on 100% of clean slices with uniform or sign noise, and on 84–100% of clean slices after PGD (section 2). Its high recall under attack partly reflects any perturbation raising the score, not tamper detection.
- **IoU.** The upgraded checkpoint's pixel threshold is 7.1e-9. At that value almost every pixel counts as positive, which explains its low IoU.

## 2. Control check: false alarms on perturbed clean slices

These are test-fold clean slices with the same perturbation budget applied and no edit. The table shows the share that alarm at each model's clean-q99 threshold.

| Perturbation of clean slices | Baseline | Upgraded |
|---|---:|---:|
| None | 4.8% | 2.1% |
| Uniform noise | 63.7% | 100% |
| Sign noise | 88.0% | 100% |
| PGD mean BCE | 7.7% | 84.4% |
| PGD max logit | 18.5% | 99.9% |

## 3. Validation-calibrated results

**Pixel threshold.** Chosen on validation from a grid, using only the training families (patch, sparse) under unattacked and both PGD conditions.

| | Selected pixel threshold | Validation macro IoU |
|---|---:|---:|
| Baseline | 0.9 (top of the grid) | 0.425 |
| Upgraded | 0.01 | 0.904 |

The upgraded model's validation IoU is flat from 1e-3 to 0.9 (0.87–0.90) and collapses below 1e-4. Its probabilities are saturated near 0 and 1.

**Image alarm threshold (perturbation-aware).** This is the highest q99 across clean-validation conditions: none, uniform noise, sign noise and both PGD attacks. Both thresholds were set by sign noise: 19.83 for the baseline and 26.91 for the upgraded model.

Test pixel IoU at each model's selected threshold:

| Family | Unattacked, baseline | Unattacked, upgraded | PGD mean BCE, baseline | PGD mean BCE, upgraded | PGD max logit, baseline | PGD max logit, upgraded |
|---|---:|---:|---:|---:|---:|---:|
| Patch | 0.971 | 0.985 | 0.019 | 0.971 | 0.060 | 0.935 |
| Sparse | 0.867 | 0.940 | 0.004 | 0.695 | 0.661 | 0.897 |
| Copy-move | 0.571 | 0.723 | 0.016 | 0.580 | 0.043 | 0.620 |

Test image recall at the perturbation-aware alarm threshold:

| Family | Unattacked, baseline | Unattacked, upgraded | PGD mean BCE, baseline | PGD mean BCE, upgraded | PGD max logit, baseline | PGD max logit, upgraded |
|---|---:|---:|---:|---:|---:|---:|
| Patch | 99.2% | 99.2% | 0.8% | 96.2% | 0% | 96.6% |
| Sparse | 100% | 100% | 0% | 99.3% | 100% | 100% |
| Copy-move | 78.4% | 79.1% | 7.2% | 54.7% | 7.9% | 54.0% |

Test false alarms on clean slices at the perturbation-aware threshold, baseline vs. upgraded:

- No perturbation: 1.2% vs. 0%.
- Uniform noise: 1.2% vs. 0.6%.
- Sign noise: 2.2% vs. 1.4%.
- PGD: 0–0.1% vs. 0%.

Clean studies alarmed for the upgraded model: 0/10 with no perturbation, 3/10 with noise.

FGSM is not part of the calibrated tables. It appears only in section 1.

## Conclusion

With thresholds calibrated on validation, robustness improved substantially:

- **Patch and sparse edits:** the upgraded model keeps about 95% of its clean IoU under max-logit PGD. Under mean-BCE PGD it keeps 99% on patch but only 74% on sparse. The baseline drops to almost nothing, except for sparse edits under max-logit PGD.
- **Copy-move:** both models are weaker, the upgraded one less so.

There was no measurable cost on clean images. Clean IoU, clean recall and the clean false-alarm rate are all equal or better for the upgraded model.

The apparent clean cost in section 1 comes from the 7.1e-9 pixel threshold, not from the model. The plain clean-q99 alarm rule is not safe for the upgraded model, because any small noise triggers it.

**Caveat: the two models differ in several ways at once.** They differ in width (8 vs. 24), training data (512 vs. 8,376 slices), batch size, adversarial training and loss weighting. These results therefore do not credit the gains to adversarial training or to bilevel weighting specifically.

## Limitations

- Attacks are white-box 10-step PGD with one random start. Stronger attacks (more steps, restarts, or attacks against the perturbation-aware threshold) have not been tested.
- Random noise is only a proxy for harmless real-world changes.
- There are 2 positive test studies per edit family, and all edits are synthetic.
- Edits and attacks change normalized, resized pixels. No claim is made about edits to native DICOM stored pixel values.

## Contents

- **`scripts/`**
  - `run_bilevel_cached.py` and `train_offline.cmd`: training launcher with an in-RAM slice cache.
  - `run_unet_evaluation_cuda_safe.py`: fixes the evaluator's CUDA/CPU device mismatch without editing the repository.
  - `run_max_logit_pgd.py`: max-logit PGD attack.
  - `run_after_training.py`: follow-up attack and comparison table.
  - `run_control_checks.py`: perturbed-clean control check.
  - `run_validation_calibration.py`: validation threshold calibration.
- **`results/`**
  - `retrained_baseline_30ep/`: baseline training outputs.
  - `bilevel_adversarial_w24_e30_b18/`: upgraded training and evaluator outputs.
  - `max_logit_pgd/`: max-logit attack results.
  - `control_checks/`: perturbed-clean control check results.
  - `validation_calibration/`: calibration results.
  - `final_comparison/`: the section 1 table.
  - Training and follow-up logs.
- **Left out:**
  - Model checkpoints (`*.pt`, 2–17 MB).
  - Two per-pixel intermediate CSVs from validation calibration (19 MB). `run_validation_calibration.py` regenerates them.
