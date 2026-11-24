# Ablation 2: full-width fixed-loss baseline

This control trains a width-24 U-Net with fixed 0.5 weighted BCE + 0.5 Dice (positive weight 3), AdamW (learning rate 0.001, weight decay 0.0001), and no scheduler. It uses seed 42, all 8,376 training slices, batch size 18, and 30 epochs. The training process never receives the test fold.

Each update uses six source slices. It includes one clean copy and two copies of the same ordinary edit per source. Replacing the adversarial view with its unperturbed source preserves the upgraded run's masks, 6:12 clean-to-edited balance, and 1,396 optimizer steps per epoch. The audited data generator supplies the same epoch-dependent patch and sparse edits. Copy-move is held out.

The fair checkpoint maximizes unattacked validation macro IoU at pixel probability 0.5 on the same 256-image checkpoint-selection subset. The upgraded model's selection also included PGD validation. The comparison measures these training procedures jointly. Both use the same final threshold-selection script.

To reproduce, use a fresh output directory and supply a checkout of AdityaGupta-debug/dip-coma-cn at f895971 as SOURCE. Run the following from the workspace with the paths replaced for your machine:

```text
python ABLATION2/train.py --source SOURCE --train TRAIN --validation VAL --output outputs/ablation_w24_clean
python WRAPPERS/run_validation_calibration.py --validation VAL --test TEST --model "Fair Baseline=outputs/ablation_w24_clean/unet_localizer.pt" --control-checks CONTROL_CHECKS --output outputs/ablation_w24_clean/validation_calibration
python ABLATION2/compare.py --old ORIGINAL_VALIDATION_CALIBRATION --fair outputs/ablation_w24_clean/validation_calibration --output outputs/ablation_w24_clean/COMPARISON.md
```

WRAPPERS uses the directory layout documented in the parent experiment. ORIGINAL_VALIDATION_CALIBRATION contains the frozen results for the old and upgraded models. The report uses each model's validation-selected pixel threshold and perturbation-aware q99 alarm threshold. It includes test slice counts of 809 clean, 236 patch, 152 sparse and 139 copy-move images. Clean study rates have a denominator of ten.
