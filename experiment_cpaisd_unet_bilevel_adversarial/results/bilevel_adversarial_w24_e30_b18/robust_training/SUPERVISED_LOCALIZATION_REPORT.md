# Supervised DICOM pixel-tamper localization

The unet model is trained to segment synthetic pixel edits. The model receives each image independently; it does not receive a clean counterpart or a hash reference at inference time.
The compound segmentation loss is 0.1971 WBCE + 0.8029 Dice at the selected checkpoint (positive-pixel BCE weight 3). Weighting mode: bilevel.
Training edit families are localized patches and sparse low-amplitude changes. The copy-move family is withheld from training and evaluated as a synthetic out-of-family condition.
The image-level alarm score is the maximum pre-sigmoid logit, avoiding probability saturation. Thresholds are calibrated using clean validation images only. For study-disjoint adaptive weighting, validation studies are split into coefficient-update and checkpoint-selection/calibration groups; the test fold is untouched. Bilevel mode uses a one-inner-step differentiable hypergradient approximation. Adversarial mode balances clean, standard synthetic, and live PGD-evaded samples in each update; PGD inputs are generated with BatchNorm in evaluation mode and are trained against their original tamper masks. The validation-selected checkpoint maximizes mean pixel IoU across unattacked and 10-step PGD validation edits, with clean q99 false-alarm rate as the tie-break. Pixel masks are resized to the model grid before evaluation; they are not native-resolution clinical annotations.

- Train images: 8376 (cap=all)
- Validation images: 980
- Test images: 809 across 10 studies
- Device: cuda; epochs completed: 30; input: 256x256
- Architecture: unet; base channels: 24
- Loss-weighting mode: bilevel; selected WBCE/Dice coefficients: 0.1971/0.8029
- Validation studies: 10 total; 5 coefficient-update; 5 checkpoint-selection/calibration
- Best epoch selected by validation macro pixel IoU: 25 (0.8667)
- Synthetic-mask validation images: 256; selection threshold: 0.5
- Clean-test image FPR at calibrated threshold: 0.0235
- Clean-test pixel alarm fraction: 0.002288

## Image-level detection and pixel localization
|   accuracy |   precision |   recall |       f1 |   specificity |   false_positive_rate |   tn |   fp |   fn |   tp | condition            |   n_images |   n_positive_images | loss_weighting   |   bce_weight |   dice_weight |   image_auroc |   image_auprc |   clean_test_image_fpr |   clean_test_pixel_alarm_fraction |   pixel_precision |   pixel_recall |   pixel_f1 |   pixel_iou |   pixel_tp |   pixel_fp |   pixel_fn |   pixel_tn |
|-----------:|------------:|---------:|---------:|--------------:|----------------------:|-----:|-----:|-----:|-----:|:---------------------|-----------:|--------------------:|:-----------------|-------------:|--------------:|--------------:|--------------:|-----------------------:|----------------------------------:|------------------:|---------------:|-----------:|------------:|-----------:|-----------:|-----------:|-----------:|
|   0.978986 |    0.936255 | 0.995763 | 0.965092 |      0.972077 |             0.0279232 |  557 |   16 |    1 |  235 | localized_patch      |        809 |                 236 | bilevel          |     0.197149 |      0.802851 |      0.997168 |      0.997378 |              0.0234858 |                        0.00228801 |         0.169845  |       0.995688 |   0.290189 |   0.16972   |      75740 |     370197 |        328 |   15020231 |
|   0.976514 |    0.888889 | 1        | 0.941176 |      0.971081 |             0.0289193 |  638 |   19 |    0 |  152 | sparse_low_amplitude |        809 |                 152 | bilevel          |     0.197149 |      0.802851 |      1        |      1        |              0.0234858 |                        0.00228801 |         0.0695751 |       0.986956 |   0.129987 |   0.0695112 |      97685 |    1306338 |       1291 |    8556158 |
|   0.953028 |    0.863309 | 0.863309 | 0.863309 |      0.971642 |             0.0283582 |  651 |   19 |   19 |  120 | copy_move_unseen     |        809 |                 139 | bilevel          |     0.197149 |      0.802851 |      0.908977 |      0.899309 |              0.0234858 |                        0.00228801 |         0.316141  |       0.757632 |   0.446125 |   0.287105  |      70506 |     152515 |      22555 |    8863928 |

## Best-checkpoint validation-mask metrics
| attack_family        | condition   |   n_images |   pixel_iou |   pixel_dice |   pixel_precision |   pixel_recall |
|:---------------------|:------------|-----------:|------------:|-------------:|------------------:|---------------:|
| localized_patch      | unattacked  |        128 |    0.983551 |     0.991707 |          0.99289  |       0.990528 |
| localized_patch      | pgd10       |        128 |    0.857638 |     0.923364 |          0.990838 |       0.864494 |
| sparse_low_amplitude | unattacked  |        128 |    0.945249 |     0.971854 |          0.992348 |       0.952189 |
| sparse_low_amplitude | pgd10       |        128 |    0.680434 |     0.809831 |          0.992227 |       0.684081 |

## Study-level detection
| condition            |   n_studies |   accuracy |   precision |   recall |       f1 |   specificity |   false_positive_rate |   tn |   fp |   fn |   tp |
|:---------------------|------------:|-----------:|------------:|---------:|---------:|--------------:|----------------------:|-----:|-----:|-----:|-----:|
| localized_patch      |          10 |        0.7 |    0.4      |        1 | 0.571429 |         0.625 |                 0.375 |    5 |    3 |    0 |    2 |
| sparse_low_amplitude |          10 |        0.5 |    0.285714 |        1 | 0.444444 |         0.375 |                 0.625 |    3 |    5 |    0 |    2 |
| copy_move_unseen     |          10 |        0.5 |    0.285714 |        1 | 0.444444 |         0.375 |                 0.625 |    3 |    5 |    0 |    2 |

## Interpretation limits
- A withheld copy-move condition tests transfer to one unseen synthetic family, not generalization to natural attacks.
- Scores and masks are evaluated on the held-out CPAISD test fold; the ten-study sample limits population-level inference.
- Localization metrics use generated masks and the resized image grid.
- This experiment does not establish clinical impact, real-world attack prevalence, or deployment readiness.