# Supervised DICOM pixel-tamper localization

A compact U-Net is trained to segment synthetic pixel edits. The model receives each image independently; it does not receive a clean counterpart or a hash reference at inference time.
Training edit families are localized patches and sparse low-amplitude changes. The copy-move family is withheld from training and evaluated as a synthetic out-of-family condition.
Thresholds are calibrated using clean validation images only. Pixel masks are resized to the model grid before evaluation; they are not native-resolution clinical annotations.

- Train images: 512 (cap=512)
- Validation images: 980
- Test images: 809 across 10 studies
- Device: cuda; epochs completed: 30; input: 256x256
- Best epoch selected by validation macro pixel IoU: 25 (0.9276)
- Synthetic-mask validation images: 256; selection threshold: 0.5
- Clean-test image FPR at calibrated threshold: 0.0482
- Clean-test pixel alarm fraction: 0.003694

## Image-level detection and pixel localization
|   accuracy |   precision |   recall |       f1 |   specificity |   false_positive_rate |   tn |   fp |   fn |   tp | condition            |   n_images |   n_positive_images |   image_auroc |   image_auprc |   clean_test_image_fpr |   clean_test_pixel_alarm_fraction |   pixel_precision |   pixel_recall |   pixel_f1 |   pixel_iou |   pixel_tp |   pixel_fp |   pixel_fn |   pixel_tn |
|-----------:|------------:|---------:|---------:|--------------:|----------------------:|-----:|-----:|-----:|-----:|:---------------------|-----------:|--------------------:|--------------:|--------------:|-----------------------:|----------------------------------:|------------------:|---------------:|-----------:|------------:|-----------:|-----------:|-----------:|-----------:|
|   0.966625 |    0.900383 | 0.995763 | 0.945674 |      0.954625 |             0.0453752 |  547 |   26 |    1 |  235 | localized_patch      |        809 |                 236 |      0.988005 |      0.949419 |              0.0482077 |                        0.00369423 |          0.612794 |       0.993572 |   0.758053 |    0.610374 |      75579 |      47756 |        489 |   15342672 |
|   0.964153 |    0.839779 | 1        | 0.912913 |      0.95586  |             0.04414   |  628 |   29 |    0 |  152 | sparse_low_amplitude |        809 |                 152 |      0.990868 |      0.926829 |              0.0482077 |                        0.00369423 |          0.41263  |       0.950412 |   0.575431 |    0.403933 |      94068 |     133904 |       4908 |    9728592 |
|   0.945612 |    0.810458 | 0.892086 | 0.849315 |      0.956716 |             0.0432836 |  641 |   29 |   15 |  124 | copy_move_unseen     |        809 |                 139 |      0.964313 |      0.874798 |              0.0482077 |                        0.00369423 |          0.679331 |       0.718991 |   0.698598 |    0.536805 |      66910 |      31584 |      26151 |    8984859 |

## Best-checkpoint validation-mask metrics
| attack_family        |   n_images |   pixel_dice |   pixel_iou |   pixel_precision |   pixel_recall |   pixel_tp |   pixel_fp |   pixel_fn |
|:---------------------|-----------:|-------------:|------------:|------------------:|---------------:|-----------:|-----------:|-----------:|
| localized_patch      |        128 |     0.979815 |    0.960428 |          0.992013 |       0.967912 |      45337 |        365 |       1503 |
| sparse_low_amplitude |        128 |     0.944451 |    0.894748 |          0.993561 |       0.899966 |      42896 |        278 |       4768 |

## Study-level detection
| condition            |   n_studies |   accuracy |   precision |   recall |       f1 |   specificity |   false_positive_rate |   tn |   fp |   fn |   tp |
|:---------------------|------------:|-----------:|------------:|---------:|---------:|--------------:|----------------------:|-----:|-----:|-----:|-----:|
| localized_patch      |          10 |        0.4 |    0.25     |        1 | 0.4      |         0.25  |                 0.75  |    2 |    6 |    0 |    2 |
| sparse_low_amplitude |          10 |        0.5 |    0.285714 |        1 | 0.444444 |         0.375 |                 0.625 |    3 |    5 |    0 |    2 |
| copy_move_unseen     |          10 |        0.5 |    0.285714 |        1 | 0.444444 |         0.375 |                 0.625 |    3 |    5 |    0 |    2 |

## Interpretation limits
- A withheld copy-move condition tests transfer to one unseen synthetic family, not generalization to natural attacks.
- Scores and masks are evaluated on the held-out CPAISD test fold; the ten-study sample limits population-level inference.
- Localization metrics use generated masks and the resized image grid.
- This experiment does not establish clinical impact, real-world attack prevalence, or deployment readiness.