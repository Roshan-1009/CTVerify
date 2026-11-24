# U-Net adversarial attack comparison

- Test studies: 10
- Positive studies per edit family: 2
- Maximum evaluated images per edit family: all selected positives
- L-infinity budget in normalized image units: 0.02
- Image score: max pre-sigmoid logit; clean validation calibration quantile: 99.0
- Clean test image FPR: 0.0210; clean test study FPR: 0.5000
- Methods evaluated: baseline, pgd
- FGSM/BIM/PGD budget: L-infinity epsilon 0.02; iterative steps / step size 10 / 0.004
- Attacks target suppression of the U-Net tamper response; attack success is reported only among synthetic edited images detected by the baseline model.
- The L-infinity and L2 attacks have different constraints and should not be interpreted as equal-budget comparisons.
- These attacks test the U-Net only. Any changed DICOM PixelData would still differ from an exact clean reference hash.

## Results
| family               | method   |   positive_images |   image_recall |   baseline_detected_images |   detected_images |   pixel_precision |   pixel_recall |   pixel_iou |   attack_success_rate |   clean_test_file_fpr |   clean_test_study_fpr |
|:---------------------|:---------|------------------:|---------------:|---------------------------:|------------------:|------------------:|---------------:|------------:|----------------------:|----------------------:|-----------------------:|
| localized_patch      | baseline |               236 |       0.995763 |                        235 |               235 |         0.169839  |       0.995688 |   0.169714  |            0          |             0.0210136 |                    0.5 |
| localized_patch      | pgd      |               236 |       0.995763 |                        235 |               235 |         0.0120999 |       0.990995 |   0.0120985 |            0.00425532 |             0.0210136 |                    0.5 |
| sparse_low_amplitude | baseline |               152 |       1        |                        152 |               152 |         0.0695766 |       0.986956 |   0.0695126 |            0          |             0.0210136 |                    0.5 |
| sparse_low_amplitude | pgd      |               152 |       1        |                        152 |               152 |         0.0333999 |       0.969003 |   0.0333643 |            0          |             0.0210136 |                    0.5 |
| copy_move_unseen     | baseline |               139 |       0.863309 |                        120 |               120 |         0.316143  |       0.757632 |   0.287107  |            0          |             0.0210136 |                    0.5 |
| copy_move_unseen     | pgd      |               139 |       0.992806 |                        120 |               138 |         0.0242293 |       0.687184 |   0.023965  |            0          |             0.0210136 |                    0.5 |

All adversarial samples are in-memory normalized arrays. This is a synthetic robustness test, not a real-world attack demonstration or clinical safety assessment.
