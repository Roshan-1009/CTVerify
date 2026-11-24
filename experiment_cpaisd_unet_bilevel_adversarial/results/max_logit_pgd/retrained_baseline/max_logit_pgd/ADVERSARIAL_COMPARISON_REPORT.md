# U-Net adversarial attack comparison

- Test studies: 10
- Positive studies per edit family: 2
- Maximum evaluated images per edit family: all selected positives
- L-infinity budget in normalized image units: 0.02
- Image score: max pre-sigmoid logit; clean validation calibration quantile: 99.0
- Clean test image FPR: 0.0482; clean test study FPR: 0.7000
- Methods evaluated: baseline, pgd
- FGSM/BIM/PGD budget: L-infinity epsilon 0.02; iterative steps / step size 10 / 0.004
- Attacks target suppression of the U-Net tamper response; attack success is reported only among synthetic edited images detected by the baseline model.
- The L-infinity and L2 attacks have different constraints and should not be interpreted as equal-budget comparisons.
- These attacks test the U-Net only. Any changed DICOM PixelData would still differ from an exact clean reference hash.

## Results
| family               | method   |   positive_images |   image_recall |   baseline_detected_images |   detected_images |   pixel_precision |   pixel_recall |   pixel_iou |   attack_success_rate |   clean_test_file_fpr |   clean_test_study_fpr |
|:---------------------|:---------|------------------:|---------------:|---------------------------:|------------------:|------------------:|---------------:|------------:|----------------------:|----------------------:|-----------------------:|
| localized_patch      | baseline |               236 |       0.995763 |                        235 |               235 |         0.612789  |       0.993572 |   0.610369  |              0        |             0.0482077 |                    0.7 |
| localized_patch      | pgd      |               236 |       0.470339 |                        235 |               111 |         0.0238301 |       0.580362 |   0.0234265 |              0.52766  |             0.0482077 |                    0.7 |
| sparse_low_amplitude | baseline |               152 |       1        |                        152 |               152 |         0.412635  |       0.950412 |   0.403939  |              0        |             0.0482077 |                    0.7 |
| sparse_low_amplitude | pgd      |               152 |       1        |                        152 |               152 |         0.0813837 |       0.918202 |   0.0807979 |              0        |             0.0482077 |                    0.7 |
| copy_move_unseen     | baseline |               139 |       0.892086 |                        124 |               124 |         0.679351  |       0.718991 |   0.536817  |              0        |             0.0482077 |                    0.7 |
| copy_move_unseen     | pgd      |               139 |       0.309353 |                        124 |                43 |         0.0266982 |       0.302748 |   0.0251516 |              0.653226 |             0.0482077 |                    0.7 |

All adversarial samples are in-memory normalized arrays. This is a synthetic robustness test, not a real-world attack demonstration or clinical safety assessment.
