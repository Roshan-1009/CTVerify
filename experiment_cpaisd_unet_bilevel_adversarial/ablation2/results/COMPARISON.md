# CPAISD controlled baseline (Ablation 2)

The old baseline used width 8 and 512 training slices. The fair baseline and upgraded model used width 24 and all 8,376 slices. The fair baseline uses fixed equal BCE/Dice weights and ordinary synthetic edits. Differences measure the combined training change. The fair checkpoint was selected on unattacked validation IoU and the upgraded checkpoint on unattacked and PGD validation IoU. Each model's final thresholds were fitted on validation.

## PGD recall on edited test slices (%)

| condition                                   |   Old baseline |   Fair baseline |   Upgraded |   Upgraded minus fair |
|:--------------------------------------------|---------------:|----------------:|-----------:|----------------------:|
| edited/copy_move_unseen/pgd10_max_logit     |        7.91367 |         0.71942 |   53.95683 |              53.23741 |
| edited/copy_move_unseen/pgd10_mean_bce      |        7.19424 |         0.71942 |   54.67626 |              53.95683 |
| edited/localized_patch/pgd10_max_logit      |        0.00000 |         0.84746 |   96.61017 |              95.76271 |
| edited/localized_patch/pgd10_mean_bce       |        0.84746 |         0.42373 |   96.18644 |              95.76271 |
| edited/sparse_low_amplitude/pgd10_max_logit |      100.00000 |         0.00000 |  100.00000 |             100.00000 |
| edited/sparse_low_amplitude/pgd10_mean_bce  |        0.00000 |         2.63158 |   99.34211 |              96.71053 |

## All edited conditions: pixel IoU

| condition                            |   Old baseline |   Fair baseline |   Upgraded |   Upgraded minus fair |
|:-------------------------------------|---------------:|----------------:|-----------:|----------------------:|
| copy_move_unseen/pgd10_max_logit     |        0.04327 |         0.07681 |    0.62002 |               0.54320 |
| copy_move_unseen/pgd10_mean_bce      |        0.01600 |         0.02622 |    0.57980 |               0.55358 |
| copy_move_unseen/unattacked          |        0.57092 |         0.78145 |    0.72267 |              -0.05878 |
| localized_patch/pgd10_max_logit      |        0.06006 |         0.14868 |    0.93455 |               0.78587 |
| localized_patch/pgd10_mean_bce       |        0.01937 |         0.02851 |    0.97116 |               0.94265 |
| localized_patch/unattacked           |        0.97142 |         0.98753 |    0.98448 |              -0.00305 |
| sparse_low_amplitude/pgd10_max_logit |        0.66068 |         0.18331 |    0.89706 |               0.71374 |
| sparse_low_amplitude/pgd10_mean_bce  |        0.00356 |         0.03184 |    0.69487 |               0.66303 |
| sparse_low_amplitude/unattacked      |        0.86682 |         0.94203 |    0.94041 |              -0.00162 |

## All edited conditions: slice recall (%)

| condition                                   |   Old baseline |   Fair baseline |   Upgraded |   Upgraded minus fair |
|:--------------------------------------------|---------------:|----------------:|-----------:|----------------------:|
| edited/copy_move_unseen/pgd10_max_logit     |        7.91367 |         0.71942 |   53.95683 |              53.23741 |
| edited/copy_move_unseen/pgd10_mean_bce      |        7.19424 |         0.71942 |   54.67626 |              53.95683 |
| edited/copy_move_unseen/unattacked          |       78.41727 |        77.69784 |   79.13669 |               1.43885 |
| edited/localized_patch/pgd10_max_logit      |        0.00000 |         0.84746 |   96.61017 |              95.76271 |
| edited/localized_patch/pgd10_mean_bce       |        0.84746 |         0.42373 |   96.18644 |              95.76271 |
| edited/localized_patch/unattacked           |       99.15254 |        98.72881 |   99.15254 |               0.42373 |
| edited/sparse_low_amplitude/pgd10_max_logit |      100.00000 |         0.00000 |  100.00000 |             100.00000 |
| edited/sparse_low_amplitude/pgd10_mean_bce  |        0.00000 |         2.63158 |   99.34211 |              96.71053 |
| edited/sparse_low_amplitude/unattacked      |      100.00000 |       100.00000 |  100.00000 |               0.00000 |

## Clean-slice false alarms (%)

| condition             |   Old baseline |   Fair baseline |   Upgraded |   Upgraded minus fair |
|:----------------------|---------------:|----------------:|-----------:|----------------------:|
| clean/none            |        1.23609 |         0.00000 |    0.00000 |               0.00000 |
| clean/pgd10_max_logit |        0.12361 |         0.00000 |    0.00000 |               0.00000 |
| clean/pgd10_mean_bce  |        0.00000 |         0.61805 |    0.00000 |              -0.61805 |
| clean/sign_noise      |        2.22497 |         0.00000 |    1.35970 |               1.35970 |
| clean/uniform_noise   |        1.23609 |         0.00000 |    0.61805 |               0.61805 |

## Clean-study false alarms (%)

| condition             |   Old baseline |   Fair baseline |   Upgraded |   Upgraded minus fair |
|:----------------------|---------------:|----------------:|-----------:|----------------------:|
| clean/none            |       40.00000 |         0.00000 |    0.00000 |               0.00000 |
| clean/pgd10_max_logit |       10.00000 |         0.00000 |    0.00000 |               0.00000 |
| clean/pgd10_mean_bce  |        0.00000 |        40.00000 |    0.00000 |             -40.00000 |
| clean/sign_noise      |       70.00000 |         0.00000 |   30.00000 |              30.00000 |
| clean/uniform_noise   |       40.00000 |         0.00000 |   30.00000 |              30.00000 |

Rate differences are percentage points. Each clean condition includes 809 slices in ten studies. Patch, sparse and held-out copy-move have 236, 152 and 139 edited test slices, respectively.
