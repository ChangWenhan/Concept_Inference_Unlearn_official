# Per-class MIA profile — cifar10 (target class 4)

Protocol: for every class c, members = class-c training images, non-members = class-c test images.
`simple_gap` = |accuracy-0.5| of the loss-based LR attack (10-fold); `fr` = paper SVM-transfer
forgetting rate (attack fitted on the ORIGINAL model, applied to each model).

## Summary (simple_gap)

| model | target value | rank of target | controls mean | controls min | controls max |
|---|---|---|---|---|---|
| original | 0.044 | 5/10 | 0.045 | 0.024 | 0.086 |
| unlearned | 0.016 | 1/10 | 0.041 | 0.018 | 0.098 |
| retrain | 0.016 | 4/10 | 0.021 | 0.005 | 0.038 |

## Summary (fr)

| model | target value | rank of target | controls mean | controls min | controls max |
|---|---|---|---|---|---|
| original | 0.009 | 10/10 | 0.004 | 0.001 | 0.009 |
| unlearned | 1.000 | 10/10 | 0.007 | 0.000 | 0.031 |
| retrain | 1.000 | 10/10 | 0.242 | 0.084 | 0.630 |

## Full table (simple_gap / fr)

| class | target? | original gap | unlearned gap | retrain gap | original Fr | unlearned Fr | retrain Fr |
|---|---|---|---|---|---|---|---|
| 0 |  | 0.055 | 0.058 | 0.032 | 0.007 | 0.031 | 0.212 |
| 1 |  | 0.028 | 0.018 | 0.015 | 0.002 | 0.000 | 0.202 |
| 2 |  | 0.048 | 0.043 | 0.022 | 0.002 | 0.001 | 0.200 |
| 3 |  | 0.086 | 0.098 | 0.038 | 0.001 | 0.008 | 0.630 |
| 4 | **yes** | 0.044 | 0.016 | 0.016 | 0.009 | 1.000 | 1.000 |
| 5 |  | 0.052 | 0.036 | 0.020 | 0.005 | 0.004 | 0.221 |
| 6 |  | 0.024 | 0.020 | 0.017 | 0.009 | 0.009 | 0.120 |
| 7 |  | 0.037 | 0.022 | 0.014 | 0.003 | 0.000 | 0.084 |
| 8 |  | 0.027 | 0.019 | 0.005 | 0.007 | 0.002 | 0.238 |
| 9 |  | 0.045 | 0.051 | 0.027 | 0.005 | 0.006 | 0.268 |
