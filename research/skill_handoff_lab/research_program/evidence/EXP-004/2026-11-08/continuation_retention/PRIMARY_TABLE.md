| Learner | NN seed | Success | Rescue | Regression | Own episodes | Own decisions | Updates | Refresh episodes | New train planning calls incl. shared |
|---|---|---|---|---|---|---|---|---|---|
| Frozen first localZ-minus | frozen | 24/32 | 9 | 3 | 0 | 0 | 0 | 0 | 0 |
| Warm | frozen | 18/32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Frozen feedback rule | frozen | 18/32 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| SAC + success BC | 32301 | 17/32 | 2 | 3 | 128 | 751 | 5120 | 47 | 3155 |
| SAC + success BC | 32302 | 17/32 | 2 | 3 | 128 | 760 | 5120 | 47 | 3161 |
| Reference KL + paired ranking | 32301 | 18/32 | 1 | 1 | 128 | 640 | 5120 | 47 | 3154 |
| Reference KL + paired ranking | 32302 | 18/32 | 1 | 1 | 128 | 707 | 5120 | 47 | 3147 |
| Supervised + paired ranking | 32301 | 11/32 | 2 | 9 | 128 | 565 | 5120 | 47 | 3153 |
| Supervised + paired ranking | 32302 | 15/32 | 3 | 6 | 128 | 605 | 5120 | 47 | 3150 |
| Soft witness support | 32301 | 17/32 | 1 | 2 | 128 | 769 | 5120 | 47 | 3140 |
| Soft witness support | 32302 | 17/32 | 0 | 1 | 128 | 852 | 5120 | 47 | 3158 |
| QP witness update | 32301 | 17/32 | 0 | 1 | 128 | 697 | 5120 | 47 | 3158 |
| QP witness update | 32302 | 16/32 | 0 | 2 | 128 | 734 | 5120 | 47 | 3149 |
