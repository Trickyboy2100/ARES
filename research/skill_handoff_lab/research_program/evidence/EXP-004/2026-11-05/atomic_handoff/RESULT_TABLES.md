| Temporal interface / NN seed | Dev0 | Dev32 | Dev64 (primary) | Final rescue / regression | Own train successes | Effective decisions | Actor / critic updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| macro_bc64 | 10/16 | 11/16 | 8/16 | 1 / 3 | 19/64 | 348 | 1280 / 1280 |
| macro_seed31702 | 10/16 | 11/16 | 12/16 | 2 / 0 | 26/64 | 339 | 1280 / 1280 |
| phase_bc64 | 10/16 | 6/16 | 6/16 | 0 / 4 | 19/64 | 443 | 1280 / 1280 |
| queue_bc64 | 10/16 | 10/16 | 9/16 | 0 / 1 | 27/64 | 300 | 1280 / 1280 |
| queue_seed31702 | 10/16 | 9/16 | 7/16 | 0 / 3 | 20/64 | 346 | 1280 / 1280 |

| Frozen method | Success | Rescue | Regression |
|---|---:|---:|---:|
| confirm_queue_bc64 | 12/24 | 0 | 0 |
| confirm_macro_bc64 | 10/24 | 1 | 3 |
| confirm_queue_seed31702 | 12/24 | 0 | 0 |
| confirm_macro_seed31702 | 12/24 | 1 | 1 |
| confirm_rule | 15/24 | 4 | 1 |
| nominal | 12/24 | 0 | 0 |

| Actual grasp pitch | Nominal pace | Feedback pace |
|---|---:|---:|
| 15 degrees | 4/4 | 4/4 |
| 30 degrees | 4/4 | 4/4 |
