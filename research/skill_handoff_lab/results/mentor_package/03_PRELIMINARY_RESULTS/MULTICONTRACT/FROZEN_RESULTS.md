# Frozen measured results

|Split|Method|Seed outcomes (success/64)|Mean success fraction ± seed SD|Mean episode seconds ± seed SD|
|---|---|---|---|---|
|validation|contract|31: 64/64, 47: 64/64, 59: 64/64|1.000000 ± 0.000000|1.630035 ± 0.000150|
|validation|contract_blind|31: 64/64, 47: 64/64, 59: 64/64|1.000000 ± 0.000000|1.634201 ± 0.007367|
|validation|nominal|31: 64/64|1.000000 (one deterministic baseline)|1.629948 (one deterministic baseline)|
|validation|ppo|31: 0/64, 47: 0/64, 59: 0/64|0.000000 ± 0.000000|3.937153 ± 0.048864|
|validation|residual|31: 64/64, 47: 64/64, 59: 64/64|1.000000 ± 0.000000|1.630469 ± 0.000902|
|test|contract|31: 64/64, 47: 64/64, 59: 64/64|1.000000 ± 0.000000|1.666667 ± 0.000260|
|test|contract_blind|31: 64/64, 47: 64/64, 59: 64/64|1.000000 ± 0.000000|1.671181 ± 0.008273|
|test|nominal|31: 64/64|1.000000 (one deterministic baseline)|1.666667 (one deterministic baseline)|
|test|ppo|31: 0/64, 47: 0/64, 59: 0/64|0.000000 ± 0.000000|3.884115 ± 0.133822|
|test|residual|31: 64/64, 47: 64/64, 59: 64/64|1.000000 ± 0.000000|1.667014 ± 0.001084|

Each training seed reuses the same 64 scene cases; do not treat three seeds as 192 independent scenes. A 64/64 Wilson 95% interval is approximately [94.3%, 100%], not proof of population success=100%.

V19 is a pilot BEFORE measured-workspace termination was added in v21: command projection did not prevent 15–16mm workspace excursions in some FAILED PPO trajectories. Successful cases independently pass. Keep original failures/time; do not present this as the corrected formal-training comparison.

All methods use the predeclared 16,384 interaction checkpoint; no validation checkpoint selection. Standard PPO at this small pilot budget is not a converged-algorithm verdict. Episode time includes failures; success-only time is stored separately in JSON.

Raw endpoint success is cross-checked by independent per-process convex geometry, nonadjacent right-arm self collision, soft limits, measured pose and workspace audit; inspect those reports before calling a trajectory qualified. No physical contact grasp or full robot safety claim.
