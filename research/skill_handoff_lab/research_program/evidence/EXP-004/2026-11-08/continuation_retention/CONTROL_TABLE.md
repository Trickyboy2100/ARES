| Policy | Success | Rescue | Regression | Own budget | Scope |
|---|---|---|---|---|---|
| joint_sac_32301 | 16/32 | 2 | 4 | 128 | Joint front/rear control; different front permission explicitly separated |
| warm | 18/32 | 0 | 0 | 0 | Frozen shared controller; no new training |
| rule | 18/32 | 1 | 1 | 0 | Frozen shared controller; no new training |
| fixed5 | 24/32 | 9 | 3 | 0 | Frozen shared controller; no new training |
| prior_sac32202 | 21/32 | 4 | 1 | 128 | Historical model/different data budget; front frozen here to shared warm, not historical score |
| old_rl | 18/32 | 1 | 1 | 64 | Historical different single-deviation interface and front; reference only |
| witness_margin_64_32301 | 18/32 | 1 | 1 | 64 | Matched64 exploratory margin ablation, single network seed |
| sac_64_32301 | 19/32 | 2 | 1 | 64 | Matched64 exploratory margin ablation, single network seed |
| witness_soft_64_32301 | 17/32 | 1 | 2 | 64 | Matched64 exploratory margin ablation, single network seed |
