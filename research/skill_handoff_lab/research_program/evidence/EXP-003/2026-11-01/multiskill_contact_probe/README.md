# Substep physical contact diagnosis: distinguish force aliasing from robot collision

Two additional native100 episodes,200 control steps /1000 underlying100Hz physics steps,zero optimizer updates. These are real physical repeats with a read-only `_after_simulation_step` hook; no simulation steps, target commands or controller limits were added/changed.

Every action, q, qvel, TCP/tool/cube pose and TCP-relative attachment matches the corresponding earlier trajectory **exactly at all100 control frames**, and initial state hashes match. Both traces have exactly five contact samples per control frame. `CONTACT_AUDIT.json` contains every maximum absolute difference (all0), old/new trajectory SHA256, per-phase contact quantities and events. No contact at reset is interpreted.

| Actual measurement |35mm grasp: failure|70mm grasp: success|
|---|---:|---:|
| Tool/cube peak during TRANSFER,100Hz |4.011N|0N|
| Tool/cube peak during HOOK,100Hz |1.590N|16.614N|
| Any robot link/cube reported force,all100Hz samples |0N|0N|
| TRANSFER attachment rotation drift from endLIFT |13.194°|0.270°|
| TRANSFER attachment translation drift from endLIFT |7.017mm|0.288mm|
| First tool/cube event above0.5N |frame58/substep2,TRANSFER|frame69/substep3,HOOK|

The failed rollout has premature **tool/cube** contact during transfer, which its original20Hz end-frame force record largely missed. The data do not support the earlier robot/cube collision hypothesis for this pair: every recorded robot-link/cube impulse query is zero. That negative result is retained. Large attachment drift accompanies the failed transfer; exact causal contributions of grasp compliance, rigid swept geometry and trajectory timing are not separated by this probe.

The source-confirmed omission of held-object/environment collision geometry remains a controller-model limitation. It must be addressed before attributing learned gains to a novel RL mechanism. This result does not prove a new clearance controller works, nor establish useful second-parameter conditional ranking.

`CONTACT_MECHANISM.png/svg` overlays100Hz within-control-frame peak and original20Hz end-frame tool force, plus native cube/base distance. It preserves the distinction between sample rates. Force-norm×dt sums in JSON are contact-exposure summaries, not net impulse or causal decomposition. Raw full traces and exact code/manifests are in `RAW_EVIDENCE.tar.gz` with per-file hashes.
