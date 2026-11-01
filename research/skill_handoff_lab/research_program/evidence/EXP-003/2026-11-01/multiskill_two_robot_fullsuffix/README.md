# Native two-robot push → pick: physical controller scout

All original failures are retained. Native100 horizon, initial distribution and25mm-plus-static success are unchanged. No RL training. Both robots act in the same continuing episode. Timing is SHARED_GPU_CONTENTION, not an exclusive benchmark.

| Case | Controller / stride | Decisions | Native success | Final error mm | Failure |
|---|---|---|---|---:|---|
|742010_full_p020|continuous-clear / 3|[0, 49]|True|1.036|None|
|742010_full_p020_other|continuous-clear / 3|[0, 49]|True|12.237|Horizon: NATIVE100|
|742010_full_p050|continuous-clear / 3|[0, 56]|True|5.397|None|
|742011_full_p020|continuous-clear / 3|[0, 53]|True|3.124|None|
|742011_full_p020_other|continuous-clear / 3|[0, 53]|False|27.600|Horizon: NATIVE100|
|742011_full_p050|continuous-clear / 3|[0, 57]|True|1.643|None|
|742012_full_p020|continuous-clear / 3|[0, 50]|True|4.195|None|
|742012_full_p020_other|continuous-clear / 3|[0, 50]|True|1.100|None|
|742012_full_p050|continuous-clear / 3|[0, 56]|True|2.811|None|
|742013_full_p020|continuous-clear / 3|[0, 54]|True|3.029|None|
|742013_full_p020_other|continuous-clear / 3|[0, 54]|False|35.845|Horizon: NATIVE100|
|742013_full_p050|continuous-clear / 3|[0, 59]|True|2.228|None|

The original full-TOPP controller exhausts its budget while pushing: repeated short segments pay repeated timing overhead. Explicit stride3 reaches a real push boundary but still leaves insufficient suffix budget or poor grasp tracking. The continuous-clear correction uses a longer push segment with actual lateral feedback and withdraws the left hand to y=-.20m/z=.15m before right grasp. It changes controller geometry and timing, not a learned policy. These coupled fixes are not a causal isolation of arm-arm collision; arm-arm contact was not logged.

Full per-step states, both q/qvel, dictionary commands, TCP/cube transforms, link/cube forces, measured SkillResult and decision observations are stored. Tracking maxima in ANALYSIS include transient commanded-path errors; they are not hardware certification. There is no registered inter-arm/world planning obstacle, so this remains a development geometric controller, not a fully collision-aware strong baseline.

A successful two-skill chain establishes feasibility. It does not establish task-dependent second-action ranking or an independent RL contribution. Same-state parameter interventions are needed before learning.

Earlier controller-development panel (742000/742001; separate raw archive): the orthogonal aligned-axis pair at742001/endpoint.02 shares the complete physical prefix to step55. Axis_x reaches10.16mm but fails the native static requirement at100; axis_y reaches82.04mm and fails at horizon. These are legitimate distinct second actions, but neither yields full success and no conditional ranking reversal is demonstrated. The next strong baseline should budget complete successor execution plus settling rather than minimize pregrasp motion alone. Native multi-agent execution permits simultaneous commands, but this controller serializes left release then right approach; budget failure can therefore reflect a shared scheduling limitation. Parallel withdrawal/approach requires actual collision and tracking validation before any learning comparison.

The742000 endpoint pair changes complete outcome (20mm target succeeds,80mm fails); current evidence is consistent with a simple smaller-push rule, not yet a task-conditioned RL opportunity. That earlier panel covered only two native seeds; raw failures diagnose controller maturity, not task impossibility. Current table and metrics above identify the present panel separately.
