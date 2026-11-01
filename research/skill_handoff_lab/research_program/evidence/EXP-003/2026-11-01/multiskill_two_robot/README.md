# Native two-robot push → pick: physical controller scout

All original failures are retained. Native100 horizon, initial distribution and25mm-plus-static success are unchanged. No RL training. Both robots act in the same continuing episode. Timing is SHARED_GPU_CONTENTION, not an exclusive benchmark.

| Case | Controller / stride | Decisions | Native success | Final error mm | Failure |
|---|---|---|---|---:|---|
|742000_continuous_clear|continuous-clear / 3|[0, 59]|True|0.804|None|
|742000_continuous_p020|continuous-clear / 3|[0, 53]|True|3.326|None|
|742000_continuous_p080|continuous-clear / 3|[0, 97]|False|82.176|Horizon: NATIVE100|
|742000_nominal|original / 1|[0]|False|209.499|Horizon: NATIVE100|
|742000_stride3|original / 3|[0, 74]|False|107.748|Failure: RIGHT_NOT_GRASPED|
|742001_continuous_clear|continuous-clear / 3|[0, 58]|False|224.642|Horizon: NATIVE100|
|742001_continuous_p020|continuous-clear / 3|[0, 55]|False|173.350|Horizon: NATIVE100|
|742001_continuous_p080|continuous-clear / 3|[0]|False|247.796|Horizon: NATIVE100|
|742001_nominal|original / 1|[0]|False|306.305|Horizon: NATIVE100|
|742001_p020_aligned|continuous-clear / 3|[0, 55]|False|10.157|None|
|742001_p020_aligned_y|continuous-clear / 3|[0, 55]|False|82.040|Horizon: NATIVE100|
|742001_stride3|original / 3|[0, 79]|False|254.172|Horizon: NATIVE100|

The original full-TOPP controller exhausts its budget while pushing: repeated short segments pay repeated timing overhead. Explicit stride3 reaches a real push boundary but still leaves insufficient suffix budget or poor grasp tracking. The continuous-clear correction uses a longer push segment with actual lateral feedback and withdraws the left hand to y=-.20m/z=.15m before right grasp. It changes controller geometry and timing, not a learned policy. These coupled fixes are not a causal isolation of arm-arm collision; arm-arm contact was not logged.

Full per-step states, both q/qvel, dictionary commands, TCP/cube transforms, link/cube forces, measured SkillResult and decision observations are stored. Tracking maxima in ANALYSIS include transient commanded-path errors; they are not hardware certification. There is no registered inter-arm/world planning obstacle, so this remains a development geometric controller, not a fully collision-aware strong baseline.

A successful two-skill chain establishes feasibility. It does not establish task-dependent second-action ranking or an independent RL contribution. Same-state parameter interventions are needed before learning.

Final diagnostic: the orthogonal aligned-axis pair at742001/endpoint.02 shares the complete physical prefix to step55. Axis_x reaches10.16mm but fails the native static requirement at100; axis_y reaches82.04mm and fails at horizon. These are legitimate distinct second actions, but neither yields full success and no conditional ranking reversal is demonstrated. The next strong baseline should budget complete successor execution plus settling rather than minimize pregrasp motion alone.

The742000 endpoint pair changes complete outcome (20mm target succeeds,80mm fails); current evidence is consistent with a simple smaller-push rule, not yet a task-conditioned RL opportunity. Only two native seeds were explored; raw failures diagnose controller maturity, not task impossibility.
