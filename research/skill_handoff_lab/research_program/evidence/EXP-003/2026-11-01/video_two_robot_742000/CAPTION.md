# Native TwoRobotPickCube parameter pair — not RL gain

Same seed742000 and exact initial-state hash; common continuous-clear controller, stride3 absolute joint targets. Only commanded push endpoint differs (20mm versus80mm). Stride3 is command decimation and is **not native TOPP timing**. Sources were executed under SHARED_GPU_CONTENTION; wall times are not isolated performance benchmarks.

Both execute100 actual native simulation steps with two decisions. 20mm: handoff at53, actual cube y20.094mm; right grasp axis cube_y; final distance3.326mm and native success. 80mm: handoff at97, actual y78.413mm; same chosen axis cube_y; only3 steps remain, final82.176mm and Horizon:NATIVE100. The failure shown is a downstream time-budget failure, not evidence of intrinsically ungraspable contact. First boundaries require y>=0/on-table; measured velocity is retained, not silently certified static. Native task final criterion is goal distance<=25mm AND right-arm static.

Video is MEASURED_POSE_REPLAY of saved full simulator states, not another execution or physical robot. It preserves101 frames including reset and the full100step horizon. Full future curves/terminal outcomes are offline analysis, not actor observations. Each decision and SkillResult is exposed only when actually reached. Original20Hz contact samples can miss faster contacts; no100Hz evidence is asserted here.

This pair establishes a reproducible physical parameter/budget contrast under this controller, not a learned-method gain or proof that strong budget-aware control cannot solve it. No new optimizer or policy training was performed for this pair. Inter-robot collision geometry is not registered with the planner; this remains an offline development controller.
