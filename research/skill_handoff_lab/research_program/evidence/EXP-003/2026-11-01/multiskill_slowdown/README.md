# Transfer slowdown attribution: four real native100 episodes

Two same-scene grasps, each fresh original-stride2 control and TRANSFER-only stride1. All four have100 real control steps and500 saved100Hz samples:400 steps/2000 samples total,0 optimizer updates. GPU0 was shared with A training; times are SHARED_GPU_CONTENTION. Every other controller parameter and selection algorithm is shared. Later feedback plans can change with resulting physical state; only the initial TRANSFER raw path and prefix are guaranteed identical.

Both fresh controls reproduce historical100-step actions, q/qvel/TCP/tool/cube/attachment exactly; all500 force-query substeps per control also exactly match historical probes. Thus explicit100Hz PhysX q/qvel/qacc readback did not change recorded motion or contact. Both slowdowns share their control's exact initial state, first-skill prefix and full raw TRANSFER joint path.

| Measured TRANSFER quantity | g035 original | g035 slower | g070 original | g070 slower |
|---|---:|---:|---:|---:|
| Control steps |20|39|24|46|
|100Hz peak arm velocity rad/s |1.753|0.945|2.040|1.063|
|100Hz peak PhysX qacc rad/s² |59.53|29.60|65.77|33.21|
|Maximum attachment rotation degrees |13.194|13.713|0.270|0.211|
|Rotation before first tool/cube contact degrees |10.410|10.784|0.051|0.054|
|TRANSFER tool/cube peak force N |4.011|2.442|0|0|
|Native terminal task success |FAIL|FAIL|PASS|FAIL|

**Slowing the same joint path reduces impact/speed but does not remove the unfavorable grasp's pre-contact drift or premature contact.** The first nonzero g035 tool/cube contact shifts fromstep58/substep2 to69/substep1, still duringTRANSFER. The entire path takes longer; this is not a fixed-time pose comparison. At matched commanded path progress the large attachment rotation remains. This rejects the narrow explanation that simply halving this commanded timing solves the failure; it does not isolate gravity, friction, compliance and residual inertial contributions individually.

The stronger g070 remains mechanically stable at both timings. Slowdown consumes22 extra TRANSFER steps and turns native success into a horizon failure. This is measured budget cost, not evidence of a learned parameter advantage. No bad-grasp rescue occurred, and the experiment does not establish an advantage over stronger grasp selection.

Dynamic contract status: actual100Hz TRANSFER velocities stay below URDF arm velocity limits (maximum ratio.938 original/.489 slower). These readbacks are stronger than prior20Hz snapshots. They do **not** prove acceleration compliance: the URDF has no acceleration limit, the planner nominal.9rad/s/.9rad/s² TOPP configuration is not preserved by command decimation or the PD plant, and even stride1 has large actual PhysX qacc. Native action-box legality and full-task success must remain separate from a certified dynamic SkillContract. No hardware or deployment claim is made.

Reproduction: `program_multiskill_slowdown_queue.py` via the isolated launcher; analysis `program_multiskill_slowdown_analyze.py`. The native sources and exact per-run source snapshots, raw trajectories,100Hz states/forces and command paths are in RAW_EVIDENCE.tar.gz. SLOWDOWN_AUDIT.json and FETCH_FORCE_AUDIT.json contain the checks. SLOWDOWN_035/070 PNG/SVG compare path-aligned drift and actual speed without resimulating physics.
