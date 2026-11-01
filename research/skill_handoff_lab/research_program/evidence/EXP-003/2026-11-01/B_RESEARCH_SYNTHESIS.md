# Workstream B — real multi-skill opportunity analysis, frozen after physical attribution

## Scientific decision

No independent RL method gain is claimed by this workstream. It did establish genuine physically separated decisions in native PullCubeTool and TwoRobotPickCube, distinguish controller geometry/timing defects from physical handoff effects, and test a stronger whole-successor model before committing to learning. The stronger TwoRobot model solves all8 predeclared new-scene baseline cases; the tested orthogonal alternatives do not outperform it. Pull slowdown does not fix the poor grasp's drift. Expanding RL on these observed deficits would currently risk learning around an avoidable controller deficiency or inferior grasp.

The conditional-learning hypothesis remains open beyond these bounded panels. Naive kinematic/budget margin is not itself a calibrated physical value: Pull gives false positives, TwoRobot conservative duration gates give seven false negatives. The untrained distribution/relabeling candidate described in prior-art notes was not implemented or falsified by these diagnostics.

## Actual physics inventory (fresh controls counted, repeats not called independent scenes)

| Evidence root | Episodes | Actual control steps | Native terminal passes | Purpose |
|---|---:|---:|---:|---|
|multiskill_pull|31|3100|16|grasp/hook/pull-distance physical interventions|
|multiskill_contact_probe|2|200|1|exact-repeat100Hz contact attribution|
|multiskill_pyramid|10|2500|5|support axes and deterministic gripper-clearance repair|
|multiskill_clearance|6|600|2|TRANSFER endpoint height, same first-skill prefixes|
|multiskill_two_robot|12|1200|2|native task/controller development and paired parameters|
|multiskill_two_robot_fullsuffix|12|1200|10|strong complete-successor model8/8, orthogonal alternatives2/4|
|multiskill_slowdown|4|400|1|two exact fresh controls plus two same-path slowdowns|
|**Total physical cost**|**77**|**9200**|—|**0 optimizer updates in B**|

Do not interpret the mixed-purpose total as a benchmark success rate. There are14 unique native scene seeds across three task families, with repeated controls and diagnostic interventions.6000 actual100Hz contact samples are saved (1000 contact probes,3000 endpoint-height cases,2000 slowdown cases). Only the slowdown2000 additionally save explicitly fetched100Hz q/qvel/qacc; earlier robot motion snapshots are20Hz. Video restoration adds zero physics steps.

All added clearance/TwoRobot/slowdown runs share GPU0 with authorized A work; resource_context is SHARED_GPU_CONTENTION. They are not exclusive timing benchmarks. No other GPU or hardware device was used.

## Mechanistic findings

- **Pull physical attachment:** g035 has10.41° measured attachment drift before its first nonzero tool/cube contact; g070 remains stable.100Hz probes expose contact missed by20Hz end frames. All queried robot/cube forces are zero in the exact-repeat pair; an earlier possible robot-contact explanation is not supported here.
- **Pull path and speed:** Raising only endpoint height from8cm to10/12cm rescues none of six cases and does not constrain the entire swept path. Slowing the exact TRANSFER joint path reduces g035 peak speed1.753→.945rad/s and impact4.01→2.44N but drift remains13.19→13.71°. Stronger g070 remains stable but loses native success because the slowdown costs22 extra steps. This separates impact magnitude from unresolved attachment/path geometry and budget; it does not prove a uniquely identified gravity/friction cause.
- **Pyramid controller defect:** Intended finger opening overlaps the neighboring support in failed side-axis cases. A shared axis-aware orientation repair rescues one of two targeted failures with an exact pre-placement prefix; the other is kinematically infeasible. This is deterministic controller repair, not RL.
- **TwoRobot actual two-boundary chain:** Native push→measured-state right grasp/goal runs without resets. Earlier segmented push and pregrasp-only planning fail budget. A shared continuous push/withdraw controller plus complete successor geometry achieves8/8 on four new native scenes×two endpoints. Four exact-boundary orthogonal alternatives score2/4, with no conditional reversal. The two failed alternatives retain grasp but finish outside25mm and nonstatic at100. Serial arm scheduling and lack of registered inter-arm swept obstacles remain known baseline limitations.
- **Value signal limits:** Pull nominal budget TP15/FP6/FN1/TN9 across31 diagnostic runs; repetitions and parameter duplicates are separately identified. TwoRobot complete-duration threshold TP3/FP0/FN7/TN2 across12 new runs; predicted first-success duration is conservatively high by2–7steps on ten successes. Model ranking succeeds while binary feasibility calibration is poor. Neither result by itself establishes a new RL mechanism.

## Executability is not certified dynamics

Native action-box legality, original task success and an exact joint path are distinct from a complete dynamic SkillContract. mplib defaults unit velocity/acceleration and the solver scales to.9; stride2/3 command decimation does not preserve that TOPP timing envelope. TwoRobot20Hz speeds reach.993×URDF arm velocity bounds;100Hz peaks were not measured there. Pull slowdown explicitly measures100Hz TRANSFER velocities below URDF arm bounds, but actual PhysX qacc remains far above the nominal planner acceleration. URDF acceleration limits are absent. The evidence therefore marks dynamic certification NOT_PROVEN, with no hardware-safety or deployment claim.

## Frozen next research implication

Do not train a new policy solely to beat the weak pregrasp-only selector, choose a symmetric gripper sign, repair a known opening collision, or shorten a serialized schedule. The next independent method test needs a residual physical parameter ranking that persists after shared complete-successor geometry and legitimate dynamics/runtime contracts. Current B evidence does not establish that ranking; retain the negative result and use A's separately trained contact-adaptation findings to decide the paper's next mechanism. No additional task or seed expansion was performed after the bounded slowdown attribution.

Each evidence directory contains exact-source raw archives, SHA manifests, necessary analysis scripts/figures and scientific notes. See particularly multiskill_two_robot_fullsuffix/SCIENCE_NOTES.md, multiskill_slowdown/README.md and the paired-prefix/dynamics audits.

Completion-contract qualification: Pull PICK_TOOL only checks retained native grasp after commanded55mm lift, not attachment stability or force closure. g035 drift may expose a weak precursor completion contract and is not automatically a clean “successful prefix needs RL adaptation” case. TwoRobot on-table/y≥0 likewise is not enforced rest, although velocities are observed. See B_CONTRACT_SCOPE_AND_NEXT_EXPERIMENT.md for qualified three-cause attribution and a predeclared stable-boundary/moderate-load falsifier.
