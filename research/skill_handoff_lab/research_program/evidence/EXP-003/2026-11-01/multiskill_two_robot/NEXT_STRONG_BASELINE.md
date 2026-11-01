# Complete-successor geometry baseline — preserved pre-run protocol

Status update: the bounded12 was subsequently executed; current results are in ../multiskill_two_robot_fullsuffix/SCIENCE_NOTES.md. The prospective design below is preserved for audit.

`program_multiskill_two_robot.py --controller continuous-clear --stride3 --axis-objective full-suffix` (CLI spelling: `--stride 3`) now uses both orthogonal cube grasp axes, with symmetric sign normalization shared by all policies. Selection starts at the actual physical push/release boundary and plans pregrasp, descent, lift and goal transfer. It predicts six close steps plus six settling-reserve steps, retains raw planned sequences and command-decimation indices, and ranks complete predicted suffix cost. Fixed cube_x/cube_y interventions use the same rule and physical execution. Full-suffix mode implies sign normalization.

Predicted grasp attachment assumes the current cube relative to the target grasp pose. This is a nominal rigid-contact assumption, not an actual execution outcome, and is explicitly logged. Subsequent actual transport still uses the measured grasp attachment and observed goal correction. Complete plan cost is a falsifiable model contract, not a learned value label or guaranteed physical duration.

No native task state, horizon100, success criterion, physics parameter or joint limit is changed. Stride3 remains absolute-target command decimation; it is not native TOPP timing. Inter-arm/world swept collision registration remains absent and must not be represented as a complete collision-aware controller.

Minimal proposed next panel, no GPU launched yet:

1. Four NEW native scenes742010–742013 × endpoints.02/.05: eight shared complete-suffix geometry baseline episodes. This tests whether the previous shorter-push dominance survives new physical initial states after the avoidable pregrasp-only selector is strengthened.
2. The same four scenes at endpoint.02 with the other orthogonal axis than the model chose: four paired physical interventions. Choice of the intervention is based on the planner's selected axis, not on outcome. This measures whether the complete model ranking is correct at an identical physical first-skill boundary.
3. Only if first endpoint changes which second axis is genuinely favorable should the remaining endpoint.05 orthogonal pairs be added (at most four). No RL training or statistical generalization claim follows from this scout.

The first eight are prerequisite information, not a gate forbidding useful repairs. A systematic remaining failure should trigger one shared controller diagnosis rather than fit RL around a known timing or geometry defect. Core question: after strong complete-successor geometry, does actual execution provide repeatable conditional information that changes the best legal parameter and full native success?

Scheduling limitation: the native environment supports simultaneous arm commands. The frozen pilot serializes left withdrawal and right approach, potentially consuming avoidable horizon budget. Budget-dominated failure must not be interpreted as RL opportunity before a shared concurrent schedule is physically collision/track validated. No such concurrency is introduced into the current bounded12 protocol.
