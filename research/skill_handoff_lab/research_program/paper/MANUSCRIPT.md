# Execution-reference separation for task-conditioned skill-parameter reinforcement learning

Working manuscript, 2026-11-05. This title describes a candidate representation contribution, not a proven new SAC algorithm. Native simulation evidence only; no hardware or sealed-test claims.

## Problem

A small skill parameter is not necessarily a small change in robot execution. In a planner-backed manipulation skill, a command can both reconstruct a target from the currently measured object–gripper transform and add a local offset. The resulting trajectory change mixes reference correction with residual adaptation. We study whether exposing these operations separately improves reinforcement learning across an actual grasp-to-insertion handoff while preserving strong nominal execution.

The task is to select a grasp parameter and subsequently adapt a contact-rich skill from physical feedback under the original task horizon. The objective is terminal native success; paired rescues and regressions and physical/planning costs diagnose reliability. We do not redefine success or treat planned kinematic feasibility as insertion success.

## Related work

Residual learning is an established foundation. rLfD (Davchev et al., RA-L 2022, https://arxiv.org/abs/2008.07682) corrects a demonstrated DMP using task-space/full-pose residual policies with an impedance controller. Its DMP runs at 100 Hz and learned correction at 10 Hz; our endpoint modification followed by a full motion replan is not that low-level residual interface. Johannink et al. (https://arxiv.org/abs/1812.03201) establish combining conventional control with learned residuals. Neither additive correction nor combining a strong controller and RL is claimed novel here.

Fabrica (https://fabrica.csail.mit.edu/, official code https://github.com/yunshengtian/Fabrica) integrates assembly planning and learned execution. It is a close planning/learning reference, not evidence that our actual grasp attachment can be treated as fixed. Policy Decorator (https://arxiv.org/abs/2412.13630) motivates base-policy protection and controlled exploration; our successful-trajectory BC and bounded exploration are mature comparators, not inventions. Option/SMDP execution and reference continuation are likewise foundations inherited from prior experiments.

The candidate distinction is operational: separately retaining an active plan, explicitly refreshing its measured reference, and modifying its existing endpoint. Its value must be established against standard residual/replanning representations under matched training, not asserted from terminology.

## Method

Let x be physical state, Q the current joint-plan queue and cursor, T its requested endpoint, and b the remaining native budget. An executable action is a map on this execution state, not an additive motor command. The actor receives the same 64-dimensional observation in all arms: measured robot/object/goal quantities, stage/budget, and queue summaries. Full Q and T are logged but are not privileged actor inputs; partial execution-state observability is a limitation.

The first boundary selects one of three physical grasp offsets. At subsequent open Option boundaries, eight legal choices retain existing shared authority: continue; refresh the target from actual attachment; goal-frame Y or Z offsets of ±3 mm; withdraw 6 mm; or hold. In the separated interface, offset actions use p'=p_T+R_goal δ and R'=R_T. In the measured interface, they use p'=p_F(x)+R_goal δ and R'=R_F(x), where F reconstructs the target from actual attachment. Thus the requested position change includes p_F(x)−p_T in the latter. This decomposition is not a measured numerical amplification factor.

Action 0 continues the current queue, including earlier accepted corrections. It never restores an original nominal trajectory. Accepted endpoint updates invoke the same full MPLib planner and shared retiming. They do not guarantee a 3 mm realized displacement or smooth local motor perturbation. Failed replans retain the existing queue. Hold and initial-plan-failure cases anchor the relative interface to the actual held TCP. These are execution semantics, not novel RL updates.

Both interfaces use the inherited discrete SMDP SAC learner and reference-continuation Option. Physical durations and remaining budgets are recorded in Option transitions. The current finite-horizon learner uses undiscounted return (gamma=1), so it does not apply an additional duration discount. The frozen reference governs continuation until the inherited termination condition. In this implementation that condition uses grasp loss or queue exhaustion together with a capped planner-call-count feature (slot 49), not a separately observed semantic phase label. It is shared by all arms and remains an implementation limitation; we do not claim a verified semantic contact-event termination mechanism. The reference weights are identical across arms, though the interface changes their physical action meaning too. Successful-trajectory BC is the second learning factor. Every online arm collects its own trajectories; transitions are never relabeled across interfaces. Three grasp parameters remain learned throughout.

## Preliminary experiments

### Matched online factorial and continuation

We retain the historical measured-interface endpoints and collect the two missing active-target cells using the same inherited neural checkpoint, 64 training initial states, replay/exploration seed 31601 and 5,120 Actor/Critic iterations per arm. Every arm collects its own physical experience. The 16 development initial states are previously seen; they are not independent testing. Three real grasp choices remain learned throughout.

| Interface / learner | Development success at 0/32/64 episodes | Final64 rescues / regressions versus nominal |
|---|---|---|
| Measured / SAC | 9 / 10 / 6 | 1 / 4 |
| Measured / SAC + successful BC | 9 / 8 / 9 | 1 / 1 |
| Active target / SAC | 9 / 7 / 6 | 0 / 3 |
| Active target / SAC + successful BC | 9 / 8 / 8 | 0 / 1 |

The shared nominal controller scores 9/16. The frozen measured and active-target rules score 8/16 and 9/16. The factorial does **not** establish a development-set advantage for the separated interface. Its final SAC/BC policies make 30/19 feedback replans versus 20/11 for the measured policies; all these replans return success. Kinematic planning status is therefore insufficient to explain terminal task outcome.

We then extend the stronger BC learner in both interfaces by the same additional 64 new training episodes, preserving each arm's model, optimizer, replay and RNG. At cumulative64/96/128 episodes, measured BC scores 9/8/8 and active-target BC 8/7/7. An independent replay/exploration repeat (31602) from the common inherited weights scores 8/7 and 7/6 at64/128. These are independent stochastic learning runs, not independent neural initializations. The predeclared highest-development-score/earliest-tie rule selects64 for all four policies. Failed later checkpoints and their full costs remain included. Historical final64 is never replaced by a selected score.

### Frozen confirmation on new natural conditions

All four models were frozen before evaluating initial states760200–760215. After both active-target runs showed fewer regressions, we declared a second cohort760216–760231 without changing any policy or rule. This adaptive extension is disclosed; the results are preliminary descriptive evidence, not a sealed fixed-sample significance test.

| Frozen method | Cohort1 /16 | Cohort2 /16 | Total /32 | Rescue / regression vs nominal | Planning queries |
|---|---:|---:|---:|---:|---:|
| Measured BC,31601 | 5 | 9 | 14 | 0 / 5 | 250 |
| Active-target BC,31601 | 7 | 10 | 17 | 0 / 2 | 269 |
| Measured BC,31602 | 4 | 8 | 12 | 0 / 7 | 262 |
| Active-target BC,31602 | 7 | 11 | 18 | 0 / 1 | 260 |
| Measured frozen rule | 6 | 11 | 17 | 1 / 3 | 262 |
| Active-target frozen rule | 5 | 11 | 16 | 0 / 3 | 256 |
| Nominal continuation | 7 | 12 | 19 | 0 / 0 | 221 |

The supported positive trend is **reduced policy-induced regression under the separated execution representation**, relative to the same mature learner. It is not new nominal-failure recovery, and nominal continuation remains strongest. Planning cost is not uniformly reduced: the first active-target run uses19 more queries than its measured comparator. The32 initial states are shared across methods and training repeats and must not be treated as64 independent scenes.

### Physical mechanism and competing explanations

Historical same-prefix probes distinguish a requested offset from reference reconstruction. A measured3mm correction can change the requested endpoint by up to21.98mm because the reference also changes; this is not a verified sevenfold numerical amplification. Across16 signed-Z comparisons in eight reachable grasp/scene contexts, median maximum first-four-step joint-command deviation from continuing is0.01233rad for measured-reference correction and0.00344rad for active-target correction. These are joint-vector command norms, not TCP displacement or tracking error. Shared contexts are not independent samples. Reduced immediate disturbance does not imply the online development gains absent above.

Twelve new full-task shared-grasp diagnostics separate predecessor and successor behavior. In759205, the active-target policy succeeds only with grasp0, whereas the measured policy succeeds with all three grasps. In759202, only the measured policy with grasp0 succeeds. These selected development cases are mechanism evidence, not new generalization scores.

Case759205/grasp1 provides a temporal counterexample. After94 identical recorded physical steps, the active-target policy requests3mm, replacing a queue at cursor8/16 with a new15-point queue at cursor0 while only6 native steps remain. Continuing succeeds; replanning fails. A further controlled experiment replans to the **identical endpoint**, producing14 points and still failing. Replanning is sufficient to cause failure without endpoint movement. It does not uniquely separate changes in joint-path geometry, initial-velocity treatment and retiming. Four additional natural mechanism cases show unchanged terminal outcomes among continue, zero-offset replan and3mm replan. These15 episodes support a counterexample, not a population frequency estimate.

“Identical prefix” means equality of the recorded initial state and per-step joint positions/velocities, TCP/peg poses and motor commands under full native reexecution. No approximate PhysX state restore is used; every hidden solver variable is not inspected. At the failing94-step state, the frozen Actor assigns approximately0.66 probability to action2 even though the Critic's mean estimate of continue is slightly larger than action2. The Critic prefers another untested action most, so this is not evidence that greedy-Q deployment would solve the case. Q values include shaping and entropy and are not calibrated success probabilities.

Two competing explanations remain open. Matching inherited weights also transfers a Critic trained under different physical action semantics; equal initialization does not imply equal adaptation difficulty. The actor also lacks complete endpoint/queue context. Neither explanation is causally established here. The frozen reference issued only continue in the active-target BC final64 evaluation, excluding repeated reference intervention as the explanation for its extra replans.

### Actual costs and reproducibility

This cycle adds512 training episodes,51,200 training physical steps and3,467 effective RL decisions. Actor and Critic each receive40,960 updates. Legacy auxiliary networks receive the same number of iterations but are not consumed by the ordinary policy or Bellman target; their cost is retained. Optimization totals208.10 seconds.

Training, evaluation and mechanism experiments total1,067 complete native episodes,106,700 physical steps and8,764 planning queries. Summed planner time is41.86 seconds and summed episode time6,303.06 seconds; these are sums across concurrent runs, not elapsed wall time. Historical measured-cell costs are separate. All raw traces, source snapshots, named checkpoints, final transactional resume states, failures and SHA records are retained. Video rendering replays measured states and adds no physical steps.

## Revised method direction and limits

The next candidate is **reference- and progress-explicit skill adaptation**. A reasonable standard local-residual comparator modifies the unexecuted nominal waypoints smoothly while preserving their schedule, subject to IK, collision and shared dynamic checks. Explicit reference refresh remains separate. An infeasible correction must retain the current queue; a15-point plan cannot be compressed into6 steps or granted additional native budget. This comparator is prepared conceptually, not claimed implemented or trained here. Standard residuals and execution fixes are not renamed as algorithmic originality.

The existing native PlugCharger200 implementation provides a concrete second-task reuse path: real grasp pitch15–45 degrees followed by contact advance/angular pacing, with measured attachment and native success. `program_native_charger_sac.py` supplies ordinary two-boundary SAC and `program_charger_feedback.py` the strong local IK servo. Its54-dimensional continuous interface is incompatible with Peg's64-dimensional discrete replay/weights. Transfer should examine actual servo reference/progress semantics, not manufacture a queued-replanner failure. Historical strong depth-preserving control is retained; no new second-task result is asserted.

The current evidence supports an execution-semantics problem, a trained representation variant, a preliminary retention benefit on frozen new conditions, and a physically tested zero-offset-replanning counterexample. It does not establish superiority over strong nominal control, new recovery capability, an independent SAC innovation, robust cross-task transfer or publication-level generalization. The negative seen-development results, shared neural initialization, adaptive confirmation extension and partial execution-state observation remain explicit. Problem, Method and Experiments can now be developed from these concrete claims while the next implementation addresses the demonstrated execution deficiency.
