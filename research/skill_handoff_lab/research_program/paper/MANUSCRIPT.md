# Executable skill adaptation above strong manipulation controllers

Working manuscript, 2026-11-05. The current study combines explicit skill execution semantics with baseline-relative learning of complete Option interventions. Historical representation experiments remain below. Independent algorithm superiority is not assumed. Native simulation evidence only; no hardware or sealed-test claims.

## Problem

A small skill parameter is not necessarily a small change in robot execution. In a planner-backed manipulation skill, a command can both reconstruct a target from the currently measured object–gripper transform and add a local offset. The resulting trajectory change mixes reference correction with residual adaptation. We study whether exposing these operations separately improves reinforcement learning across an actual grasp-to-insertion handoff while preserving strong nominal execution.

The task is to select a grasp parameter and subsequently adapt a contact-rich skill from physical feedback under the original task horizon. The objective is terminal native success; paired rescues and regressions and physical/planning costs diagnose reliability. We do not redefine success or treat planned kinematic feasibility as insertion success.

## Related work

Residual learning is an established foundation. rLfD (Davchev et al., RA-L 2022, https://arxiv.org/abs/2008.07682) corrects a demonstrated DMP using task-space/full-pose residual policies with an impedance controller. Its DMP runs at 100 Hz and learned correction at 10 Hz; our endpoint modification followed by a full motion replan is not that low-level residual interface. Johannink et al. (https://arxiv.org/abs/1812.03201) establish combining conventional control with learned residuals. Neither additive correction nor combining a strong controller and RL is claimed novel here.

Fabrica (https://fabrica.csail.mit.edu/, official code https://github.com/yunshengtian/Fabrica) integrates assembly planning and learned execution. It is a close planning/learning reference, not evidence that our actual grasp attachment can be treated as fixed. Policy Decorator (https://arxiv.org/abs/2412.13630) motivates base-policy protection and controlled exploration; our successful-trajectory BC and bounded exploration are mature comparators, not inventions. Option/SMDP execution and reference continuation are likewise foundations inherited from prior experiments.

The candidate distinction is operational: separately retaining an active plan, explicitly refreshing its measured reference, and modifying its existing endpoint. Its value must be established against standard residual/replanning representations under matched training, not asserted from terminology. Equal action counts do not imply equal execution authority: measured offsets can refresh and shift atomically, whereas the separated offsets cannot. Temporal access to decisions must also be matched or treated explicitly as the experimental factor.

## Method

Let x be physical state, Q the current joint-plan queue and cursor, T its requested endpoint, and b the remaining native budget. An executable action is a map on this execution state, not an additive motor command. The actor receives the same 64-dimensional observation in all arms: measured robot/object/goal quantities, stage/budget, and queue summaries. Full Q and T are logged but are not privileged actor inputs; partial execution-state observability is a limitation.

The first boundary selects one of three physical grasp offsets. At subsequent open Option boundaries, eight labels expose different one-boundary physical maps: continue; refresh the target from actual attachment; goal-frame Y or Z offsets of ±3 mm; withdraw 6 mm; or hold. In the separated interface, offset actions use p'=p_T+R_goal δ and R'=R_T. In the measured interface, they use p'=p_F(x)+R_goal δ and R'=R_F(x), where F reconstructs the target from actual attachment. Thus the requested position change includes p_F(x)−p_T in the latter. This decomposition is not a measured numerical amplification factor.

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

## Historical 9a66e94 method direction and limits (superseded below)

The next candidate is **reference- and progress-explicit skill adaptation**. A reasonable standard local-residual comparator modifies the unexecuted nominal waypoints smoothly while preserving their schedule, subject to IK, collision and shared dynamic checks. Explicit reference refresh remains separate. An infeasible correction must retain the current queue; a15-point plan cannot be compressed into6 steps or granted additional native budget. This comparator is prepared conceptually, not claimed implemented or trained here. Standard residuals and execution fixes are not renamed as algorithmic originality.

The existing native PlugCharger200 implementation provides a concrete second-task reuse path: real grasp pitch15–45 degrees followed by contact advance/angular pacing, with measured attachment and native success. `program_native_charger_sac.py` supplies ordinary two-boundary SAC and `program_charger_feedback.py` the strong local IK servo. Its54-dimensional continuous interface is incompatible with Peg's64-dimensional discrete replay/weights. Transfer should examine actual servo reference/progress semantics, not manufacture a queued-replanner failure. Historical strong depth-preserving control is retained; no new second-task result is asserted.

The current evidence supports an execution-semantics problem, a trained representation variant, a preliminary retention benefit on frozen new conditions, and a physically tested zero-offset-replanning counterexample. It does not establish superiority over strong nominal control, new recovery capability, an independent SAC innovation, robust cross-task transfer or publication-level generalization. The negative seen-development results, shared neural initialization, adaptive confirmation extension and partial execution-state observation remain explicit. Problem, Method and Experiments can now be developed from these concrete claims while the next implementation addresses the demonstrated execution deficiency.


## Current revision: atomic execution and temporal authority (2026-11-05)

The new evidence changes the interpretation of the earlier interface comparison. On the previously viewed32 conditions, fixing the real predecessor to grasp1 yields15/32 for the measured successor and17/32 for the relative successor (seed31601). Swapping predecessor policies gives13/32 and17/32 respectively, versus original joint14/32 and17/32. This posthoc decomposition retains a successor difference but shows that predecessor effects and individual scene identities matter; the joint difference is not an isolated rear-controller causal estimate.

At760213, all branches reexecute the actual prefix from the native initial state. At step76, one refresh, one local offset, one atomic refresh+offset, and sequential refresh/offset all fail. Repeated atomic updates at subsequent refinement boundaries succeed; insertion-queue commitment prevents the necessary updates, while phase reopening preserves them. An initial epsilon-based probe triggered at72 instead and is retained separately; the exact76-step experiment confirms the result. A parameterized closed-loop Option reproduces the original frozen-rule success across all100 recorded physical steps. This establishes an executable standard skill capability, not a learned generalization result.

The new common interface has12 actions: continuation, refresh, four active-target offsets, withdrawal, hold, and four atomic refreshed offsets. All learned arms receive83 observed features, including executor phase and current reference differences; they still do not receive the full future queue or future outcome. Fresh neural weights are initialized by512 standard supervised nominal-policy updates on16 actual native training episodes. Online arms retain separate empty replay initially and learn all three grasp choices. Shared discrete SAC and successful-episode BC are unchanged.

The controlled learning factor is temporal execution: queue commitment, semantic phase reopening, or a parameterized atomic Option that refreshes the selected offset at refinement boundaries and then continues insertion. The last changes the temporal policy class; it is not an equal-primitive-frequency algorithm-only comparison. A policy's zero action continues the current queue. No branch extends the100-step horizon or uses waypoint count as a success feasibility oracle. Final64 remains the original endpoint; future frozen confirmation uses a separately declared identical development-selection rule. All declared jobs completed; the final tables and limitations are reported below.

### Executable action authority and temporal policy class (2026-11-05)

We model a feedback decision by its reference transformation, physical planner call,
and permitted continuation, rather than by its integer action label alone. Let
\(T_a\) denote the active target and \(T_r(x)\) the target reconstructed from the
measured grasp. Local adaptation applies \(T_a\Delta(\delta)\); atomic adaptation
applies \(T_r(x)\Delta(\delta)\) at one execution boundary. A refresh followed later
by a local action is not generally equivalent: intervening physics, queue ownership,
and remaining time can differ. Action zero continues the active queue; it does not
restore an earlier nominal plan. These are execution semantics, not a novel SAC loss.

The current common interface exposes twelve physical actions and 83 observed
features, including actual executor phase and active/measured-reference differences.
The predecessor retains three real grasp choices. Standard discrete SAC with
success-behavior regularization learns from its own native episodes. At genuine
policy-over-Option boundaries we aggregate realized rewards and physical durations;
for the inherited finite-horizon objective, gamma is one. Hence the standard target
is the aggregate return plus the nonterminal soft successor value, rather than a
new budget-probability surrogate. Auxiliary legacy heads remain unused in deployment
and their optimization cost is reported.

Three temporal policy classes share the executor and action alphabet. The queue
class delegates continuation to a frozen reference until the inherited termination;
the phase class also reopens at semantic phase changes. The parameterized atomic
Option holds a chosen reference-relative offset, refreshes that same offset at
subsequent refinement boundaries, and continues the insertion queue. Grasp loss and
insertion-queue exhaustion release control. It uses neither future success nor extra
physical time. This is a standard parameterized closed-loop Option, with precedents
in temporal abstraction; identical action alphabets do not make these temporal
policy classes identical. We compare their learning behavior as representations,
not as evidence of an independently novel SAC algorithm.

### Causal evidence and falsified simplifications

On seen diagnostic condition 760213, complete native reexecutions share the first
76 recorded physical steps. Neither a single refreshed offset nor a sequential
refresh/local operation rescues the task. Repeated refreshed correction with phase
reopening succeeds; the queue-committed single intervention fails. The parameterized
atomic Option reproduces all 100 recorded physical steps of the original successful
rule. Independent contact instrumentation measures first nonzero peg/fixture force
at step87 versus98 for the successful phase and failed queue continuations. Contact
force is diagnostic evidence, not an additional policy observation.

The learned policy on condition761202 selects a local correction at step63, whereas
a failing queue learner intervenes at step59. Four full native reexecutions establish
a shared63-step prefix: learned execution succeeds, suppression of that intervention
fails, nominal continuation fails, and that intervention alone followed by continuation
succeeds. This supports an actually learned useful action, not merely prediction
improvement. Its deployment uses a local action, so it does not isolate atomic
refresh as the cause of the training benefit.

A conservative fixed-time-grid IK splice rejected all ten probe corrections (eight
acceleration checks, one lost grasp, one insufficient queue). Rejecting and keeping
the old queue preserves the known late success but establishes no new correction
capability. The acceleration check is a setpoint finite-difference feasibility test,
not measured motor acceleration: tracking lag already makes some original next
commands exceed that test. We therefore do not infer that all requested residuals
are physically impossible, or claim certified contact safety.

### Completed own-policy training and frozen confirmation

Each arm uses64 own-policy native episodes and1280 actor/critic iterations; the five
completed arms total320 training episodes. Two neural initializations are independent,
while shared nominal warmup uses16 real episodes. Primary final64 development results
are queue9/16 and7/16, macro8/16 and12/16, phase6/16 (single seed). Intermediate macro
checkpoints reach11/16 for both seeds. This shows useful learned behavior and seed/time
instability, not robust improvement.

Before new confirmation, all methods used the same declared maximum dev0/32/64 score
and earliest-tie selection. Both queue models select initialization; macro models select
32 and64. The full64-episode cost is charged to each. On24 new natural conditions,
queue scores12/24 for each seed, macro10/24 and12/24, nominal12/24, and the frozen
feedback rule15/24. Macro rescue/regression counts are1/3 and1/1; rule counts4/1.
One learned rescue (762214) is not covered by the frozen rule, but aggregate superiority
is absent. These conditions are development confirmation, not sealed test data.

The first macro model assigns positive mean-Q differences to harmful first interventions
on762201,762204 and762206. On762206, both independent models prefer their harmful
intervention to continuation in estimated Q (+0.201 and+0.174). A simple Q-filter is
therefore not a verified solution. Conversely, the second model's successful762213
intervention has a slightly negative estimated difference (-0.009), emphasizing that
actual action value and policy extraction must be studied together. These are
descriptive frozen-model diagnostics, not estimates of calibrated success probability.

The only non-grasp1 choice in frozen confirmation is grasp0 on762209 by the second
macro model. Joint, fixed-front/learned-rear, and learned-front/fixed-rear all succeed
on full native reexecution there; this does not establish a new synergistic handoff
benefit. Three grasp parameters are genuinely explored during training, but the
selected policies predominantly use the nominal predecessor.

Four new native PlugCharger conditions with two actual grasp pitches and nominal/feedback
servo complete16/16 tasks at the original200-step horizon. The prefix duration varies
with actual grasp, but this small cohort is saturated and supports reusable controller
capability only, not cross-task learning transfer.

All current experiments account for1002 complete native episodes,101800 physical
steps and8934 planning queries. This includes every repeated physical prefix, training,
evaluation and mechanism control; video rendering contributes zero physics. Named
models, final transactional resume, exact training sources and all raw trajectories
are archived. Per-arm tables and costs are generated in
`evidence/EXP-004/2026-11-05/atomic_handoff/RESULT_TABLES.md`.

The current paper can support an executable-reference/temporal-authority formulation,
its concrete learned implementation, causal rescue cases, and a reproducible account
of generalization failures. It cannot yet claim stable superiority over mature
control, independently novel SAC, cross-stage synergy, or learned second-task transfer.


## Rule-relative, continuation-bound skill adaptation (current cycle)

### Problem and executable policy class

A useful intervention is conditional on its subsequent execution. The present experiment fixes the strongest inherited geometric controller, the native Peg100 predicate, the learned three-grasp predecessor, and two previously trained successor proposals. At a real successor decision, a policy chooses either the frozen rule or one complete proposal Option. After the chosen Option terminates, control returns to the frozen rule for the remainder of the same native episode. Choosing zero continues the currently active queue; it does not restore the original plan. Atomic offsets maintain their selected parameter across refinement boundaries, whereas non-atomic proposals retain their inherited continuation contract. Actual release, physical duration and command origin are logged.

The actor sees 83 execution features plus one-hot identities of the rule and candidate actions (107 inputs). A shared 107–32–1 tanh network scores each of the two proposals; a fixed zero logit represents waiting. Proposals equal to the rule action are masked. The physical predecessor remains present and can select three grasps, but its weights are frozen in this cycle. Consequently these experiments do not establish newly learned cross-stage coordination.

### Measured evidence and policy update

For each training initial condition, we physically execute the frozen-rule baseline from the original reset. A separate branch executes the candidate policy from that same original reset. Before its first deviation, robot/object states and physical commands are compared by exact serialized-prefix hashes. No contact-state restoration is used. The terminal label is delta = G(candidate Option then rule) − G(rule), with the original binary success predicate. This is an initial-condition-matched experimental label, not privileged information supplied to the deployed actor.

Let H be the observed waiting/adoption decision sequence, and let p_theta(H) be its probability. The online candidate uses a clipped likelihood-ratio policy update with advantage delta for rescues, 2*delta for regressions, and zero otherwise. The ratio is the product over all decisions in H, evaluated stably as an exponentiated difference of summed log probabilities. Eight gradient steps follow each nonzero outcome; no extra success-value head is introduced. Gamma=1 is appropriate to the terminal, finite native-horizon objective. The regression weight changes the optimization criterion and is shared with the cost-sensitive supervision comparator.

The significance of the full sequence is explicit: waiting choices determine which later intervention state is reached. An entry-only update omits their score-function terms and is a mechanistic ablation. The temporal likelihood ratio, clipping and measured baseline are mature policy-gradient ingredients. Neither the algorithm nor its safety is established by naming this combination. SPIBB's support constraints and theorem do not apply to this neural gate.

A shared cost-sensitive supervised initialization is followed by either this real online policy update or ordinary supervision on that arm's own adopted Option outcomes. Both arms explore with the same bounded categorical mixture and use the same observations, frozen proposals and physical budget. Greedy deployment is evaluated separately from stochastic exploratory training. A rehearsal control retains the common paired data during supervised online updates to test whether any apparent RL benefit is merely avoidance of ordinary forgetting.

### Experimental design and limitations

The first common dataset contains 64 new natural training initial conditions and 128 complete native executions. Each initial condition contributes one randomly selected reachable disagreement, rather than being counted multiple times as independent evidence. It contains three rescues, seven regressions and 54 unchanged outcomes. All seven harmful examples have positive historical mean-Q differences, while only two of the three rescues do. This directly rejects using the old behavior Critic as a calibrated rule-relative Option-value oracle.

Each online arm uses 64 different natural initial conditions and its own sampled physical data. The 16 development conditions are disjoint from both training groups; once examined they are development data, not unseen tests. Fixed final64, common-initialization checks and fresh confirmation conditions are declared before confirmation execution. Shared paired physics and counterfactual baseline episodes are charged to every comparison arm, while unique executed costs are additionally reported. No sealed data are accessed.

A first 500-update shared initialization did not even adopt the three positive training examples. A training-only convergence check motivated 2000 shared updates for both arms, retaining the earlier run as a negative result. The resulting initial model already rescues two development failures. Any benefit present at this checkpoint is supervised initialization, not online RL. The final tables distinguish that foundation from subsequent policy improvement.

The one-deviation restriction intentionally resolves one source of continuation mismatch but restricts recovery capacity. The observation summarizes rather than exposes the full future queue. Effects can be rare, initial-condition-correlated and time-sensitive. Current success retention is empirical, not a safety guarantee; a second contact mechanism and stable multi-stage learning remain necessary for a broader paper claim.


### Completed rule-relative results

| Method | Dev /16 | Frozen confirmation /24 | Rescue / regression vs shared rule |
|---|---:|---:|---|
| Frozen rule, shared predecessor | 9 | 16 | 0 / 0 |
| Frozen rule, fixed grasp1 | 9 | 16 | 1 / 1 |
| Frozen complete macro 31701 | 7 | — | — |
| Frozen complete macro 31702 | 10 | — | — |
| Historical Q gate | 8 | — | — |
| Fixed first proposal1 | 11 | 16 | 4 / 4 |
| Shared supervised initialization | 11 / 11 | 19 / 18 | 3/0 ; 3/1 |
| Online supervision | 10 / 10 | 18 / 19 | 2/0 ; 3/0 |
| Supervision + shared-data rehearsal | 11 | 19 | 3 / 0 |
| Online sequence RL | 12 / 12 | 19 / 18 | 3/0 ; 3/1 |
| Entry-only RL ablation | 12 | 19 | 3 / 0 |

The24-scene frozen confirmation separates a useful learned interface from an unproven online algorithm increment. Both final RL models exactly match their respective supervised initializers in per-scene task success. Simple online supervision swaps the19/18 ordering to18/19, and retaining the common paired data reaches19/24. The sequence-credit and entry-only ablation have identical64 training and48 development trajectories: every nonzero update in that seed occurs at the first disagreement, making their gradients equivalent on the observed updates. The proposed distinguishing mechanism was therefore not exercised, rather than merely statistically underpowered.

Fixed first-proposal adoption achieves16/24 with four rescues and four regressions. Conditional adoption retains three of those rescues with zero or one regression. This supports a role for state-dependent selection, already obtained from supervised actual Option effects; it does not establish a new RL algorithm. The512 own-policy episodes,13000 warm optimizer steps,2336 online optimizer steps and all40 named weight checkpoints are preserved. Unique physical cost is1528 native episodes/152800 control steps/14644 planner queries, including all controls, both iterations, Tool experiments and the continuation repair.

Tool provides a second execution-semantics counterexample rather than a learned-transfer result. Its inherited early-contact path discarded the nominal continuation even when feedback declined intervention, and could apply a correction before the requested event. A corrected version preserves all eight nominal trajectories and three declined-intervention trajectories exactly for100 steps. Corrected early nominal/feedback/press performance is3/3/2 of8, with no rescue. The original0/8 feedback result is retained but excluded from fair algorithm comparison. New joint predecessor/successor training and independently confirmed online improvement remain open.

## Joint-continuation development and executed results (2026-11-06)

### Problem and executable policy class

The previous learned one-deviation policy is a useful strong reference: paired
physical supervision explains its confirmed advantage over the rule. We now
study a different, explicitly sequential question. A trainable predecessor
chooses one of three real grasp offsets; continuous physical execution yields
an observed handoff. A successor chooses either a chunk of strong feedback or a
complete frozen parameterized Option. Control can reopen after that Option's
actual release. Thus the predecessor's best choice can depend on the learned
successor, and the successor's value can depend on later decisions. A single
fixed-suffix pair classifier does not in general estimate this current-policy
value. No object reset occurs at the handoff.

A simple counterexample separates these claims: suppose two legal grasps reach
states with fixed-suffix success probabilities0.8/0.6. If an improved successor
changes those probabilities to0.8/0.9, the optimal predecessor reverses. These
numbers are an illustrative counterexample, **not measured experiment data**.
The existence of such a possibility is not evidence that our physical task or
trained policy realizes it; measured cross-stage comparisons are required.

### Shared training foundation and minimal probe

The existing paired-data scorer and pretrained grasp policy provide common
initial logits. A zero-output residual network learns the three categorical
choices at both boundaries. Two newly initialized critics learn actual complete
Option transitions. We do not transplant a Critic trained for a different
continuation contract. Discrete SAC, successful experience replay and bounded
categorical exploration are mature shared ingredients. Proposal models and
intra-Option execution are frozen, as is the strong controller. Rear choice0
means executing that controller, which may physically issue a nonzero action;
physical action0 continues the current queue.

For native finite-horizon gamma=1, the actual aggregated transition target is
R_option+(1-done)V(next). We retain measured duration and remaining budget;
no additional native steps are granted. The minimal comparison changes only
predecessor target estimation: half of the TD target is mixed with a recent
actual suffix return. This is a standard empirical multistep ingredient and
not claimed independently novel. Behavior-policy suffix returns may still
differ from current-policy returns; the experiment tests practical value
propagation, not a policy-consistency guarantee. Both arms collect own-policy
physical experience and share equal update and interaction budgets. Existing
one-shot initialization remains a strong separate learned reference.

STAP models downstream skill feasibility for planning; RecoveryChaining learns
recovery and handback; Policy Decorator protects/refines a base policy; SPIBB
constrains unsupported policy changes. Our executable joint interface draws on
these foundations. No new safe-improvement theorem, learned termination
algorithm or motor-level residual controller is asserted.


### Executed revisions and preliminary results (2026-11-06)

| Arm | Own episodes | Initial → middle → final |
|---|---:|---|
| identity_flat_32001 | 64 | 9 → 8 → 9 /16 |
| identity_semantic_32001 | 64 | 9 → 7 → 7 /16 |
| paired_pg_32101 | 32 | 9 → 9 → 9 /16 |
| paired_sac_32101 | 32 | 9 → 9 → 9 /16 |
| return_mix_31901 | 64 | 13 → 12 → 11 /16 |
| return_mix_31902 | 64 | 13 → 12 → 11 /16 |
| sac_31901 | 64 | 13 → 11 → 11 /16 |
| sac_31902 | 64 | 13 → 12 → 10 /16 |

The later execution-conditioned revision gives identical executed proposals equal scalar values while granting the flat comparator identical identity observations. Four native alias trajectories verify execution equivalence, but success falls from9/16 to7/16; numerical consistency does not imply improved policy. The third revision uses an action-independent root reference return to train full sampled joint trajectories with clipped episodic policy gradients. The matched SAC receives identical physical reference data. Both finish9/16, with no new deployment success. A training rescue is explained by either predecessor or successor change alone in a subsequent native factorial, so synergy is not established. These are executed negative controls, not new algorithm superiority.

Frozen historical policies on32 new natural scenes score19(rule),20(supervised initialization),21(prior online RL),20(supervised replay). Small paired gains do not establish robust online advantage. The first revision uses two new-network/exploration seeds with shared pretraining; later revisions one. Fixed-final model selection and cohort separation are retained. The run costs1163 full native episodes/116300steps/9857planner calls, including448 own-policy training episodes. All models, actual support, failures and costs are retained in [the evidence](../evidence/EXP-004/2026-11-06/joint_continuation/README.md).

The paper can now describe a genuinely sequential executable interface, current-policy physical collection, the limits of fixed-suffix credit, and controlled failed learning mechanisms. It cannot yet claim independent joint RL improvement, safe policy improvement, unrestricted action coverage, convergence, or second-task transfer. The next discriminating experiment concerns useful legal physical Options omitted by the frozen proposal catalogue.
