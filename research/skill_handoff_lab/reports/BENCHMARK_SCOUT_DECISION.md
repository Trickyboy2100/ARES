# Benchmark pivot: measured decision, 2026-10-27

## Scope and preservation

Fetched `codex/ares-sci-autonomous-research-visual-20261027`; starting HEAD
`13211fee5cc9448e59902b2f8673707bfd959aed`. The checkout was clean before this
iteration. No previous P1/P2, WideCabinet, checkpoint or field evidence was
overwritten. The new work order was read in full. ARES-R remains read-only.
All new experiments are under the non-Git `runs/benchmark_scout_20261027/`.
No field data, field geometry or robot-controller connection was used.

This is **a real benchmark pivot and baseline iteration**, not a demonstrated
new algorithm, completed multi-seed study or finished JAKA manipulation transfer.
Completed counts and checkpoint hashes are in the deliverable's generated index.

## Actual scouts

| Candidate | Actual physical execution | Strong baseline / issue | Decision |
|---|---|---|---|
| Isaac Lab Lift, Franka | 1/4 env, feedback lift; single cube z rose from 0.021 to 0.268 m | Native configuration has no success termination; do not invent a score | JAKA migration reference, not main algorithm test |
| Isaac Lab Stack, Franka | 1/4 env reset/step/video | Native configuration has `rewards=None`; not a ready dense-reward PPO comparison | Preserve native scout; avoid silently changing reward and calling it official |
| Isaac Lab Drawer, Franka | 1/4 env; official state machine physically opens four drawers about 0.306 m | Native config has timeout, not a common success predicate | Useful constrained-contact application reference |
| ManiSkill PickCube | Native random, motion planning, 256k official PPO, 16-case policy evaluation | PPO 0/16 at 256k; planner grasps but exceeds 50-step goal budget | Training/control sanity only |
| ManiSkill StackCube | Native random, planner, feedback, official PPO and matched-interface PPO pilots | Feedback 58/64 within 50 steps; failures include release, placement, stability | Selected first method-development task: cheap closed loops and measurable failure |
| ManiSkill PegInsertionSide | Native random and planner; GPU observation adapter repaired | Corrected planner finishes 4/4, but only 1/4 success-ever within native 100-step horizon | Second contact-condition task; budget-aware alignment/recovery candidate |

The final validation plot is generated from completed manifests, rather than
manually entered numbers. Development seeds are 27101 + 4 * batch_index with
four parallel native environments. These are **not a sealed final test**.

## Decisive engineering diagnosis: GPU planner geometry

The pinned official planner obtains an OBB from `actor._objs[0]`, a CPU component.
Under the selected GPU simulation backend it repeatedly reports cube centre
`[0,0,0.1]`, while the GPU actor pose is the actual randomized cube position.
This made the original Stack planner grasp empty space. The adapter now obtains
the local collision OBB and transforms it with the current GPU actor pose.
Only planner observation conversion changes: no actor pose, dynamics, collision,
success rule or reward is rewritten. Original failures remain saved.

Four paired Stack diagnostic cases changed from 0/4 to 4/4 after the fix.
However, corrected executions take 88-103 control steps, exceeding native 50.
These are **out-of-horizon planner diagnostics**, not official success scores.
Peg completes in 108-161 steps; its first case succeeds before step 100 but the
others do not. This is a budget/interface issue, not proof that RL is necessary.
The fix is not a proposed algorithmic contribution.

## Matched training experiment

Pinned ManiSkill source: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`.
Official PPO source SHA-256:
`3fa4818861d428480244aae889bc679b2c8e89b59d6289014003c266a84dbab0`.
Isolated runtime: `envs/maniskill-scout-20261027`; Isaac environment unchanged.

1. Upstream PPO unchanged: Pick 256k; Stack 256k; Stack 1.024M from scratch.
2. Matched Cartesian PPO and ordinary residual PPO: 256k each, seed 31,
   same 64 training envs, 50-step rollouts, observation and network capacity.
3. Same 64-case evaluation plus residual forced to zero; exact actions before
   and after clipping, controller phases and actual physical states are saved.

Both Cartesian variants observe native state, controller phase and nominal
command. The native physics/reward/horizon/success predicate is unchanged.
The residual executes `clip(nominal + 0.15 * actor, -1, 1)`; direct PPO executes
`clip(actor, -1, 1)`. These are standard control baselines, not a new method.
Current native object poses are state-based simulation observations; there is
no claim of vision or real-world observability already being validated.

The original joint-delta PPO is an upstream reproducibility reference, **not a
fair causal ablation against Cartesian residual control**. Its action interface
differs. Every training evaluation, including regressions, is retained.
Four-env training evaluation is too small to establish stable improvement.

### Completed same-case result

| Controller | Online training interactions | Final success |
|---|---:|---:|
| Structured feedback | 0 | 58/64 |
| Matched Cartesian PPO seed31 | 256,000 | 0/64 |
| Ordinary residual PPO seed31 | 256,000 | 61/64 |
| Ordinary residual PPO seed37 | 256,000 | 62/64 |
| Ordinary residual PPO seed43 | 256,000 | 60/64 |
| Seed31 checkpoint, residual forced zero | no retraining | 58/64 |
| Upstream joint PPO seed31 | 1,024,000 | 0/64 |

The three residual runs average 95.3125%, sample standard deviation 1.5625
percentage points, versus feedback 90.625%. Each fixes the same five feedback
failures but introduces respectively two, one and three new failures. Seed31's
exact paired test is p=0.453125: a small positive difference, not statistical
superiority. Reusing the same 64 cases across seeds does not create 192 independent
scenes. All initial state dictionaries match exactly; zero-residual actions and
all outcomes match feedback exactly. The learned XYZ command differs by mean
normalized norm 0.1481, 0.1555 and 0.1445 for seeds31/37/43.

Seven final checkpoints and **2,560,000 online training interactions** were
completed in this iteration, plus separately logged scout/evaluation execution.
No checkpoint was selected by validation score; all evaluations use the final
fixed-budget checkpoint. This is ordinary residual RL evidence, not the requested
independent novel mechanism. A fair new-method ablation remains unfinished.

## Selection scores (qualitative, evidence-based; not performance claims)

| Criterion | Stack | Peg | Isaac Drawer |
|---|---|---|---|
| Legal parameter / recovery decisions | Medium-high: release, recontact, placement | High: grasp offset, standoff, alignment, insertion | Medium: contact and constrained pull |
| Standard task comparability | High, native success and reward | High, native success and reward | Medium, this configuration lacks success termination |
| Non-saturated but solvable | Verified 58/64 feedback | Verified solvability; native-budget baseline limited | Opening verified; not a success-rate study |
| Strong baseline readiness | High | Medium: planner observation fixed | High official state machine |
| JAKA transfer cost | Medium-high: physical EG2 grasp required | High: accurate grasp/alignment/contact | High: tool/handle reach and contact |
| Measured iteration speed | About 0.7-1.4k training interactions/s during concurrent pilots | Planning/video tested; PPO not yet run | Fast state-machine execution; no RL pilot |
| Skill/contract interface | Clear grasp/lift/place/release | Clear grasp/prepare/align/insert | Clear reach/grasp/pull/release |

**Public main platform: ManiSkill3**, initially Stack with Peg as the second
physical task family. **JAKA application platform: existing Isaac Lab 2.3.0**,
initial grasp/lift/place family, then alignment/insertion after gripper readiness.
Do not install Meta-World/RoboCasa/RoboTwin merely to increase task counts.
The scope may move from Stack to Peg if a sufficiently tuned Stack controller
saturates the held-out conditions. Do not weaken the controller to retain Stack.

## Method hypothesis, not established contribution

Learn legal, object-relative skill parameters or bounded recovery choices using
current execution results (grasp retained, alignment, release and settling) and
remaining budget. The policy must alter actual next physical states. A predictor
loss alone is not evidence. Ordinary residual PPO is the minimum baseline.

The next discriminating experiment is a same-observation, same-action-range
comparison between ordinary residual control and execution-result-conditioned
parameter decisions, with zero-residual and information-removal ablations.
Use matched training seeds and separately frozen ID/geometry/dynamics groups.
Extra suffix or recovery interactions count toward the training budget.
STAP/RAPS/ViSkill/RecoveryChaining already cover relevant broad ideas; neither
parameterized skills nor downstream value is claimed as a first invention.

## JAKA status and safety boundary

Public JAKA + EG2-4C2 URDF and original STL collisions were used for an isolated
import/drive smoke. Digit-leading link and mesh names are illegal USD identifiers;
the import copy renames them and hashes byte-identical copied meshes. Joint
origins, limits, mimic ratios, inertia and mesh geometry are retained. ROS-only
extensions are removed from the import copy, not production assets.

Initial imports failed; the relative-path conversion stalled and was stopped
after its 420-second budget. Absolute paths, a local primitive floor and float
initial joint values then permitted **720 genuine physical steps** and a video.
Independent audit nevertheless found EG2 Joint3 limit excess about 0.200 rad,
large mimic deviations and transient velocity excess. This is an execution
qualification failure, not a reason to remove collisions or widen limits.
**No successful JAKA grasp/Stack/insertion claim is supported by this run.**
Do not use the old analytic gripper demo, static USD or Panda video as substitute.
The dual-arm/mobile platform is retained; it is not silently replaced by Panda
in the application track. Single-arm importer readiness is narrower than a full
dual-arm installation, collision or industrial safety validation.

## Reproduction, resources and licensing

Sources, commands, hashes, errors, checkpoints and full state/action records are
saved per run. GPU workers were bounded; no driver/reboot/system modifications,
shared-container replacement or unknown-file deletion was performed.
The temporary network proxy is loopback-only and used only for this installation.
All large raw data/models remain outside Git. No asset re-distribution.

Primary references: [ManiSkill repository](https://github.com/mani-skill/ManiSkill),
[official PPO baselines](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html),
[Isaac Lab 2.3 tasks](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html).
ManiSkill code is Apache-2.0; its upstream asset CC BY-NC notice needs separate
review for client/commercial use. This report does not grant commercial rights.

## Publication judgment

This establishes runnable public tasks, meaningful strong controls, actual
online RL, causal residual evaluation and a concrete engineering failure diagnosis.
It does **not** yet establish a novel method, repeatable multi-seed gain, two-task
generalization or JAKA application success. The scientific claim must remain
smaller than the software infrastructure. The previous P1/P2 evidence is historical,
not silently relabelled as official benchmark evidence.
