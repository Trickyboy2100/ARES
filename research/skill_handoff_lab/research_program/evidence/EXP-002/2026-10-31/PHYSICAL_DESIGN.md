# EXP-002 physical design: source-backed recommendations

2026-10-31. CPU-only source review; **no new physics or RL result in this note**.
All parameter ranges below are proposed experimental ranges, not empirically
validated industrial distributions. Native PlugCharger scores and altered-physics
ARES-AdaptivePlug scores must remain separate.

## What the pinned implementation permits

Pinned ManiSkill commit: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`.
Source root: `external/benchmark_scout_20261027/ManiSkill` under the workspace.

* `mani_skill/envs/tasks/tabletop/plug_charger.py`, class constants and
  `_build_charger`: base half size 20/15/12 mm; two prong half sizes
  8/0.75/3.2 mm, offset ±7 mm. The three collision primitives use default
  density. Their summed volume at 1000 kg/m³ implies **29.1072 g**, a geometric
  calculation to verify by querying the built PhysX body's actual mass.
* Same source `_build_receptacle`: kinematic socket, clearance 0.5 mm per side.
  `_initialize_episode` samples real task variation, including charger yaw
  ±60°, socket yaw 180°±22.5°, and randomized planar positions. The goal is
  defined from the actual receptacle pose. Original horizon is 200 steps.
* `mani_skill/utils/structs/types.py:DefaultMaterialsConfig`: static and dynamic
  friction default to **0.3**, restitution 0. `envs/sapien_env.py` calls
  `physx.set_default_material` from that config. The charger/socket code supplies
  no explicit collision material. Render roughness is not contact friction.
* Installed SAPIEN `wrapper/actor_builder.py:add_box_collision` accepts
  `material` and `density=1000`; it stores them in collision records.
  ManiSkill `utils/building/actor_builder.py:build_physx_component` creates the
  collision shape, calls `shape.set_density`, and attaches it to the body.
  Thus set material/density **during construction**, before GPU initialization.
  Do not assume a post-reset Python body mass assignment updates GPU dynamics.
* SAPIEN builder `set_mass_and_inertia(mass, cmass_local_pose, inertia)` disables
  automatic inertia. Prefer uniform collision-density scaling so mass and
  inertia change coherently; setting only mass is a confounded experiment.
* `agents/robots/panda/panda.py`: arm PD stiffness 1000, damping 100, force limit
  100; gripper friction 2 and gripper PD 1000/100/100. Controller
  `agents/controllers/pd_joint_pos.py:set_drive_property` applies properties
  through `joint.set_drive_properties`. Any lower actuator strength is an
  explicitly altered task, not a native score or an RL algorithm advantage.

## Recommended minimal branching experiment

1. Fresh native initial states with the unchanged strongest controller; report
   prefix grasp/lift, post-grasp IK, alignment/contact and budget failures.
   Grasp angle changes must be screened against **both** pick and downstream
   pre-insert IK; a grasp-only success is not a rescue. Existing four inspected
   angle diagnostics all failed full task, so they do not establish authority.
2. For an altered-physics development family, first use charger density
   multipliers `{0.5, 1, 2}` and socket static/dynamic friction
   `{0.2, 0.5, 1.0}`, plus an untouched-native-material control (`None`,
   native friction 0.3). Keep charger/gripper contact material unchanged initially:
   changing both would confound grasp retention with insertion friction.
   These are moderate low/nominal/high probes, not a promise they are difficult.
   Build a fresh environment for each physical setting and log actual body
   mass/inertia and each collider's material. Never teleport after grasp.
3. If strong geometric control solves these settings, use naturally varied
   native grasp attachments and approach orientations first. Avoid inventing
   extreme friction or weakening actuation just to produce failures.
4. A bounded finite-observation extension can separately test delayed object
   pose (0/1/2 control steps = 0/50/100 ms). All competing policies must see the
   same delayed pose; include a constant-velocity predictor and delay-aware
   MPC baseline. This is observation adaptation, not contact identification.
   Do not mix it into friction experiments before independent attribution.

## Same-authority strong adaptation baseline

The current `tools/program_charger_feedback.py` has actual-tool feedback,
actual attachment refresh, bounded integral correction (±8 mm), ≤4 mm TCP
translation per command, ≤0.04 rad rotation and ≤0.05 rad per-joint change.
Keep these limits and its depth-hold rule. It already updates attachment each
step; a new policy cannot claim that operation as its contribution.

A useful strong competitor estimates a local response matrix online, using
task-frame **measured** tool increments as output and **actually applied**
bounded command increments as input. Fit lateral y/z and axial x response
separately across free-space/contact mode. Use regularized RLS, forgetful
updates, and revert to geometric control when excitation is insufficient or
prediction uncertainty is large. Solve a small constrained one-step correction
with the same step limits and remaining native budget. Include all probing
steps and computation in the episode budget. A deterministic two-gain PI sweep
and short-horizon adaptive model controller should precede an RL claim.

Do not fit a map from requested residual to effect when IK clipping or joint
projection changed the action. Log request → Cartesian clip → IK solution →
joint clip → actual TCP/tool response; report projection frequency. Mask rank
deficient response directions rather than treating an unexcited estimate as a
learned physical parameter.

## Observability and mechanism test

Current logs expose full q/qvel, TCP and tool pose, tool velocity, true grasp
state and scalar charger/socket force. Tool pose relative to TCP is already
computable from current state; concatenating that quantity or velocity is not
new posterior information. Current physics mass/friction can remain hidden
from *all* policies, but must remain available to offline diagnostic analysis.
The scalar contact-force norm is not a contact point, force vector or wrench.

For response identifiability compare three held-out predictors on the same
logged transitions: current state+command, same state+short effect history, and
state+true physics parameters (diagnostic oracle only). Separate train/test by
scene and physics setting, not adjacent timesteps. A history benefit must
persist after including current velocities, attachment and remaining budget.
Establish parameter ranking reversals and complete-task rescues/regressions
on paired physical executions before committing to expensive RL training.

If action-effect history helps only because command delay was omitted from a
state-only policy, report that limitation explicitly. If adaptive model control
already solves the new family, retain the useful control result but seek a
different physical mechanism instead of declaring an RL innovation.

## Historical evidence used

`evidence/EXP-001/confirmation-v2-2026-10-31/README.md`: strongest suffix 7/7
among reached cases; one shared prefix failure. Four additional grasp-angle
diagnostics yielded zero complete-task rescues; one change reached grasp then
failed suffix IK. Existing depth-hold improvement is deterministic control.
