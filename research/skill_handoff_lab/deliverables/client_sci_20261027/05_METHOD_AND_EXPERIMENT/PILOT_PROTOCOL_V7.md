# 2026-10-27 v7 有界接触参数 pilot（先验冻结）

Scope: SYNTHETIC ARES JAKA / Isaac Sim 5.1 / Isaac Lab 2.3.0 PhysX; no field data. Primary metric **P2 independently verified complete success**, not P1 macro average. No physical state writes during actions.

## Hypotheses and fair controls

- B0: corrected task-local rear-face contact alignment and damped endpoint feedback; all original joint, reference-lead, force-proxy, posture and stable-completion limits retained.
- B1: BODY-XY residual PPO with the same corrected B0, observed 64-dimensional state and bounded third contact-parameter action.
- B2: task-frame residual and state representation; same network/actions/reward as B1. Execution-effect channels zeroed. This is an engineering/inductive-bias baseline, **not a novel algorithm**.
- B3: B2 with measured previous-step command/actual object velocity mismatch, angular response and actual contact-parameter history entering the Actor. The bounded third action alters the phase-0 lateral endpoint by at most 4 mm; full-chain physical reward trains it. No auxiliary prediction head or future success labels.
- B3 no-outcome: matched B2; B3 no-next: known downstream goal/context zeroed only during phase 0. Same checkpoint input interventions and separately trained no-next seed31 pilot test reliance. A parameter effect alone is not evidence that outcome conditioning helps.

All learned methods have the same 64-64 Actor/Critic and 3 Gaussian action outputs, ±0.6 mm residual/control and ±4 mm contact parameter, identical hard projection/IK/dynamics. Third action is active in phase0 for all learning controls. State includes simulator pose/velocity and known scene/dynamics context: **state-based simulation, not deployable multimodal perception**. B1 also has zero measured-effect channels so task frame is the intended B1/B2 difference. B3 differs by real effect history, not extra action capacity.

Post-implementation representation scope (not a physics/tuning change): task normalization sets directioncos/sin18/19 to1/0, while q/qd, world-orientedtool and workspace remain BODY. Therefore B1/B2 are NOT claimed exactly information-equivalent invertible representations or fullySE(2)-equivariant. They share underlying state sources, capacity and legal resources; mainB3/B2 are matched on the same normalized base. Geometryroundtrip tests require externallyknown n and do not prove an Actor inverse.

## Frozen budget and sequence

1 env two-episode B0 feasibility → 4 env eight-episode B0 → 16 env. Online first pilot: B1/B2/B3 each seeds31/41/51, **16,384 environment interactions**, rollout256, batch256,6epochs, gamma1, learningrate3e-4, logstd−1.5. Same reset seed per paired method. Three means are not 96 independent scene samples.

Each trained checkpoint receives a separate new validation evaluation:32 P2 cases, seed102741. Training source/config/asset SHA, real action history, episodes, q/object/velocity trace, gradient updates and checkpoint kept. The final checkpoint is fixed by budget, never best-validation selected. 16,384 is a pilot and may undertrain; zero success is retained, not discarded.

No broad tuning: only one bounded iteration may change physical interface after a diagnosed software/geometry error. If B3 fails validation promotion, do not grow networks, train5seeds or touch final tests.

Controllability probe: same32 validation cold-start cases, fixed −4mm/0/+4mm phase0 contact parameter and identical zero-parameter damped suffix. Full continuous robot/object motion, handoff snapshots and final outcomes saved. This is a three-execution physical intervention, not a free online search baseline; its total execution budget is three times B0. It must not be compared as an equal-budget oracle.

Attribution diagnostic (before seeing matched intervention results): one additional32case inference on B3seed31 removes only history0/1/4/5/6 (measured object finite-difference velocity, velocity mismatch, angular response), retaining applied-reference velocity and requested-parameter memory2/3/7. Separate diagnostic source snapshot, no training, no final tests. This avoids interpreting the all-history mask as purely an outcome intervention. Current56D pose/velocity/force remains shared and visible; this is not a policy without all feedback.

G3 promotion predeclared: B3−B2 P2 success positive in all3 paired seeds AND mean at least2/32; no-outcome/next intervention must alter executed parameters and identify a consistent result/cost contribution. Otherwise report PROTOTYPE/negative. Final Test-ID/Geometry/Dynamics not used for selection; frozen hashes below, evaluation only after decision.

## New scene isolation

New scene files `configs/client_sci_v7/`. Train8groups,Validation2,eachfinal2. No old v6b test cases used for tuning. Rails100mm,40mm free cube,30mm research paddle; direction, mass/friction, gap and goal varied. P2 alone main pilot; P1 optional base-control validation, not claimed shared policy learning.

Final SHA256 frozen before any pilot:

```
test_id.json       e2ba4ea99c7b4c5430a208f71c57aaa23a4cdecfa81236be4d0019b6a0a84a39
test_geometry.json c398b2d787f28931649e8d45bba485237f75a09f2dfc039667acb354dfe4433d
test_dynamics.json d0d93d77390371431612020eceb2e235b30a69bec1a61920ad5759edda23827c
```

Success: both stages real continuity, final error<8mm,speed<10mm/s,angularspeed<0.2rad/s,relative square yaw<8°,upright<5°,30 consecutive60Hz steps. Independent 120Hz sampled geometric/post-step audit separates raw/qualified results. Contact force is a simulated proxy, not hardware safety certification. Timeout, constraint fail, software error and UNKNOWN remain separate.

## Execution and provenance

Remote authorized root `/home/andyee/robot-research/ARES-Skilllab/runs/client_sci_pilot_20261027/`; one worker per GPU0/1; parent render GPU2. Per-child timeout≤900s, own process group only, `RUN_COMPLETE.json` and exception inspection. No driver/host changes. `engine/` isolated exact source; local models and raw traces nonGit; public only synthetic summaries/hashes.

## Scientific limit

Measured outcome concatenation, coordinate rotation and residual PPO are existing ideas, not standalone novelty. This bounded experiment tests whether they create a useful physical parameter adaptation effect. A null result does not erase previous residual geometry gains, nor become a new algorithm claim. Only actual downstream behavior and matched ablations can justify expansion.
