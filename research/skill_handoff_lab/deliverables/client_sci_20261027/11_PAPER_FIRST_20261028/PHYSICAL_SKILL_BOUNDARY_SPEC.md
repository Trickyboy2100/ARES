# Common physical executor contract

Public pinned ManiSkill StackPyramid-v1, native reward, success predicate, physics and250-control-step horizon. Public Panda MPLib URDF/SRDF. No object setters, teleportation or resets between skills. Independent environment servo programs are interleaved on the single RTX3080.

1. Decision0 from initial geometry/proprioception: A placement-direction offset in {-90,0,+90} degrees relative to initial A-B direction. Grasp/lift A, move actual object to adjacent support pose, open fingers, retreat60mm, then verify.
2. Boundary0: three consecutive control readbacks with finger opening>65mm, A not grasped, A/B/C linear speed<10mm/s and angular speed<0.5rad/s, robot speed<0.15rad/s, A XY target error<15mm, A within8mm of table height and actual A-table contact force>0.0001N. Only then is decision1 requested.
3. Decision1 from real boundary state: C grasp-yaw offset in {-0.35,0,+0.35}rad, around the observed OBB face. Physically grasp/lift C, retarget using measured object-to-TCP offset, place over actual A/B midpoint, open, retreat60mm.
4. Boundary1: same stability/opening checks, target height error<8mm and measured C-A or C-B contact>0.0001N. Both-force and projected-overlap diagnostics are recorded; projected overlap is not measured contact area. Native success is independently evaluated at step250. Native success and verified-program completion are reported separately.

A screw plan has two attempts and a bounded0.5s RRTConnect fallback. Symmetry-equivalent gripper closing direction minimizes unnecessary wrist rotation. The common controller correction is not an RL gain. Planner geometry does not include scene cubes as avoidance obstacles; actual contact and failure are determined in native PhysX. No claim of production collision certification.

Candidate-plan cache: reuse only when seed, label, physically selected action prefix match and the entire per-environment measured object poses/velocities plus robot q/qvel and target pose differ by<2e-6 max absolute. Cached objects are joint plans, never states/outcomes. Actual PhysX servo always runs. Initial scene SHA and prefix state differences are audited. Cache costs and solve time are counted.

All batches physically execute all250 steps, including absorbing terminal holds. Logical completion is diagnostic only, not saved execution time. Native horizon failure remains failure; no extra251st action. gamma=1 finite-horizon terminal native success trains both physically visited decisions. No synthetic second token after failed A.

No-second ablation commits both parameter factors from the initial observation, then still executes and verifies both physical skills. No-posterior replaces object geometry/effect features with the initial snapshot, preserving live proprioception/time. No-next removes all C geometry channels at the first decision. Privileged state is explicit; merely deleting redundant effect deltas would not isolate posterior information.
