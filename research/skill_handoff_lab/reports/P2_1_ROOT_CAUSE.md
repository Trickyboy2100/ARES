# P2.1 root-cause report

Date: 2026-10-13. Branch: `codex/ares-skill-handoff-p2-1-20261013`.
No robot command was issued. ARES-R commit `9de17d2` was read-only.

## What the original 6x4 result did and did not establish

The preserved P1/P2 V4 run had 24/24 legal center IK endpoints, 0/12 Groove
successor endpoints and 12/12 Cabinet endpoints. It did **not** contain a
qualified predecessor state, attached-object check, scene-revision check, full
trajectory, cuRobo evidence, contact physics, or execution evidence.

P2.1 rechecked the exact same 24 joint states without regenerating them. The
result remains 0/12 Groove and 12/12 Cabinet. Thus adding explicit scene and
attached-object revisions did not reverse the frozen result. The exact rows are
`results/p2_1_cpu_v3/v4_exact_6x4_recheck.jsonl`.

## Root causes

1. **The four original candidates were mostly IK solutions, not different
   physical terminal intents.** `bench/generate.py::_candidates` fixed one roll
   branch and one center yaw per family. P2.1 replaces this with a fixed
   factorial support: NORMAL/FLIPPED_180, entrance yaw {-30,0,+30} degrees, and
   at most two IK solutions. Failed cells remain failed and are not replenished.
2. **Groove V4 is geometrically self-inconsistent.** Every old Groove
   successor collides with `groove/left_wall`; after adding the held tray, both
   `right/gripper` and `attached/held_tray` collide. The nominal wall spacing is
   0.140 m center-to-center with 0.026 m walls (0.114 m inner gap), smaller than
   the 0.120 m research tray-proxy width before orientation projection. The 0/12
   is therefore a common target/fixture collision, not candidate sensitivity.
3. **Cabinet V4 is too easy at the endpoint.** All four legal candidates in
   each frozen Cabinet scene reach a clear direct-preplace endpoint. Their proxy
   full-chain costs vary strongly (e.g. 8.63--21.85 in scene 4201), but no legal
   candidate changes the Boolean endpoint outcome.
4. **The first P2.1 v1 chassis gate was invalid.** ARES-R marks
   `body/chassis` as a filter-owned conservative proxy pending measured CAD. It
   overlaps normal commissioned link1/link2 states. P2.1 now reports chassis
   collision as `NOT_TESTED`, rather than using that proxy to reject candidates.
5. **The first P2.1 predecessor was invented and invalid.** P2.1 now uses the
   ARES-R `folded` right-arm joint state, commissioned while holding a payload,
   as the common predecessor. It never uses an all-zero joint vector.
6. **The initial P2.1 v2 mixed two successor contracts.** Current runtime
   evidence uses direct `move_preplace`; the experimental TransferChainPlanner
   includes a rear-preplace waypoint. The synthetic rear target was unreachable
   and incorrectly killed 46 legal candidates. V3 gates on canonical direct
   preplace and marks rear-preplace `NOT_QUALIFIED`.
7. **The CPU interpolation proxy is uniformly negative.** 0/52 legal center
   candidates have a collision-free straight joint interpolation to preplace.
   This says the straight line is not a valid trajectory; it is not a cuRobo
   failure and not proof that a planned path does not exist.

## V3 bounded results

- 10 fixed scenes, 120 fixed factorial slots, 52 legal center terminals.
- 16/120 downstream geometric endpoint members.
- Frozen scenes: Groove 0/20 among legal candidates; Cabinet 12/12.
- Predeclared mechanism scenes: narrow/wide Groove apertures 0/16 among legal;
  Cabinet apertures 4/4.
- **Within each scene, legal candidates never split on endpoint membership.**
- For every scene with at least one endpoint member, minimizing the existing
  predecessor-to-center `_edge_cost` also minimized the CPU full-chain proxy;
  observed heuristic regret was 0 in all five such scenes.

The run's coarse `endpoint_sensitive` field compares all fixed slots and is true
when legal and illegal slots coexist. The paper-relevant quantity is
`within_legal_endpoint_sensitive` in `P2_1_SCENE_SUMMARY.csv`; it is false for
all ten scenes.

## Evidence boundary

Tested: pinned URDF FK/IK, site soft limits, explicit grasp/yaw/IK factors,
scene cuboid OBBs, parked left-arm OBBs, attached cuboid proxy, central TCP
margin, revision lineage, and deterministic CPU joint-line descriptors.

Unknown/not tested: measured chassis CAD collision, right-arm self-collision,
current live TCP rather than the pinned 2026-09-21 audit, persistent cuRobo
paths, contact/hold physics, execution time, and real execution. The held tray
is a revisioned cuboid research proxy, not a verified mesh/contact model.

## Conclusion

The original 0/6 scene-sensitivity result is explained by one universally
blocked Groove model and one universally feasible Cabinet endpoint. P2.1 does
show physically distinct terminal choices and 3--4x cost spread, but it still
does not show two legal terminals in the same scene with different downstream
feasibility. That scientific precondition remains unestablished.

