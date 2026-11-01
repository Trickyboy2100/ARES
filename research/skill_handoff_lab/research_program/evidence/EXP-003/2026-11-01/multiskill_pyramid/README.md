# Native placement-axis opportunity scout

**Final panel:**10 native250 episodes,2500 physical control steps. Initial8 are described below; two deterministic geometry-rule follow-ups are separately reported at the end.

Eight real PhysX episodes, two new native StackPyramid seeds741000/741001,250 steps each,2000 physical steps and zero optimizer updates. Task geometry, initial distribution, native success and horizon are unchanged. `program_multiskill_pyramid.py` subclasses the existing strong `HandoffExecutor` and retains its grasp/lift, actual attachment compensation, OPEN/RETREAT and three-consecutive-observation release/stability checks. No inter-skill reset or object state setter is used.

The new first parameter is support placement axis (0,±90,180 degrees); the second is top offset in the **measured normalized A-minus-B XY frame**, not global X. This first scout holds the latter at its strong nominal zero, recomputing the top target from actual released support positions. It therefore establishes first-action physics and real boundaries, not second-action benefit.

## Results and failure separation

| Policy | Native complete success | First release verified | Interpretation |
|---|---:|---:|---|
| Four-axis geometric search |2/2|2/2|Selected0 in both cases; full successor kinematic cost plus target occupancy test|
| Fixed axis0 |2/2|2/2|Exactly matches strong search physical outcomes|
| Fixed axis+90 |0/2|0/2|Precursor release target error36.1/16.9mm|
| Fixed axis−90 |0/2|0/2|Precursor release target error16.4/15.9mm|

All four failures had open fingers, no grasp, static objects and static robot; target XY error exceeded the existing15mm threshold. These are not downstream-policy failures. The failures displaced untouched support B by16.0–23.4mm, compared with1.9–3.7mm for axis0. Seed741000 +90 also moved C22.7mm and was correctly rejected by the search's initial occupied-target test; it was executed only as a declared intervention. The other three failed axis choices were kinematically valid predictions but failed actual release. This demonstrates the limit of kinematic feasibility as a physical completion label, without demonstrating an RL gain.

The scalar native A/B distance margin was still positive15.5mm for the +90 failure on741000, so a single support-distance number does not replace a complete skill contract. The other three failures had negative margins−8.0/−1.5/−3.9mm. Actual contact partner forces were not recorded; displacement is observed, its precise contact attribution is not proven.

Only4/8 episodes reached the second true decision; no failed first skill is silently relabeled a successful handoff. Search CPU wall time is recorded separately in RESULTS/CSV, approximately0.14–0.21s for four candidates, in addition to execution-planning wall time. Predicted motion steps exclude release/settling overhead and are not contact-success predictions.

## Research decision

On these two scene draws, strong search/axis0 dominates; do not train a new RL policy on this tiny saturated comparison. An informative bounded follow-up is axis180 on both states (currently unexecuted). Only if it produces a verified but difficult support configuration should a paired top-offset test be added. Repeating offsets on already robust axis0 is low information. No independent two-decision learning advantage has been established.

## Reproduction and source

- `RAW_EVIDENCE.tar.gz`: every complete250-step trajectory, initial full simulator state, planning paths, decisions, physical release measurements, exact per-run source snapshots and manifest. `RAW_EVIDENCE_MANIFEST.json` gives every SHA256.
- `EPISODES.json/csv`, `QA.json`, `SUMMARY.json`: CPU-generated source-backed metrics; same-seed initial hashes match and all terminal/native trace checks pass.
- `SUPPORT_AND_OUTCOME.png/svg`: measured first-release support margin and full task outcome.
- Native source: pinned ManiSkill `mani_skill/envs/tasks/tabletop/stack_pyramid.py`; hash/revision/license in each manifest.

Example offline invocation from repository root (runtime path is local):

```bash
bash research/skill_handoff_lab/tools/handoff_offline.sh \
  /home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python \
  research/skill_handoff_lab/tools/program_multiskill_pyramid.py \
  --seed 741000 --axis 90 --label axis90 --out /tmp/pyramid-new-output
```

This reproduces a simulated physical intervention, never a real-robot command. No data here are field/ARES-R assets.

## Additional geometry falsification

`GEOMETRY_AUDIT.json` reconstructs predicted placed-A orientation from the actual held attachment and commanded TCP orientation. The original controller assumes20mm A projection; actual projection is20.08–20.34mm. Commanded A/B separation remains positive2.66–2.92mm, both after LIFT and just before PLACE. Thus the hypothesized missing A-yaw footprint correction is submillimeter here and does not explain16–36mm placement error.

The support B stays still through PREPLACE, then moves7.8–10.9mm during PLACE and15.6–23.0mm by OPEN for the failed ±90 layouts; axis0 only moves1.0–2.8mm/1.9–3.7mm respectively. The fixed world-Y gripper opening direction points toward B for ±90 layouts, suggesting an omitted gripper-clearance condition. This is a controller-geometry hypothesis, not a measured contact-partner conclusion. It should be tested with a shared geometric orientation/clearance rule before using these failures as evidence for RL. No such repair has yet been physically evaluated.

A further CPU check uses the actual PandaWristCamV3 URDF/FCL, recorded arm q at endPLACE, intended fully open finger joints0.04m each, and B's recorded pose beforePLACE. All four failed ±90 layouts produce finger/B overlap (+90:rightfinger,−90:leftfinger); all four successful axis0/search runs produce none. `INTENDED_OPEN_GEOMETRY.json` preserves inputs, model hashes and exact pairs. This is **static intended-open geometry**, not an additional simulated rollout, measured force, or counterfactual success label. It identifies a testable gripper-clearance defect that a geometric controller should handle before RL is credited.

## Deterministic release-direction repair: two actual paired follow-ups

The optional shared rule places with gripper opening perpendicular to the support axis and computes nonzero-axis A footprint from the actual attachment. Top placement uses the measured released support axis. Axis0 retains the exact legacy branch. Four-axis kinematic search uses the updated orientation when this mode is enabled. No native limits,250-step horizon or completion checks change.

- Seed741000,axis−90: former precursor failure becomes full native success. Verified releases occur at78/149. B displacement falls15.9707mm→0.000555mm; static intended-open finger/B overlap changes1→0. The same recorded initial state and every action/state through A_LIFT step39 match exactly.
- Seed741001,axis+90: the new orientation is kinematically infeasible at A_PREPLACE; remains failure. Initial and prefix through step42 match exactly. The updated search also predicts that candidate invalid. No physical success is invented, and the complete250-step terminal trajectory is retained.

This is one deterministic controller repair rescue out of two targeted attempts, **not an RL gain** and not a universal repair. It removes a concrete weak-controller confound before method research. `REPAIR_PREFIX_AUDIT.json`, updated raw package and `INTENDED_OPEN_GEOMETRY.json` retain all evidence. An improved baseline must jointly consider finger clearance and reachable orientation (including symmetric gripper orientation alternatives), before these failures can become a fair learning benchmark.
