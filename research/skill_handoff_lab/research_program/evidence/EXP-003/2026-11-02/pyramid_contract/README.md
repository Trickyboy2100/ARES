# Cumulative physical research: successor orientation and strong geometry

Open [REVIEW.html](REVIEW.html) for the result matrix and full synchronized videos. **UNVERIFIED_INDEPENDENT_ALGORITHM_GAIN**. This round adds90 real PhysX episodes/22,500 control steps,0 optimizer updates. No sealed tests, hardware commands or restricted assets. Prior models, controllers and negative results are preserved.

## Question and competing mechanisms

Can actual support placement change which top-placement parameters succeed under different requested top orientations? Competing explanations: (1) contact/support geometry creates a learnable predecessor–successor interaction; (2) apparent differences are ordinary orientation/kinematic selection failures that strong planning can resolve.

Reuse the existing native StackPyramid support/grasp/release/stability controller. New training-design seeds744000/744001, unchanged initial distribution, geometry and250-step horizon. Two actual parameter decisions: support-gap correction−6/0/+6mm; after measured released/stable SkillResult, top offset−8/0/+8mm along measured support axis. Successor contract requests object-world yaw0° or45° modulo90, within10°. Original native success and additional contract success are reported separately. This is **ARES-PyramidOrientation-adaptation**, not an unchanged official benchmark. Desired TCP orientation compensates the actual measured object-to-gripper transform; no objects are reset between skills.

## Actual result

| Controller | Physical scenes | Episodes | Native success | Additional contract success |
|---|---:|---:|---:|---:|
| Single representative orientation, diagnostic pilot |1|18|0/18|0/18|
| Nearest symmetry-equivalent orientation |2|36|29/36|29/36|
| Four equivalent preplace/place path checks, choose shortest valid |2|36|36/36|36/36|

All25 failures are C_PREPLACE planning failures; all90 predecessor boundaries pass. No failure in these runs establishes contact-adaptation opportunity. The last two controllers are strictly paired:36/36 identical initial state and every recorded support-prefix transition;7 rescues,0 regressions. These are **controller improvements**, not learned gains. The pilot uses only the first scene, so0/18→36/36 is not a paired rate comparison.

Nearest symmetry:3 conditions have9/9 successes,1 has2/9; its panel-best fixed and measured oracle both cover4/4 conditions. Strong feasible symmetry:4 conditions all9/9, every fixed pair and oracle4/4. Strict contract-dependent pairwise ranking reversals:0. Four conditions are2 physical initial scenes×2 contracts, not4 independent scenes; no confidence/generalization claim. Training a new value head here would again target saturated labels.

The four checks are ordinary geometric planning, use no future physical labels, and retain both stage parameters. There are144 orientation candidate queries in the strong36 episodes; each checks preplace and, if possible, place. Total episode wall time:pilot104.32s, nearest195.79s, feasible194.60s. These include execution/search and exclude environment construction; execution-only planning fields exclude candidate search and cannot be called total planning cost. No speedup claim. Native horizon remains250 steps, including terminal hold. The inherited stride3 path executor is disclosed and not native timing or hardware dynamic certification. Planner world-cloud collision checking is absent; actual PhysX outcomes do not certify swept collision safety.

## Raw evidence rechecked before experiments

- Newly added `audit_skillresult_opportunity.py` passes. Independently reread all96 original tool traces and recompute terminal native-plus-target success:49/96, exactly matching CSV. Their16 contexts remain7 all-pass,7 all-fail,2 mixed, fixed8/16 vs measured oracle9/16.
- Reread all36 original vector Pyramid traces:576 episodes,505 terminal successes, matching summaries. Re-executed historical opportunity analysis:best fixed59/64 vs oracle62/64;33 predecessor failures,36 successor incomplete,2 both-boundaries-valid/native-fail.
- Historical26 trace-index SHA entries also match. Existing PPO/SAC/STAP kernel/log-Q corrections and negative auxiliary experiments remain unchanged; no new RL or STAP training claimed.

## Cumulative decision

**RETAINED:** actual SkillResult release/stability boundary; source/trajectory SHA verification; same-prefix pairing; full250-step evaluation; cube-symmetry-aware bounded feasible-pose selection as a tested controller component, limited to these conditions.

**LOCAL_NEGATIVE:** these two successor yaw contracts and nine parameter pairs do not leave a success-rate opportunity after strong geometry. Numerical outcome variation under a weaker pose selector is not evidence for an RL mechanism.

**OPEN_MECHANISM:** other real successor requirements may constrain support layout or grasp clearance. Next inspect the already available axis-aware PlaceSupport→PlaceTop and TwoRobot full-suffix geometry. Predeclare a small crossed support-layout/actual successor-approach experiment only if geometry predicts mutually different feasible parameter preferences. Preserve all feasible symmetries and the strong full-suffix comparator; if it solves the panel, move to an existing physically contact-sensitive task instead of training on saturated labels. Do not expand this yaw grid or the old narrow hook heads. Ordinary RL plus a minimal mechanism remains conditional on real remaining opportunity.

## Files and reproducibility

- [KEY_FIGURE.png](KEY_FIGURE.png)/PDF: each measured outcome; [SUMMARY.json](SUMMARY.json), [MATCHED_CONTROL.json](MATCHED_CONTROL.json).
- [video/SYNC_VIEWER.html](video/SYNC_VIEWER.html): same-initial full250-step RAW/English annotation pair and six source-linked curves.251 frames,20fps; MEASURED_POSE_REPLAY,0 added physics, state-readback error1.19209e-7. [TRACE.pdf](video/TRACE.pdf), [caption](video/CAPTION.md), [visual QA](VISUAL_QA.json).
- `physics.tar.gz` contains all90 traces, failures, per-run exact source snapshots, protocols and logs; `PHYSICS_FILES.json` verifies every archive entry. Native ManiSkill pin/source scope in `PROVENANCE.json`. No trained model was created this round.
- `RAW_TOOL_AUDIT.json`, `RAW_PYRAMID_AUDIT.json`, `HISTORICAL_OPPORTUNITY.json`: prior raw paths, hashes and reanalysis.
- Tools: `program_pyramid_contract_probe.py`, `program_pyramid_contract_analyze.py`, `program_pyramid_contract_video.py`, `program_pyramid_contract_review.py`, `program_pyramid_contract_package.py`; historical/raw audit helpers retained. All under `research/skill_handoff_lab/tools`.
- [reproduce.sh](reproduce.sh) executes the current three control variants in fresh output directories. Exact historical code is in each archive folder. Public simulator setup is the existing pinned ManiSkill environment; no new driver/package installation.
