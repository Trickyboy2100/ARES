# ARES-R field reference (2026-10-12) — archived evidence, not live configuration

Source: user-supplied \`ARES-R_2026-10-12_Preland_Planning_Records.tar.gz\`, especially
\`README_原始记录索引.md\` and \`2026-10-12_ONSITE_PRELAND_HANDOFF.md\`.
This file is a **human-readable reference extracted from archival records**, not
a copy of the live .32 controller config or an executable scene.

Important newer-than-remote facts:

- Field ARES-R HEAD recorded on 2026-10-12: \`a9ec30eeadcb84bc702f660e6678e69743a282ee\`. Older GitHub integration HEAD in the lab checkout is \`7120c3f3...\`; many October changes may be uncommitted.
- Right arm center was reportedly reached by actual cuRobo execution; a new run must still read live joints and tool/TCP before planning.
- Epic cabinet/preland profile: \`space=2, object(config)=4, camera=1\`; **not** the earlier \`right_place_rightmost / object=3\` slot profile.
- Right-arm detected cabinet target BODY: \`[0.604866, -0.184327, 1.152342]\` m.
- Semantic preland target **70 mm above**: BODY \`[0.604866, -0.184327, 1.222342]\` m.
- Orientation: \`LEVEL_YAW_TARGET\`, BODY yaw \`0.044793 rad ~= +2.567 deg\`; tool local +Z is approach/forward axis.
- The 2026-10-12 archive describes modeled clearance policy at 10 mm, and the virtual central exclusion \`enabled=false\`. This NEVER disables true arm/world collision, self collision, controller joint limits or E-stop.
- Recorded successful direct \`center -> 70 mm preland\` plan: 301 knots, horizontal error ~0.5255 deg, modeled path clearance ~22.3668 mm, but **native packaging rejected it under the site soft joint limits**, so it was not an accepted execution trajectory.
- Previous \`center -> transit\` planning SUCCESS: 181 knots, ~48.28 mm modeled min clearance; following \`transit -> preland\` planning FAILURE: zero knots. The target was not proved blocked by a hard collision.
- The original frozen-scene planning was not reliably repeatable after re-runs / different planning settings. Do not promote one SUCCESS sample to a guarantee.

Implications for **simulation alignment**:

1. Build **two task profiles**, keeping groove \`right_place_rightmost\` separate from the cabinet \`object=4\` Preland task. Do not conflate their targets or orientation constraints.
2. Center is a **Cartesian task waypoint/region**; archived joint values can seed IK but are not immutable targets.
3. The lab must test the execution *packager's* site soft limits in addition to URDF and cuRobo; a purely geometric planning SUCCESS is insufficient.
4. Use archived successful and failed plans as **regression fixtures for error classification**, not as independent RL episodes. Several share starts, targets and scene revisions.
5. For research, generate new randomized **simulation** scenes with provenance; never label an archived scene as currently fresh on the real robot.
6. The latest controller tool/TCP full SE(3) and site joint limits require a **new read-only export** if claiming current physical matching. The static repository parameters are a baseline, not an authorization.

The original archive should remain in \`$LAB/data/site_exports/\`, outside Git. Preserve SHA and original folder structure. No hardware access is required by this note.
