# SimForge Milestone Index

All milestone archives are stored in the original playground directories.
They are read-only — **do not modify milestone contents**.

---

## v1 Milestones
`isaac_sim/playground_dual_arm_control/milestones/`

| Directory | Status | Key Achievement |
|-----------|--------|-----------------|
| `20260609_191151_CST_left_tray_ear_to_chest_physics_grasp` | pass | First physics ear-grasp + carry to chest |
| `20260609_231206_CST_left_arm_IK_vertical_jaw_pad_contact` | pass | Vertical jaw IK, pad contact detection |
| `20260609_gemini2_camera_orientation_fix` | — | Camera orientation fix for Gemini2 |
| `20260610_001449_CST_horizontal_constraint_chest_retarget` | pass | Horizontal constraint + chest retarget |

---

## v2 Milestones
`isaac_sim/playground_dual_arm_control_v2/milestones/`

| Directory | Status | Key Achievement |
|-----------|--------|-----------------|
| `20260613_015554_CST_dual_arm_home_pose_gripper_fk_api` | pass ✓ | Dual arm home pose, gripper FK API documented, lift=147.5mm, force_stop at step 136 |

---

## SimForge Milestones

| Date | Demo | Status | Key Achievement |
|------|------|--------|-----------------|
| 2026-06-13 | `tray_grasp_cycle` | pass ✓ | Pure xform FK + analytical Y-contact spring; force-stop at F=3.21N; kinematic lift 29.6cm; 5+ cycles confirmed; dual-arm GUI with force history + approach chart |
| 2026-06-15 | `tray_grasp_cycle` | pass ✓ | Full orientation lock: jaw+forward both 0.0° throughout 80-step Y-linear approach; FixedJoint grasp; merged TO_NEAR→APPROACH; tray center-X alignment |

**tray_grasp_cycle verified metrics (2026-06-13):**
- Scene: `~/isaacsim/playground/20260613_tray_ear_grasp_cycle.usd`
- Grasp prim: `/World/Tray/tray_grasp_point` → world pos `[0.0727, 0.395, 1.0092]`
- IK pre-grasp pos_err: **0.0 mm**
- Force-stop: **F = 3.21 N** at `pad_y = 0.4191`, `contact_y = 0.4230`
- Lift height: **29.6 cm**
- Hold force: **≈ 10 N**
- Cycle stability: **5+ cycles** without drift or failure

**tray_grasp_cycle verified metrics (2026-06-15):**
- Scene: `simforge/scenes/main.usd` (checkpointed from `~/isaacsim/playground/2026061100_main.usd`)
- Approach: 80-step Y-linear path, `_ik_L` jaw+forward, `ORIENT_WEIGHT_STRONG=0.30`
- IK pos_err: **0.0 mm** all 80 steps; jaw_err: **0.0°**; fw_err: **0.0°**
- Force-stop: **F = 3.28 N** at `pad=[0.0725, 0.4207, 1.0022]`
- Pad X aligned to tray centre-X; approach strictly along world Y axis (position + orientation)

---

## Active Scene
`simforge/scenes/main.usd`  ← current tracked scene (snapshot of `2026061100_main.usd`, includes `tray_grasp_point` prim)

Scene changelog committed to git. Run `simforge/scenes/checkpoint.sh "msg"` to
snapshot and commit the current playground scene.
