# Controller model omission precedes any RL claim

## Confirmed by local source

The inherited `pull_tool_physics.py` controller and `program_multiskill_pull.py` call `plan_screw(..., use_point_cloud=False)` for their lookahead/candidate search and never register the cube or held tool as planner collision geometry. Native `base_motionplanner/motionplanner.py::setup_planner` constructs only the robot URDF/SRDF model. Its point-cloud flag starts false and scene geometry appears only through explicit `add_box_collision`/`add_collision_pts`, neither called here. The inherited RRT fallback also explicitly disables point clouds. The measured attachment transforms target TCP poses; it does **not** automatically attach collision geometry.

The installed mplib `Planner.plan_screw` separately defaults `use_attach=False`. Its native APIs support `planning_world.set_normal_object(name, FCLCollisionObject)` and `update_attached_mesh(path,T_link_object)`, but these are absent from the executed baseline. Therefore “strong geometric lookahead” in the scout means the strongest inherited geometric/feedback controller, **not a complete environment/attached-object collision-aware baseline**. Learning around this omission would not establish an independent RL mechanism.

## What is not yet proven

The35mm-grasp failure moves the cube away during TRANSFER/HOOK while20Hz recorded tool/cube force stays below0.178N. That does not prove a robot collision. CPU replay of saved measured q/cube poses against the Panda URDF/FCL robot and cube gives0/100 static end-frame overlap samples in both failure and successful lookahead traces (`CPU_STATIC_COLLISION_*.json`). Static separated poses do not exclude transient contact or PhysX contact margins.

Native simulation uses100Hz physics and20Hz control. `scene.get_pairwise_contact_forces` divides pair impulse by the physics timestep. End-of-control queries can miss contact inside the five physics substeps. The prepared probe therefore uses existing `_after_simulation_step`, recording all robot-link and held-tool/cube force vectors after each actual substep without adding env.step or altering commands. It must match the original complete action/state trajectory before contact attribution. Until then, the exact colliding partner is unresolved.

## Prepared enhancement, not a validated result

`program_multiskill_collision.py` has a CPU-tested native40mm cube FCL obstacle and the exact two-box L-tool mesh (not its filled AABB), attached using the measured TCP/tool transform. APIs add/remove the cube successfully and enable attached checks for TRANSFER. HOOK/PULL can retain cube-vs-robot checks while allowing tool/cube contact by disabling the attached checker. This old mplib API exposes no pair-specific allowed-collision matrix: that operation disables all attached checks, so it is not a complete-world validator. It is not yet integrated into or credited as a successful physical baseline.

Next useful evidence is the two instrumentation-only native100 replays, then a measured collision/clearance correction if warranted. No Tool RL training should precede that diagnosis and a fair strengthened controller comparison.

## Resolved by the subsequent read-only100Hz probe

Both repeat trajectories exactly match all original actions/states. Failure35mm grasp reports held-tool/cube4.011N duringTRANSFER, while every robot-link/cube force remains0 at all500 physics substeps. Success70mm reports noTRANSFERcontact and16.614N duringHOOK. Therefore the near-zero20Hz value was sampling aliasing, and these data support premature tool contact rather than the proposed robot collision. The robot-collision hypothesis is not retained as the explanation for this pair. Attachment drift is13.194°/7.017mm versus0.270°/0.288mm duringTRANSFER; rigid geometry versus physical grip compliance contributions still require separation. See `../multiskill_contact_probe/` for exact-repeat audits and raw substep evidence.
