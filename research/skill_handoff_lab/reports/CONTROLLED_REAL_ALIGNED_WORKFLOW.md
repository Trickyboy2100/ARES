# Controlled real-aligned research workflow

The complete October archive has now been received and verified on the research
host. The previous audit at b0c8b58 remains historical evidence; it is not deleted
or silently rewritten. ARES-R remains a read-only reference worktree.

## Publication boundary

This repository contains generic tooling and synthetic security tests ONLY.
Do not add archive payloads, field configuration, robot states, planner requests,
generated Site USDs, pointclouds, diagnostic plots, or videos to public Git/LFS.
Derived field figures can disclose physical scene state too. Use a restricted
non-Git directory and an owner-only access policy. The tools reject enclosing
Git worktrees; that is defense in depth, not a substitute for review.

## Order of operations

1. Read the public index README, DELIVERY_MANIFEST and verify_archive.sh at the
   specified index commit. Fetch objects without switching the ARES-R worktree.
2. `ingest_controlled_archive.py` verifies outer SHA/size/gzip, safe tar paths,
   public/internal source hashes and all three CAMERA-to-BODY point sets before
   formal extraction. Output is read-only, owner-reversible (not WORM).
3. `audit_controlled_field.py` binds requests to actual scene epochs and performs
   bounded independent modeled replays. Reference configuration dates stay
   explicit; a later config is not relabelled as an original frozen config.
4. `plot_controlled_field.py` generates PRIVATE measured-surface overlays and
   sections. Unknown space is not inferred as solid or collision-free. Actual
   request goals override unrelated scene diagnostic A/B goals.
5. `prepare_controlled_replay.py` records relocation and optional six-axis YAML
   adapter changes. Original URDF hashes and frozen planner constraints remain
   checked. Missing original onsite planner YAML stays a reproduction limitation.
6. `replay_worker_compat.py` repairs a reporting-only six-vs-seven-DOF reshape in
   an in-memory research copy; no ARES-R production source is edited. The original
   and research worker hashes are both recorded.
7. `build_controlled_site.py` and `replay_controlled_site.py` test a SINGLE
   physically stepped, archived transport segment. Original robot mesh colliders
   remain; field attached AABB is explicitly NOT tray CAD. Initialization-only
   state writes are separated from Articulation drive targets. This is not RL,
   successful placement, or whole-robot physical safety certification.
8. `qualify_controlled_physx.py` checks actual pose, yaw, reference joint dynamics
   and soft limits independently. Terminal reach is not complete execution
   acceptance; net contact force does not certify penetration-free physics.
9. `probe_controlled_ik.py` is a fixed-budget endpoint-only numerical probe, not
   a trajectory label. `summarize_controlled_run.py` keeps private diagnostics
   and artifact hashes. Preserve failed probes, software errors and shutdown logs.

Run all GPU work with a private output directory, one selected idle GPU, an
external finite timeout, and independent logs. Never call hardware interfaces,
global pkill, shared driver maintenance, or overwrite old training outputs.

The existing 16-env training platform and pilot checkpoints are retained. New
field-aligned RL training requires correctly defined, legal complete transitions
and fair baselines; no claim that RL must be uniquely necessary is imposed.
