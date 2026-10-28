# Protocol execution notes

PROTOCOL.json is a locally written pre-development pilot record, not an external preregistration. E0 code and main-training learner snapshots are separately preserved with hashes.

After E0 and before opening development cases, reporting-only plan-cache / verified-success fields and full development trace persistence were added. The geometry baseline was strengthened from nominal C to a current-jaw-clearance heuristic before any development results; no learned architecture, main-training hyperparameter or online budget was changed. These baseline/evaluation changes explain the phase-source hashes.

After development, analysis corrected the meaning of raw verified_program_success (see RAW_SCHEMA_NOTE), wrapped figure labels and audited risk calibration. No native score, checkpoint or physical trajectory was modified, and no development outcome was used to select a replacement model. The calibrated-risk audit is posthoc diagnosis only.

RRT fallback uses a0.5s OMPL search budget plus IK overhead, not a guaranteed0.5s end-to-end deadline. Actual measured planning wall time is retained in PERFORMANCE.csv. MPLib0.1.1 does not expose a global RNG seed in the installed Python API; strict start-state-matching candidate-plan reuse aligns paired prefixes, but a new independent collection can differ in randomized fallback plans. Neither publication-grade planner repeatability nor industrial collision certification is claimed.
