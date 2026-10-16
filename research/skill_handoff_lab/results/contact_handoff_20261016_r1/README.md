# Contact handoff v6 — real online physics, including negative results

2026-10-16. Inherits ARES `8ce55fa`; core fixes `f34a39c`. The containing Git commit is the release version. **SYNTHETIC ARES TASKS / STATE-BASED / ACTUAL_PHYSX**, not field, real robot, grasping, or an official benchmark. ARES-R production code was not changed.

## Start here

- `RESULTS_V6.json.gz`: all observed run rows, independent success, paired mechanism, seed statistics and matched-success timing.
- `fig_frozen_test_v6.png`, `fig_online_learning_v6.png`, `fig_handoff_mechanism.png`, `fig_contact_failure_diagnostic.png`, `fig_method_v6.png`.
- `v6b_actual_training_video/actual_physics.gif`: actual 16-env **online training**, a separate 16,384-interaction continuation; not the model selected for the main table.
- `RELEASE_INDEX.json`: every run, original/compressed hashes, and authoritative execution-source mapping.
- `SHA256_MANIFEST.json`: published file hashes (excluding the manifest itself).
- `ARTIFACT_MANIFEST_V6.json.gz`: original complete local archive file hashes, including files deliberately not published here.

## Exact scope

Nine main models: `v6b_{ppo,residual,recover}_seed{31,41,51}`, each 65,536 real interactions. All 32k/64k after-update checkpoints are retained; final checkpoint fixed in advance. Other models are named pilot, active2 diagnostics, bootstrap ablation, or training-video continuation. Total new training: **901,120 interactions across 16 trained runs**, not 16 formal seeds.

All completed frozen case records, independent audits, training episodes, credit audits, optimizer CSV logs, configs and model files are retained, including failed methods. JSON case/episode files use gzip, with original byte hashes recorded. Exact execution Python snapshots are deduplicated in `source_snapshots/`; do not assume current HEAD equals all historical executed sources. Snapshot files named `boundary_credit_source.py` in runs contain the actual v2 estimator bytes, not the old v5 algorithm.

Full local archive: `/home/andyee/robot-research/ARES-Skilllab/runs/contact_handoff_20261016_r1/` (~1.2 GB). Full 120 Hz traces and preclip/clipped/applied action logs remain there. This public subset additionally includes full traces/actions for residual31 validation, recover31 validation and forced-resume31, plus measured contact-history diagnostics. Excluded raw traces are **not** silently treated as public files; see the archive manifest for hashes. Software-incomplete early runs remain identified, not counted as physical task failures or erased. Older v5 helper-source provenance gaps are not retroactively filled.

No controlled field cloud, production configuration, real robot readback, or field-derived visualization is included. New states are generated in the self-built synthetic task.

## Results / interpretation

Independent task success (seed31,41,51), 32 same cases per split:

| Method | ID | Geometry | Dynamics |
|---|---|---|---|
| Damped feedback | 29 | 8 | 28 |
| Direct PPO | 0,0,0 | 0,0,0 | 0,0,0 |
| Residual PPO | 28,28,29 | 13,18,23 | 28,29,29 |
| Residual + hold/proceed | 28,30,30 | 15,15,18 | 28,29,29 |

Residual RL shows preliminary geometric adaptation vs tuned feedback. It is **not a new algorithm**. Hold/proceed does not beat ordinary residual consistently. Joint/active-likelihood pilot: 56 vs 57/64 validation, failed predeclared +4 promotion threshold. Do not advertise this as a validated core SCI increment. Three seeds and repeated case groups are not hundreds of independent trials. Direct PPO remains insufficiently reliable; actual 2D diagnostics are included rather than hiding this limitation.

## Reproduction

Requires the deployed Isaac Sim 5.1 / Isaac Lab 2.3.0 environment, SB3/Gymnasium/Torch, and existing JAKA asset dependencies. Environment/dependency setup is inherited, not a new platform installation. `configs/contact_handoff_v6b/` and `asset_manifest.json` freeze research geometry. `build_contact_composition_asset.py --help` exposes the narrow-paddle asset constructor; compare the resulting asset hash rather than assuming regeneration is byte-identical across USD builds. The exact 59 MB USD is also in the full archive, not pushed repeatedly to Git.

1. Verify release hashes using the command below. Read `run_config.json` and corresponding actual batch JSON for exact arguments and case seed. Substitute only local path roots/GPU/output; use a fresh output directory.
2. Restore that run's task and helper source snapshots using `exact_source_snapshot_mapping`, renaming them to `contact_composition_v6.py`, `contact_credit_v2.py`, and (if present) `contact_hybrid_policy.py`. Use `run_contact_v6.py` as launcher. Do not overwrite the frozen source or original run directory.
3. Main queues were `run_handoff_v6_study.py`; final frozen tests `run_handoff_v6_evaluation.py`. They launch one bounded worker per GPU, preserve failures, require `RUN_COMPLETE.json`, and never call global pkill. The actual command batches are archived as gzip JSON.
4. Independent audits use `validate_contact_composition.py` with the correct narrow research asset and fixed JAKA URDF; old v5 replays require the old wide asset. `audit_contact_robot_mesh.py` provides an explicitly sampled additional mesh diagnostic, not full CCD certification.
5. Rebuild figures with `summarize_handoff_v6.py --root <full archive>` and this compact release with `package_contact_v6.py`. SB3 checkpoints are trusted project artifacts, not untrusted models.

```bash
python - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('research/skill_handoff_lab/results/contact_handoff_20261016_r1')
for x in json.loads((r/'SHA256_MANIFEST.json').read_text()):
    p=r/x['path']
    assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'], p
print('published hashes verified')
PY
```

49 unit tests passed; unit fixtures are not formal training labels. Full findings, nine requested questions and research decisions: `reports/CONTACT_HANDOFF_RESULTS_V6_20261016.md`. SCI manuscript is a working draft, not submission-ready.
