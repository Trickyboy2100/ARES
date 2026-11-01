# Reproduction

Run from the ARES-SCI-program repository, with the existing isolated environment. No hardware/network adapter is imported by these runners. All physics/rendering goes through `tools/handoff_offline.sh` (network namespace disabled, GPU0, ARES-R read-only). Python path used:
`/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python`.
ManiSkill source pinned at `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`.

```bash
python research/skill_handoff_lab/research_program/scripts/audit_exp003.py
python research/skill_handoff_lab/research_program/scripts/program_status.py check
```

Use the pinned environment's Python for the following. Commands create **new** output directories; use a fresh output name, never overwrite an old experiment.

1. `program_stap_fixture.py --source research/skill_handoff_lab/research_program/evidence/EXP-003/2026-11-01/stap_followup/kernel/official_source --out NEW_FIXTURE`: original STAP algorithm body vs port. Source snapshot and LICENSE are in `kernel/official_source`; preserve original directory layout when using that snapshot. Official full repository pin is in STAP_MAPPING.md.
2. Extract `physical_evidence.tar.gz` into a fresh directory and verify every SHA in ARCHIVE_MANIFEST.json. No path in the archive is absolute or contains `..`.
3. `program_stap_learning.py --root EXTRACTED/target_feedback64 --out NEW_MODEL --updates 800`; repeat with `--residual-dynamics` for the paired common-model repair. Exactly the recorded64 real episodes are reused, not re-simulated. One training seed31103 by default; two first-stage variants `fq,foresight`.
4. Run via `handoff_offline.sh`: `program_stap_evaluate.py --out NEW_EVAL --model NEW_MODEL/MODEL.pt --methods fixed,stap,fq,foresight`. Defaults match2 new seeds×2 masses×2 declared targets×4 policies=32 complete actual100-step episodes. `stap,stap_open` uses identical per-case RNG to isolate posterior replanning. `--samples 64` changes CEM model-query budget; it does not buy simulator rollouts.
5. Strong controls: `--methods fixed --fixed-grasp .1 --fixed-hook=-.065`; contract control: `--methods stap_guard`. The latter uses known remaining budget and actual-state planner costs to retain the originally planned legal hook when the proposed hook is over budget. It is deterministic planning, **not RL novelty**; extra geometry candidates are counted.
6. Recompute all tables: `program_stap_analyze.py --root EXTRACTED --out NEW_ANALYSIS`; source-path fields inside historic model indexes are audit provenance, not a requirement to have those absolute paths. Physical data loaders discover actual files beneath `--root`.
7. `program_stap_visual_summary.py --root EXTRACTED --out NEW_ANALYSIS` generates independent scientific figures. Matching video DATA in sibling directories additionally generates TRACE figures.
8. Video: `program_stap_video.py --left TRACE1 --right TRACE2 --left-label '...' --right-label '...' --out NEW_VIDEO` through the offline wrapper. Requires matching initial SHA, two actual decisions,100 steps. Restores measured poses only; never calls env.step. Full101 frames at20fps. Original/annotated files remain separate. This is not original online-render capture.

Historical per-episode source snapshots are authoritative for old experimental variants. Current tools include later explicit improvements, so reproducing an **earlier** controller version requires its saved sources rather than assuming current HEAD is identical. Training and evaluation manifests record model SHA. First startup failure, initial absolute-model results, uncontrolled-RNG exploration and failed transfer repair remain in the archive.

No sealed test was used. Calibration, model repair and guard decisions used these development cases; future generalization requires new scenes and training seeds. Exact render pixel reproduction can depend on driver; measured state readback and source hashes are independently checked.
