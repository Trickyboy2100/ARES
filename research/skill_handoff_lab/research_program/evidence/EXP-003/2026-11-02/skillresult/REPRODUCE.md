# Reproduce (offline simulation only)

Use the existing pinned ManiSkill62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3 and method-evolution Python environment. No driver, OS or live hardware changes. Public models only. All physical invocations below must go through `tools/handoff_offline.sh` (network namespace isolated, no serial devices, GPU0). ARES-R stays read-only.

From repository root set shell variables:
```bash
PY=/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python
TOOL=research/skill_handoff_lab/tools
OUT=/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-003/skillresult_reproduction
```
Use a fresh output directory; tools refuse to overwrite old runs.

```bash
bash "$TOOL/handoff_offline.sh" "$PY" "$TOOL/program_stap_stratified.py" --out "$OUT/train96"
```
The full original96 trajectories, every failure and source snapshot are in `training_physics.tar.gz`; the224 evaluation trajectories (diagnostic112 + corrected112) are in `policy_physics.tar.gz`. Extract archives into a fresh run root. Their FILES.json checksums are independently re-read from the archive when packaging. Historical64 training trajectories remain in the preceding2026-11-01/stap_followup physical_evidence.tar.gz (`target_feedback64/`). No new simulator steps are needed to reproduce training:

```bash
"$PY" "$TOOL/program_skillresult_dataset.py" --old /path/to/extracted/stap_followup/target_feedback64 --new "$OUT/train96" --out "$OUT/datasets"
"$PY" "$TOOL/program_skillresult_learning.py" --root "$OUT/datasets/train" --out "$OUT/models31110" --updates 1200
bash "$TOOL/handoff_offline.sh" "$PY" "$TOOL/program_skillresult_evaluate_v2.py" --out "$OUT/corrected112" --model "$OUT/models31110/MODEL.pt" --methods fixed,fixed90,geometry_fixed,geometry,ordinary,generic,contract,relative,relative_contract,stap,stap_open,stap_guard,stap_uq,foresight --seeds 743050,743051 --fixed-grasp .1 --fixed-hook -.065
"$PY" "$TOOL/program_skillresult_numerics.py"
```

The new selector also requires `SUCCESSOR.pt` beside MODEL.pt. Check both SHA256 in PACKAGE.json, not just the legacy evaluator's base-model SHA. Every source and all optimizer settings are retained. Runtime v2 additionally imports `program_skillresult_logkernel.py`; packaged evaluator hashes and source in the same commit identify it. Do not treat the legacy per-case source list as exhaustive for this wrapper.

For analysis, use `program_stap_regression_audit.py` with the prior absolute_paired16 and prior models31103/MODEL.pt to generate REGRESSIONS.json, then `program_skillresult_analyze.py --root RUNROOT --out EVIDENCE_DIR` and `program_skillresult_budget.py --root RUNROOT --out EVIDENCE_DIR/BUDGET.json`. Main analysis selects corrected112 only; diagnostic outcomes remain separately reported. Checkpoint and actual source trajectories are sufficient for model replay.

Video generator `program_stap_video.py` reads the two source trajectories named in each video's MANIFEST.json. Replay introduces0 physical steps. Run `program_skillresult_video_notes.py --root RUNROOT` for source-linked TRACE figures and captions; `program_skillresult_package.py --root RUNROOT --out EVIDENCE_DIR` builds REVIEW.html and verifies every archive member and101 frames per video. Pure ranking fixture is not physical evidence. Archive path strings preserve original provenance; relocate using their relative paths when reproducing elsewhere.

For reliable browser timeline seeking use the included byte-range server (Python's basic `http.server` may expose a zero seekable range):
```bash
python research/skill_handoff_lab/tools/program_evidence_server.py --root research/skill_handoff_lab/research_program/evidence/EXP-003/2026-11-02/skillresult --port 0
```
Open the printed loopback URL; Ctrl-C stops preview. No hardware/network robot endpoints are exposed. Direct local-file viewing is also possible in browsers permitting local media.
