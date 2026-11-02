# Reproduction

Use the pinned existing method-evolution environment and offline wrapper. All tasks are simulation; ARES-R read-only. Do not launch concurrently on the GPU.

```bash
cd /home/andyee/robot-research/ARES-Skilllab/repos/ARES-SCI-program
PY=/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python
ROOT=/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-003/contract_sac_20261102
# Existing result root: analyze, do not overwrite physics files.
$PY research/skill_handoff_lab/tools/program_contract_diagnostics.py --root "$ROOT" --out "$ROOT/analysis"
$PY research/skill_handoff_lab/tools/program_contract_prediction_audit.py --root "$ROOT"
$PY research/skill_handoff_lab/tools/program_contract_figures.py --root "$ROOT" --out "$ROOT/figures"
# New runs require a NEW output directory. Typical full native physical controller pair:
bash research/skill_handoff_lab/tools/handoff_offline.sh "$PY" research/skill_handoff_lab/tools/program_tool_contract_control.py --out "$ROOT/../ares-contract-new" --seeds 745000 745010 --masses .75 1.25 --grasps .035 .105 --modes corrected_feedback corrected_arrival
# True two-boundary ordinary SAC (new output). 256 is a checkpoint budget, not convergence.
bash research/skill_handoff_lab/tools/handoff_offline.sh "$PY" research/skill_handoff_lab/tools/program_tool_smdp_sac.py --out "$ROOT/../ares-smdp-new" --decisions 256
```

The original recovered run used `--warm "$ROOT/sac/episodes"`, seed31141, requested1024 decisions, intentionally interrupted after checkpoint512 for mechanism evaluation; saved575 completed decisions and1022updates. Do not claim requested1024 were completed. The failed first run actor was not recovered. Model checksums and every training physical row are in RAW_MANIFEST; unpack all RAW_*.tar.gz into the original root or rewrite old-root prefixes in RESULTS/manifest references to your new root. Never use archived pose replay as physical evaluation.

Each archive's SHA and each contained member SHA are available. `package_contract_evidence.py` reads back every member to verify. The `final_source/` directory is the final code; per-run source snapshots and copy timing limits are documented. Restore checkpoints with the pinned existing SB3-2.7.0 and custom `program_sac_policy.CriticLayerNormPolicy`; never load untrusted pickle models. The stored models are this experiment's own output.

Full query-count instrumentation was added after early training; only recorded queries are evidence. Counts of requested tests, configured training steps, rendered frames or synthetic playback are never substituted for executed physical steps.
