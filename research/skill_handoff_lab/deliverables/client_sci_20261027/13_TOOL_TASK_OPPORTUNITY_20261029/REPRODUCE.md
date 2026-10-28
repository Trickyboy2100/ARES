# Reproduce and audit

Run from the research worktree. No ARES-R modification or physical devices are required. The launcher makes ARES-R read-only, removes networking and serial devices, and exposes GPU 0 only. Do not launch GPU jobs concurrently.

Pinned environment:

- Python: `/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python`
- ManiSkill source: `external/benchmark_scout_20261027/ManiSkill`, revision `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`
- Public native tasks: `PullCubeTool-v1` (100 steps) and `PlugCharger-v1` (200 steps).
- Exact installed package versions: `provenance/ENVIRONMENT.json`.

## Recompute evidence without physics

Set `B` to this deliverable directory and `R` to a **new** output directory beneath the workspace `runs/` directory. Use an absolute path for `R`.

```bash
B=research/skill_handoff_lab/deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029
R=/home/andyee/robot-research/ARES-Skilllab/runs/tool_evidence_reproduction
mkdir -p "$R/evidence"
tar -xzf "$B/recorded_physics.tar.gz" -C "$R/evidence"
python3 research/skill_handoff_lab/tools/verify_tool_evidence.py --root "$R/evidence"
python3 research/skill_handoff_lab/tools/tool_task_report.py --root "$R/evidence" --out "$R/metrics"
python3 research/skill_handoff_lab/tools/tool_task_analysis.py --run "$R/evidence" --out "$R/analysis"
```

Archive members are relative paths. Compare each extracted file to `provenance/RAW_MANIFEST.json`. All native endpoint flags, horizons, initial-state hashes, available trace hashes and finite pose/action values are checked by the verifier. No fabricated state trajectories or outcome replacement.

## Graphics-only replay

```bash
PY=/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python
bash research/skill_handoff_lab/tools/handoff_offline.sh "$PY" \
  research/skill_handoff_lab/tools/tool_task_render.py --root "$R/evidence" --out "$R/visuals"
```

The renderer calls `set_state_dict` only to display recorded robot q and actor poses, then renders. It calls **no env.step**, performs no new policy evaluation, and cannot establish new success results. Native reset constructs the scene. Unrecorded cube angular velocity is not reconstructed; this is pose playback, not a complete dynamical checkpoint. The video receipt reports readback error and original input SHA256. Camera refinement changes no outcomes.

## Refit the bounded models and evaluate new physics

The original evaluation seeds have now been inspected; rerunning them is reproduction, not a new held-out test. For future scientific confirmation, preselect genuinely new seeds and freeze code before viewing outcomes.

```bash
mkdir -p "$R/refit"
ln -s "$R/evidence/charger_design4" "$R/refit/charger_design4"
ln -s "$R/evidence/charger_train8" "$R/refit/charger_train8"
bash research/skill_handoff_lab/tools/handoff_offline.sh "$PY" \
  research/skill_handoff_lab/tools/charger_value_probe.py --root "$R/refit" --phase fit
bash research/skill_handoff_lab/tools/handoff_offline.sh "$PY" \
  research/skill_handoff_lab/tools/charger_value_probe.py --root "$R/refit" --phase eval
```

`fit` is CPU neural-network training, 3 methods × 2 boundaries × 300 optimizer updates, one fixed seed. `eval` executes 5 methods × 8 native200 episodes on GPU. Existing output directories are rejected to preserve evidence. Models use PyTorch state dictionaries and normalization arrays; load only the trusted supplied artifacts. `charger_value_probe.learned()` is the policy loader.

## Regenerate native experiments

Current strong Pull implementation is v5 (cube-oriented geometry / bounded lookahead), while the clean 8-case grid and Pull video use the **v4** source snapshot. Do not present current-code reruns as identical v4 experiments.

```bash
bash research/skill_handoff_lab/tools/handoff_offline.sh "$PY" \
  research/skill_handoff_lab/tools/pull_tool_physics.py --out "$R/native_pull" \
  --seeds 713000,713001,713002,713003,713004,713005,713006,713007 \
  --modes strong,lookahead --grasp-x .105
bash research/skill_handoff_lab/tools/handoff_offline.sh "$PY" \
  research/skill_handoff_lab/tools/charger_probe.py --out "$R/native_charger" \
  --seeds 722000,722001,722002,722003,722004,722005,722006,722007 --grid
```

Official references: use `--modes official` with Pull seeds 710000–710003 or Charger seeds 720000–720003. The original planner implementation is from the pinned ManiSkill source; instrumentation records its paths and enforces the native horizon. Strong baseline path resampling is explicitly not preservation of official TOPP timing.

Historical snapshots in `provenance/source_*.py` map to groups:

| Groups | Source |
|---|---|
| initial / debug_v2 / design8 | source_initial / source_debug_v2 / source_design8 |
| design8_v4_retry | source_design_v4 |
| confirmation8 | source_confirmation |
| geometry_v5 | final_pull_tool_physics (same physical control; later logging added) |
| charger_initial / charger_feedback_v2_retry | source_charger_initial / source_charger_v2 |
| charger_integral_v3 / charger_design4 | source_charger_integral_v3 |
| charger_train8 | final_charger_probe (same integral controller; decision observations added) |
| charger_evaluation8 | final_charger_value_probe + final_charger_probe + final_pull_tool_physics |

Snapshots import the common instrumentation and existing `handoff_physics.serial`. To execute a snapshot outside the tools directory, add the repository tools directory and pinned ManiSkill source to `PYTHONPATH` **inside** the offline launcher, or copy the snapshot to a separate scratch tools directory with those dependencies. Preserve historical files; do not overwrite current scripts. Initial wrapper versions were corrected during debugging; exact original recorded trajectories remain the authoritative historical evidence. RRT/physics repeatability is not bitwise guaranteed.

## Scope of validation

- Recorded complete episodes: 388; fully recorded native steps: 56,400.
- One early partial empty-path crash has no saved step count (positive and ≤200); one horizon configuration crash occurred before the first step. Both logs remain in provenance.
- Initial-state equality is enforced within each seed/method group. It does not imply all stochastic prefix executions are identical. Pull clean C-branch checks have zero recorded prefix difference; the Charger highlighted video also has identical recorded lift boundary state.
- No actual robot, no private field assets, no deployability or force-safety claim. No calibrated GPU-memory peak was recorded for this probe; wall-clock and planning call costs are available.
