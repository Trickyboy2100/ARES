# Reproduce the bounded physical comparisons

Use this worktree's published commit and the pinned isolated environment in the run manifests. ManiSkill SHA is `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`; Stable-Baselines3 is2.7.0. Exact local source snapshots, CLI arguments, scene seeds and hashes accompany each physical run. Existing evidence directories must not be overwritten. All commands below use the offline launcher: no devices/network, GPU0 only. Run one queue at a time for exclusive wall timing; this session's later overlaps are recorded in RESOURCE_SCHEDULE.json.

From the ARES-SCI-program repository root, set task-specific locations:

```bash
TASK_PY=/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python
TASK_SB3=/home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0
TASK_OUT=/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-003-reproduction
```

An ordinary baseline training/evaluation pair (replace `full` with `y_only` or `task` for the other fresh4096 runs):

```bash
bash research/skill_handoff_lab/tools/handoff_offline.sh "$TASK_PY" research/skill_handoff_lab/tools/program_skill_baselines.py --sb3-root "$TASK_SB3" --mode train --authority full --steps 4096 --out "$TASK_OUT/train_full"
bash research/skill_handoff_lab/tools/handoff_offline.sh "$TASK_PY" research/skill_handoff_lab/tools/program_skill_baselines.py --sb3-root "$TASK_SB3" --mode eval --authority full --model "$TASK_OUT/train_full/MODEL.zip" --out "$TASK_OUT/eval_full"
```

For real intermediate evaluation use `CHECKPOINT_2048.zip` and a new output directory. Default evaluation is the fixed six scenes732000–732005×six conditions in the source. This is a development panel, including732000 used for completion engineering; not a sealed benchmark. `--mode controls --policies geometry,search` reproduces original shared-rule controls; `search_unique_fast` is the explicitly declared stronger deterministic comparator. `--completion legacy --seeds 732000,732001` reproduces the small old-stop panel. Do not pool all policies or families into an invented benchmark rate.

The stronger actual two-skill native task panel and the causal timing controls:

```bash
bash research/skill_handoff_lab/tools/handoff_offline.sh "$TASK_PY" research/skill_handoff_lab/tools/program_multiskill_two_robot_queue.py --out "$TASK_OUT/two_robot_fullsuffix"
bash research/skill_handoff_lab/tools/handoff_offline.sh "$TASK_PY" research/skill_handoff_lab/tools/program_multiskill_slowdown_queue.py --out "$TASK_OUT/tool_slowdown"
```

The first queue plans both grasp axes from the actual push boundary, then runs the opposite-axis intervention chosen from the model decision (not from outcomes). Native task100-step limits and success remain unchanged. Its stride3 command decimation is **not preserved native TOPP timing or certified robot dynamics**. The second queue preserves the geometric TRANSFER path and changes only that phase's stride from2 to1; genuine100Hz body feedback and matched controls audit the intervention. Both are actual physics, not video restoration.

Each `RAW_EVIDENCE.tar.gz`, or `component_A/raw/<run>.tar.gz`, contains full raw traces and exact local source snapshots. A training archive includes actual MODEL.zip, intermediate weights and replay buffer; INDEX.json gives member SHA256 values verified by re-reading the archives. The referenced unchanged repository modules are retained at this published commit; external assets come from the pinned ManiSkill environment. Failed smoke versions remain historical records, not recommended commands.

The visual tools restore recorded states solely for explicitly labeled MEASURED_POSE_REPLAY. RAW means the unannotated full replay, not a live camera recording or a new physical trial. RAW/ANNOTATED frame times and synchronized event sources are audited within each video directory. `python research/skill_handoff_lab/tools/program_research_review.py` regenerates the single REVIEW.html entry after CPU analysis; no additional physics or training occurs.

## CPU reports and plots

After the actual physics jobs finish, aggregate their saved records, archive models and regenerate the review. These commands do not step a simulator:

```bash
"$TASK_PY" research/skill_handoff_lab/tools/program_skill_baselines_analyze.py --root "$TASK_OUT" --out "$TASK_OUT/reports/component_A"
"$TASK_PY" research/skill_handoff_lab/tools/program_skill_baselines_package.py --root "$TASK_OUT" --out "$TASK_OUT/reports/component_A/raw"
MPLCONFIGDIR=/tmp/ares-mpl "$TASK_PY" research/skill_handoff_lab/tools/program_skill_baselines_figures.py --evidence "$TASK_OUT/reports/component_A"
```

Point `--root` to the common parent of the actual train/eval/control directories, as in the training example above. Run manifests remain the exact invocation authority. The plots reject incompletely paired initial states; pending results are not rendered as zero successes. Existing published evidence is preserved.
