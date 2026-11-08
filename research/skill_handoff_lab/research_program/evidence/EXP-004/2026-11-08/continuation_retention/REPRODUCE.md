# Reproduce this completed-cycle evidence

## Data and versions

`RAW_MANIFEST.json` lists every archived source, actual simulation trace, replay,
checkpoint, optimizer/RNG state, log, protocol and per-memberSHA. All RAW parts
belong to one run tree, `runs/research_program/EXP-004/continuation_retention_20261108`.
Extract them together into an empty run directory after verifying archiveSHA;
never overwrite a historical run. `SOURCE_f3f57c8`, `SOURCE_c9efa71` and
`SOURCE_7155000` retain earlier tool versions. Per-run `MANIFEST.json` and source
hashes are authoritative for each trained model; the analysis code is in this
release's repository commit. `FROZEN_COMPARISON.json` pins every evaluation model
before opening768600–631.

`ENVIRONMENT.json` pins the actual interpreter, ManiSkill source, SAPIEN,
Torch/CUDA and native task source. This work uses the existing isolated Python
and the public ManiSkill checkout; no package, driver or system upgrade is needed.
The shell wrapper `handoff_offline.sh` disables networking for physics jobs and
restricts execution to the local GPU. Do not invoke production robot adapters.

`DEPENDENCY_MODELS.json` maps the seven public-simulation historical checkpoints
used by the warm executor and historical controls to their original paths andSHA.
Exact copies are included under `dependency_models/` in the RAW archive. Restore
those under the listed paths relative to the workspace `runs/` directory if
absent; verify an existing file rather than replacing it. Native robot/scene
assets remain the pinned public ManiSkill assets, not ARES-R field assets.

These research scripts retain the existing workspace layout rooted at
`/home/andyee/robot-research/ARES-Skilllab`. On another host, consistently rebase
that root in source constants and archived path-bearing metadata; changing only
one output path is insufficient. No claim of a newly packaged portable platform
is made. Run under the pinned environment, from `repos/ARES-SCI-program`.

## Independent recount, without physics

```bash
LAB=/home/andyee/robot-research/ARES-Skilllab
RUN="$LAB/runs/research_program/EXP-004/continuation_retention_20261108"
PY="$LAB/envs/method-evolution-20261028/bin/python"
TOOLS=research/skill_handoff_lab/tools
"$PY" "$TOOLS/summarize_continuation_release.py" --root "$RUN" --out /tmp/ares-recount
"$PY" "$TOOLS/continuation_paired_intervals.py" --summary /tmp/ares-recount/SUMMARY.json --out /tmp/ares-paired-intervals.json
"$PY" "$TOOLS/analyze_positive_option_windows.py" --root "$RUN"
```

The first two commands write separate output. The last recomputes the derived
window diagnosis in the extracted run tree. Archived native traces are never
modified. `HORIZON` is a recorded task-budget termination, not automatically an
infrastructure error. Planner counts are recorded `plan_screw` outer calls, not
all internal collision/IK operations.

## Actual learning and resume

The archived `run_seed32301.sh`, `run_seed32302.sh`, `run_witness_qp.sh`,
`run_witness_soft_full.sh` and `run_witness_margin_pilot.sh` preserve actual
commands. Their old process-waitPIDs are provenance, not reusable scheduling
instructions: rerun individual commands into **new output directories**.

For example, the primary corrected candidate is:

```bash
bash "$TOOLS/handoff_offline.sh" "$PY" "$TOOLS/train_witness_qp.py" \
  --out "$LAB/runs/reproduction_witness_qp_32301" \
  --shared "$RUN/SHARED.pt" --arm witness_qp --seed 32301 --episodes 128
```

Default train seeds768100–227, dev768400–415 and40 optimizer updates/episode
match this cycle. Add `--resume` only to resume that same run/manifest. Checkpoint
and transaction replay preserve Actor, Critic, optimizer, RNG and collection
state. Soft32301 was actually stopped at64 and resumed to128. Numerical checks
exercise exact Actor/Critic continuation; these checks are not additional
physical successes. Ordinary arms use `train_continuation_retention.py` with
`--arm sac`, `ranking`, or `retention`; soft uses `train_witness_option.py`
(the archived full command is authoritative). Primary rear experiments keep
the task-conditioned warm predecessor fixed. The jointSAC control alone passes
`--front-unlock 0`; it is reported separately.

## Frozen complete evaluation

`run_frozen_slot0.sh` and `run_frozen_slot1.sh` contain every exact invocation.
Use new output directories and omit historical waitPIDs. Evaluation uses
`evaluate_continuation.py --freeze-front --mode learned --model CHECKPOINT_128.pt
--start 768600 --count 32`; historicaloldRL uses its original evaluator/interface.
No training, handoff reset, budget extension or dev-best checkpoint selection
occurs. New confirmation is developmental data, not the sealed test.

## Visual evidence

Open `REVIEW.html` locally. Each figure has a separate vectorPDF andPNG.
The two `video_*/` folders contain completeRAW and EnglishANNOTATED MP4s,
exact-frame `SYNC_VIEWER.html`, sourceSHA and readback error. These are explicitly
labeled `MEASURED_STATE_REPLAY`: measured physical joint/object trajectories are
rendered without taking new simulator steps. Their native success evidence is
the source physical run, not the rendering.
