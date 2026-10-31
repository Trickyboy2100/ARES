# Reproduction and evidence boundaries

Run from the ARES-SCI-program repository root. Use a **new output directory**;
all runners reject an existing output directory. Never reset or overwrite evidence.
Only the offline launcher is authorized; it isolates network/IPC, exposes GPU0,
and keeps ARES-R read-only. No physical robot API is involved.

Environment: Python3.11 in `envs/method-evolution-20261028`, pinned ManiSkill
`62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`, SB3 2.7.0 (MIT) installed separately
with `--no-deps` under `external/sb3-2.7.0`. No CUDA/Torch/driver replacement.
Absolute paths in execution scripts document the actual local environment;
adapt roots on another machine without changing conditions or source snapshots.

```bash
bash research/skill_handoff_lab/tools/handoff_offline.sh \
 /home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python \
 research/skill_handoff_lab/tools/program_adaptive_rl.py \
 --sb3-root /home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0 \
 --mode train --method sac --steps 1024 --alpha .02 --out /tmp/NEW_SAC_RUN
```

Change method to `effect_sac` for the finite-history geometric compatibility
feature candidate, or `ln_sac` for the known critic-only LayerNorm baseline.
Use `effect_ln_sac` for the same features and the exact same LayerNorm critic.
Use explicit `--alpha .02` for matched-entropy comparisons: the historical CLI default
was .1 (now corrected to .02). The first LN run accidentally used that default and is separately
reported; it cannot isolate LayerNorm alone. No results are discarded.

Evaluation: `--mode eval --model MODEL.zip --seeds 729000,729001,729002,729003`;
the runner evaluates each seed with bias−1.2/0/+1.2mm. These are inspected
**development** scenes, not sealed tests. `--prior-only` disables the effect
update at the same checkpoint. `--zero-z` sets only the requested Z parameter
to zero, preserving learned Y and advance; it is a diagnostic action projection.
`--project-y-only` additionally fixes advance, but has not been used here.

Strong controls use `program_adaptive_contact.py --policies geometry,search,rls
--seeds 729000,729001,729002,729003 --bias-y-mm VALUE --out NEW_DIR` through
the same launcher. Each reset physically executes the native prefix; no
contact state restore or object teleport substitutes for simulation.

CPU analysis:
```bash
python research/skill_handoff_lab/tools/program_adaptive_analyze.py \
 --root /home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002 \
 --out /tmp/NEW_ANALYSIS
```

Raw per-step episodes, full source snapshots, failed malformed logs, exact
manifests and replay buffers remain in that local run root. Git contains
SHA-indexed metrics, trained models, compact `SOURCE_SNAPSHOTS.tar.gz`, and
portable raw traces for the new visualized pair,
not duplicate large replay buffers. `execution/` preserves the actual batch
commands plus the corrected follow-up script; the original LN invocation
omitted `--alpha .02` (its immutable MANIFEST records .1).

The original failed SAC serialization run consumed four full episodes/800
physics steps before any optimizer update. Recovered trace728003 is counted
once. The mistaken earlier recovery of728001 duplicates a valid trace and
is not counted twice. Preserve both and the original truncated gzip files.

No result is real-robot evidence. Native geometry/success/horizon remain, but
ARES-AdaptivePlug-v0 observation bias and mass/friction variations are custom;
±1.2mm is an exploratory calibration-error range, not a validated industrial
sensor specification. Force and full tool motion are simulator privileges
shared by every method, not validated real force sensing.
