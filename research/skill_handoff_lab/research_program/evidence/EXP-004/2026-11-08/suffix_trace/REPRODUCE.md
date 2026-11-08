# Reproduction

Use the existing pinned public ManiSkill environment and dependency models from
`../continuation_retention/REPRODUCE.md`. This is the established workspace layout,
not a new portable framework. `ENVIRONMENT.json` in RAW pins Python/Torch/ManiSkill.
No system package or driver upgrade is required. ARES-R stays read-only.

Verify every archive SHA in `RAW_MANIFEST.json`, extract all parts together into
an EMPTY directory, then verify memberSHA. The archive contains actual traces,
all checkpoints, replay, optimizer/RNG states, executed source copies, logs,
protocols and the exact fork32 snapshots. Historical public dependency checkpoints
remain available in the preceding release's archive; no restricted field assets
are needed. Original absolute paths can be consistently rebased on another host.

## Recount without simulation

```bash
LAB=/home/andyee/robot-research/ARES-Skilllab
RUN="$LAB/runs/research_program/EXP-004/suffix_trace_20261108"
PY="$LAB/envs/method-evolution-20261028/bin/python"
TOOLS=research/skill_handoff_lab/tools
"$PY" "$TOOLS/analyze_suffix_trace.py" --root "$RUN" --out /tmp/suffix-main
"$PY" "$TOOLS/analyze_suffix_trace.py" --root "$RUN/ablation" --out /tmp/suffix-ablation
```

Add the two cost objects for unique new physical execution. Each primary arm
consumes the SAME inherited2218-transition replay and strong learned checkpoint;
its historical collection/training cost is inherited, not free and not rerun.
Zero-entropy forks inherit their corresponding first32 own episodes. These32 are
charged in each full-policy comparison but not counted twice in unique cost.

## Actual own-policy learning

Commands are preserved in `run_matched_extension.sh` and manifests. Example:

```bash
bash "$TOOLS/handoff_offline.sh" "$PY" "$TOOLS/train_suffix_trace.py" \
 --out "$LAB/runs/reproduction_suffix_trace_sac" \
 --initial "$LAB/runs/research_program/EXP-004/action_frontier_20261107/sac_32202/CURRENT.pt" \
 --mode sac --stop-after 32
# Same command and output, with --resume and without --stop-after, continues to64.
```

Use `--mode trace` or `--mode lagged` for matched arms and NEW output directories.
Defaults:64 own episodes, training769100–163, dev769400–411,40 Q/Actor updates per
own episode, fixed0/32/64 checkpoints. No best-dev selection. Reference-policy
snapshot interval8 applies only to lagged. All real collection is network-isolated
with `handoff_offline.sh`, localGPU0, officialPeg100 success and horizon.

`make_suffix_forks.py` reconstructs checkpoint32 models and exact prefix replay in
a NEW destination. The archived `FORK32_sac.pt` and `FORK32_trace.pt` are the actual
initial files used. `train_suffix_zero_entropy.py` takes those via `--initial`,
`--mode sac/trace`, a NEW `--out`,32 new episodes769132–163. Only the entropy terms
are zero; uniform exploration and successBC remain. Fork evaluation0 was actually
reexecuted and matched all100 recorded steps on each of12 scenes in both arms.

The primary and zero-entropy source files intentionally preserve separate frozen
versions; compare their diffs to see the exact changed terms. Numerical tests
`check_suffix_trace.py` check termination, successor indexing, unknown-likelihood
fallback, and exact model/optimizer/RNG update continuation on collected data.

## Frozen full task evaluation and visualization

`run_confirmation.sh` and `run_ablation_confirmation.sh` retain exact commands.
Use NEW destinations if repeating; native evaluation begins at reset, not at a
handoff setter. Primary model hashes were fixed before confirmation; ablation
hashes were added after primary evaluation began, with no confirmation-driven
training or checkpoint choice. This remains exploratory development, not sealed
testing or a multi-initialization convergence result.

`render_selected.sh` reconstructs the representative full video from saved
measured states. It adds zero physics steps and cannot provide a second success
trial. The source native runs, not rendering, define outcomes. RAW and English
annotation share101 frames/20fps; the synchronized viewer embeds the same data.
