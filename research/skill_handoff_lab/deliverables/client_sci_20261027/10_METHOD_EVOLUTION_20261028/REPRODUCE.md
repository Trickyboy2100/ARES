# Reproduce / inspect without site hardware

Run from the independent ARES worktree. Original paths below describe the executed machine; change `workspace`/paths on another machine. Never invoke ARES-R production scripts. All physical commands must go through `method_offline.sh` (bubblewrap, no network, private devices, GPU0 only, ARES-R read-only). Installing dependencies/fetching public upstream is a separate preparation step, not an action inside simulation.

## Effective environment

Pinned public ManiSkill `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`, SAPIEN3.0.3, Torch2.7.0+cu128, Gymnasium1.4.0, NumPy1.26.4, Python3.11.15, MPLib native public planner. The executable used is `/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python`, a system-site-packages venv over `jnl263_round42_maniskill`. Only missing `defusedxml==0.7.1` and `gymnasium==1.4.0` were installed there. Inherited IsaacLab/old ManiSkill distribution metadata has conflicts; the effective module is **the pinned source on PYTHONPATH**, not the inherited distribution version. Do not replace healthy Torch/CUDA or NVIDIA drivers to match a metadata label. See actual module/version/source hashes in the environment receipt.

Layout expected by the isolated launcher:

```bash
workspace=/home/andyee/robot-research/ARES-Skilllab
root="$workspace/worktrees/ares-sci-method-evolution-20261028"
python="$workspace/envs/method-evolution-20261028/bin/python"
tools="$root/research/skill_handoff_lab/tools"
upstream="$workspace/external/benchmark_scout_20261027/ManiSkill/examples/baselines/ppo/ppo.py"
cd "$root"
```

The launcher derives `workspace` from this worktree layout. On another layout adapt those two path derivations first. Native Panda assets come from the pinned public source; asset hashes are in `PUBLIC_ASSET_INDEX.json`. Driver/GPU availability is a prerequisite, not mocked by the scripts.

## Verify existing evidence (CPU)

```bash
"$python" "$tools/method_verify.py" --runs "$workspace/runs/method_evolution_20261028" \
 --output "$root/research/skill_handoff_lab/deliverables/client_sci_20261027/10_METHOD_EVOLUTION_20261028"
```

This checks real weight hashes/loaders, paired state hashes, consumed authority, native caps, train/calibration splits, forbidden sealed cases, and original task source hashes. The models were already evaluated through the real native simulator, not only loaded successfully. `RESEARCH_BUNDLE.json` identifies the portable archive. Extract it to a fresh directory; archive-relative `runs/...`, `source/...`, and `public_evidence/...` preserve exact inputs. The original absolute paths in old JSONs are provenance, not a requirement to overwrite any current run.

## Minimal actual GPU reproduction (fresh output only)

```bash
fresh="$workspace/runs/method_reproduction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$fresh"
bash "$tools/method_offline.sh" "$python" "$tools/method_stack_physics.py" \
 --output "$fresh/stack_probe16" --cases 16 --num-envs 16 --height-only
bash "$tools/method_offline.sh" "$python" "$tools/method_native_planner_probe.py" \
 --output "$fresh/pyramid_case122007" --seed 122007 --cases 1 --parameter-probe --video
```

The first command costs2,400 control interactions; the second executes real single-environment native planning and all three paired programs, never beyond250 steps each. Batch arity is part of the reset condition: an N1 seed is not asserted to equal the same numeric seed in an N16 batch.

## Exact PPO reproduction and learning cost

Use `repro/frozen_r1/method_train.py` and its adjacent frozen controller for the training sources that generated the recorded weights. The current controller's logging distinguishes issued height history from consumed height; actor inputs/actions are unchanged. Preserve the frozen versions to match controller hashes.

```bash
frozen="$root/research/skill_handoff_lab/deliverables/client_sci_20261027/10_METHOD_EVOLUTION_20261028/repro/frozen_r1"
for seed in 31 37 43; do
 for mode in residual parameter; do
  bash "$tools/method_offline.sh" "$python" "$frozen/method_train.py" \
   --upstream "$upstream" --mode "$mode" --seed "$seed" --steps 102400 \
   --name "${mode}_${seed}_102400" --run-root "$fresh/training"
 done
done
```

Six runs cost614,400 online interactions plus native monitoring. This is ordinary upstream PPO, no custom likelihood. The monitor uses development conditions and does not fit the supervised selector. It is not a sealed test. Native reward/dynamics/horizon are never edited.

## Paired collection, capacity-matched fits, equal-cost search

```bash
bash "$tools/method_offline.sh" "$python" "$tools/method_stack_physics.py" \
 --output "$fresh/train512" --cases 512 --seed 100000 --num-envs 16 --height-only
bash "$tools/method_offline.sh" "$python" "$tools/method_outcome_selector.py" fit \
 --data "$fresh/train512" --output "$fresh/selectors"
bash "$tools/method_offline.sh" "$python" "$tools/method_cem.py" --output "$fresh/cem"
bash "$tools/method_offline.sh" "$python" "$tools/method_outcome_selector.py" eval \
 --upstream "$upstream" --models "$fresh/training/runs" --selector "$fresh/selectors" \
 --cem "$fresh/cem/RESULTS.json" --output "$fresh/development64" --cases 64 --seed 200000
```

Collection and CEM each cost76,800 actual interactions. Development64 with all16 policies costs51,200. All three branch costs are charged; no test oracle selects CEM parameters. Lower/higher are single-boundary choices, so this is a contextual bandit, not an SMDP. Every branch really executes50 steps; earlier stable success is diagnostic only. For actual Stack video add `--cases 16 --video --video-indices 3,6 --only nominal,parameter_ppo_43,residual_ppo_43` with a **new** output directory. Never overwrite the original evidence.

Peg scripts `method_peg_physics.py` (default round1 / `--iteration2`) and `method_peg_feedback.py` reproduce bounded native100-step negative controls. `method_native_planner_probe.py --seed 122000 --cases 16 --parameter-probe` reproduces the first-skill Pyramid authority study at native250. See script `--help` and recorded run-result SHA indexes for full run identities.

## Figures, PDFs and historical evidence

`method_analyze.py` reconciles the original run-folder names from this release, exports measured TensorBoard curves and actor audits. `method_publish.py` builds individual PNG/SVG figures, actual-frame filmstrips and labeled videos; `method_history.py` separately re-reads prior public Stack64. Their default replay is plotting, not new physics. Sources are in figure/raw manifests. Re-run the history publisher after the main publisher, which rebuilds `index.html`.

Base Python with NumPy/Matplotlib/Pillow/ffmpeg builds figures. The bundled document Python builds the Chinese PDF with `--pdf-only` (ReportLab, CJK/Latin fallback). Compile English from this directory with `pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error PAPER_METHOD_EVOLUTION.tex` twice. System `/usr/bin/pdftoppm` renders final PDFs; the bundled Poppler binary requires a newer glibc on this host and was not used. All final pages were visually reviewed.

## Data / model preservation

All final/initial models, useful intermediate models, failure runs and original full trajectories remain outside Git. No unique checkpoint was deleted. Only reproducible TeX auxiliary files were cleaned after their diagnostics were copied outside Git. No site logs, images, production assets or SDK binaries are included. There is no permanent scheduler or detached training process; scripts have bounded loops and the batch runner terminates only its own child on timeout.
