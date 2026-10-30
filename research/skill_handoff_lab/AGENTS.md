# Scoped Codex instructions — ARES skill and client SCI research

## Read before research execution
1. `research_program/README.md` — one program entry point.
2. `research_program/STATE.json` — **authoritative current experiment and stage**.
3. `research_program/PAPER_CONTRACT.md` — stable one-paper question and scientific claims.
4. `research_program/experiments/<active experiment card>` — actual next research task.
5. `research_program/EVIDENCE_LEDGER.md` and `DECISIONS.md` — recent and historical results.
6. Relevant exact source, config, checkpoint and raw evidence for the experiment; do not
   rely on historical README summaries as current results.

## Research behavior
- Behave as an autonomous experimental researcher: inspect failures, read papers/code,
  improve physically valid interfaces, run bounded real simulation and strong controls,
  analyze causal contrasts, then choose the next informative step. A failed pilot is
  a reason to revise an experiment, not an automatic reason to end research.
- Follow the stable paper question: task-conditioned executable skill-parameter RL
  with measurable physical execution adaptation on robotic manipulation.
- First validate task difficulty, action controllability, observations and nominal
  controller. Do not immediately add a loss term or train multiple new architectures.
- Fairness: same native task/horizon/success, observation privileges, action authority,
  reward and comparable physical+offline data cost. Disclose unequal algorithms
  and additional planning search separately.
- Maintain `research_program/STATE.json`, the current experiment card, and append
  `research_program/DECISIONS.md` after substantive evidence; do not refresh the
  entire client proposal/paper or produce dozens of videos every iteration.
- Preserve raw traces, source snapshots, SHA, failures, previous checkpoints and
  sealed tests. Report training optimizer steps and simulator control steps separately.
- Use trained weights and actual physical sim feedback; never substitute pose
  replay or hand-set objects for an independent physics result. Pose playback may
  be used for labeled visualization only.
- External code/benchmarks: pin version, verify license and cite true prior-art.
  Neither standard residual RL, task conditioning, Q-learning nor action-effect
  histories are claimed novel without an independent mechanism and evidence.

## Machine and safety boundaries
ARES-R is read-only research context, never a writable deployment target.
No hardware robot/gripper/base motions, production SDK edits, network/serial
control, or use of restricted site captures in public ARES results. Do not
upgrade/reinstall Isaac, CUDA, Torch, driver or base OS to make one old example work.
Use an isolated environment for ManiSkill/SAPIEN versus Isaac Sim 5.1/Lab 2.3.0.
Inspect GPU jobs before launching. Never indiscriminately kill another person's
Isaac Sim jobs; any local cleanup must be ownership- and PID-scoped.

Review the root `CLAUDE.md` and any local worktree safety constraints, but do not
execute legacy broad `pkill` snippets against unrelated user processes.

## Execution finish
Run `python research/skill_handoff_lab/research_program/scripts/program_status.py check`
(or the available interpreter), and provide a concise Chinese summary:
what was tested, actual case/seed results, strongest competitor, uncertainty,
source/run/checkpoint paths, decision and next experiment. Do not claim an
unvalidated new method or promise background work after this session exits.
