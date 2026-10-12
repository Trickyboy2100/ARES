# P1/P2 CPU SkillHandoffBench smoke — 2026-10-13

## Scope and provenance

This is a CPU-only planning-evidence smoke. It did not launch Isaac, cuRobo,
contact dynamics, or robot control. The immutable run used for conclusions is
`/home/andyee/robot-research/ARES-Skilllab/runs/p1_p2_cpu_smoke_20261013_v4`.
Failed engineering versions v1--v3 are retained beside it. The requested
`RESEARCH_ROADMAP_20261013.md` and `CODEX_NEXT_P1_P2_20261013.md` were not
present in the local or fetched ARES branch; no historical content was inferred.

Inputs are the pinned ARES JAKA MiniCobo + EG2-4C2 URDF, ARES-R BODY right-arm
mount, the commissioned site soft limits with 0.15 rad margin, and the archived
2026-09-21 read-only right-arm FK correction/tool-2 record. The latter is not a
current live TCP export. `center` is generated as a Cartesian BODY goal. Four
distinct joint-space IK branches target the same center pose per scenario.

## Contracts and environment

`Scenario`, `Candidate`, `Transition`, and `Outcome` use
`skill-handoff-bench-v1-20261013` and carry record hashes. `reset` accepts only
the scenario's frozen seed. `step` binds a candidate to its scenario and checks
candidate legality, held-object contract, fresh scene revision, successor IK,
successor endpoint orientation, and endpoint collision. A 25-sample linear
joint interpolation is logged as a descriptor but is **not** called a planned
trajectory and is not the endpoint-start label.

ARES-R `_edge_cost` is recorded as the hand-built baseline. It never supplies
the Outcome label. Labels come from successor IK plus endpoint geometry.

## Fixed smoke and result

| Family | Scenarios | IK candidates | Legal center candidates | Successor-start feasible | Candidate-sensitive scenarios |
|---|---:|---:|---:|---:|---:|
| Tray to Groove | 3 | 12/12 | 12/12 | 0/12 | 0/3 |
| Tray to Cabinet Preland | 3 | 12/12 | 12/12 | 12/12 | 0/3 |
| Total | 6 | 24/24 | 24/24 | 12/24 | 0/6 |

The generator is stable for this frozen sample: all six scenes and all 24
terminal candidates were generated and passed endpoint FK/IK, site-soft-limit,
orientation, and conservative collision checks. However, this batch does **not**
show within-scene terminal-candidate sensitivity. Groove is uniformly infeasible
and Cabinet uniformly feasible under the CPU successor-start proxy. This is a
negative result for the current headroom hypothesis, not evidence that terminal
state can never matter.

## Failure interpretation and evidence levels

- Groove: successor endpoint IK/collision gate is uniformly negative.
- Cabinet: successor endpoint start gate is uniformly positive.
- Dense joint interpolation often intersects conservative geometry; it is not a
  cuRobo trajectory and is not upgraded to a planning failure claim.
- cuRobo: **NOT TESTED — CUDA/NVML unavailable**.
- contact/attachment physics: **NOT TESTED**.
- real execution: **NOT TESTED; no device was addressed**.

The task-family split indicates the current endpoint proxy is dominated by the
target geometry. It does not yet justify contextual RL or any claim that the
ARES-R heuristic is inferior to an oracle.

## Next GPU oracle: bounded estimate

Minimum increment after driver recovery: replace only the CPU dense-path
descriptor with the existing persistent cuRobo segment validator, retain the
same contracts/seeds/candidates, and add attached-object collision. No policy
training is needed. For 24 candidates, a first qualification is approximately
48--96 segment plans (one or two successor segments per candidate), expected to
fit comfortably below one RTX-3080 GPU-hour after warm-up; actual timing must be
measured. Only if that oracle creates reproducible within-scenario label
variation should a contextual ranker or RL study begin.
