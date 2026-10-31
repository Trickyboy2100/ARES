# EXP-002 → bounded EXP-003: useful actions, inadequate learned selection

**Scientific decision:** retain the paper question, but do not promote the current
finite-window geometric effect feature to a new method. Actual contact corrections
can rescue full tasks; the present short SAC pilots do not beat strong search.
The clearest causal evidence is unnecessary Z authority: fixing only that channel
at the same trained checkpoint rescues six cases without regression. This is an
interface diagnostic, not a newly trained policy or an RL innovation.

## What was actually executed

All episodes execute native grasp/lift/alignment and contact physics to step200,
with no mid-episode object teleport or state restoration. RTX3080 GPU PhysX,
CPU SAC updates, single GPU process at a time, offline network namespace. No
ARES-R writes or hardware operations. Development conditions are four scene
seeds729000–729003 × three observed-socket biases; they are **not 12 independent
scene draws**, a sealed test, or physical-robot measurements.

* Fresh native PlugCharger seeds727000–003: strongest frozen control3/4;
  remaining failure is grasp, not a residual insertion opportunity.
* Named `ARES-AdaptivePlug-v0`: native geometry/initialization/success/200-step
  horizon, explicitly varied density/material and socket observation bias.
  Moderate density0.5/2× plus friction0.2/1.0: all12 design runs succeed under
  geometry/search/RLS. This does not justify learning dynamics adaptation.
* Bias±1.2mm creates actual conditional correction effects: opposite fixed
  offsets reverse their value. Early source versions used privileged native
  stopping and are segregated; the final comparison uses observed-geometry
  stopping. Do not pool these versions or call their scores official native.
* Two genuine1024-decision pilots, ordinary SAC and effect-feature SAC,
  each896 actor/critic optimizer updates, new training scenes728000–031.
  Four raw observations plus six zero/derived feature slots; no hidden bias
  or true goal input. Physical budgets differ9000 vs8800 steps and are counted.
* Five real training runs in total: ordinary SAC, effect SAC, LayerNorm SAC
  at alpha0.1/0.02, and matched effect+LayerNorm. Each1024 skill decisions and
  896 actor/critic updates; all trained weights and costs are preserved.
* Known critic LayerNorm and an action-channel ablation follow the observed
  failures. Their final separate results/costs are in the machine-readable
  analysis; none is a new algorithm claim.

Total this-turn recorded cost: **426 complete200-step trajectories,85200 physical
control steps,36069 prefix steps,4480 actor and4480 critic optimizer updates**.
Five training-tail trajectories are counted as physics but excluded from policy
success scores; failed initial training and deduplicated recovery are retained.

## Frozen first comparison and causal follow-up

| Controller/policy | Full-task terminal success /12 |
|---|---:|
| Geometric depth-hold | 4 |
| Online RLS + contact search | 6 |
| Bounded progress-based contact search | **10** |
| Ordinary residual SAC pilot | 1 |
| Finite-window contact-contract feature SAC | 3 |
| Same feature policy, prior-only inputs | 3 |
| Same ordinary SAC, Z action set to zero | 7 |
| SAC + critic LayerNorm, matched alpha0.02 | **10** |
| SAC + critic LayerNorm, alpha0.1 protocol deviation | 4 |
| Same LayerNorm/alpha0.02 + effect features | 2 |
| Same effect+LayerNorm checkpoint, prior-only | 2 |

Search vs geometry: six rescues, zero regressions. Feature SAC vs search:
one rescue, eight regressions. Feature versus prior-only: identical twelve
binary outcomes, despite updated features on132/316 evaluation decisions.
Features activate on110/1024 training decisions. This is **no demonstrated
benefit of execution-feedback updates**. Both agents already see the same
short execution history, so derived features add an inductive bias, not
new information. Their weights are compatibility scores, not calibrated
posteriors or success probabilities.

The known critic-LayerNorm baseline at matched alpha0.02 reaches10/12 after
the same1024 decisions/896 updates; the earlier alpha0.1 variant reaches4/12.
Against search it rescues two and regresses two; the same total hides
complementary failures. Four cold repeats of geometry/search on seed729000
with ±1.2mm bias reproduce all200 recorded steps exactly for the checked
state/action/force fields. This limited reproducibility check is not universal
GPU determinism.
This is a useful ordinary-RL optimization result, **not an ARES method gain**.
Do not isolate LayerNorm using the alpha0.1 result or compare raw Q values
across these independently trained critics. Both remain single-training-seed
limited-budget pilots, not converged or sealed-test results.

The subsequent **matched LayerNorm candidate test** also fails to demonstrate
an advantage: effect+LN2/12, same checkpoint prior-only2/12, against ordinary
LN10/12. The candidate uses identical critic normalization, alpha, network
size, decision budget and raw history. This closes the weak-optimizer confound
for this pilot; it does not establish a universal negative across training
seeds or longer budgets. Do not scale this feature mechanism on these results. On96 activated
effect-LN evaluation decisions, the inferred hidden-offset vector RMSE is
1.016mm versus0.995mm for the zero prior; sign is correct in21/66 nonzero-bias
decisions. These correlated, policy-selected samples are an offline diagnostic,
not a calibrated probability test. The features have not shown better
identification here, rather than merely failing to influence the actor.

Zero-Z versus ordinary SAC: six rescues, zero regressions, same saved checkpoint
and recorded physical prefix. Compared with search it rescues two but loses
five: useful complementarity, not superiority. The intervention is post-hoc
on development data. A reduced-authority trained agent still needs a new,
fair comparison; this result does not establish generalization to Z errors.

## Evidence you can inspect

- [New same-initial-state rescue video and synchronized measured/truth curves](new_video/SYNC_VIEWER.html):
  geometry16.075mm failure versus search3.473mm success; full201frames,
  RAW/ANNOTATED switch and contact-detail zoom. [Portable raw traces and QA](new_video/README.md).
- [Synchronized historical control mechanism video](video/SYNC_VIEWER.html):
  full RAW/ANNOTATED videos, exact201frames/20Hz, browser click/seek QA.
- [Scientific analysis](analysis/SCIENCE_NOTES.md), [scores and paired outcomes](analysis/EVAL_COMPARISON.json),
  [chart](analysis/EVAL_MAIN.png), [exact budgets](analysis/SUMMARY.json).
- [Episode/trajectory SHA index](analysis/EPISODES.json), [source index](analysis/RUN_INDEX.json),
  [prefix comparisons](analysis/PREFIX_PAIRS.json), [QA](analysis/QA.json).
- [Frozen critic/action ranking diagnosis](CRITIC_RANK.md): real weights prefer
  harmful corrections; soft-Q ranking is not a success probability.
- [Closest papers/code and license audit](PRIOR_ART_REVIEW.md),
  [method/fairness review](RESEARCH_REVIEW.md), [physical design](PHYSICAL_DESIGN.md).
- [Trained checkpoint index](models/INDEX.json), [reproduction](REPRODUCE.md).

Video source type is **MEASURED_POSE_REPLAY**, reconstructed from actual saved
simulation states, not online camera capture or another experiment. Rendering
adds zero physics steps. No artificially generated contact imagery.

## Next scientifically discriminating experiment

Use EXP-003 to establish a capable same-authority baseline before promoting
EXP-004. Compare a task-justified lateral-only interface with full-YZ authority
on newly separated Y-only and YZ-error distributions. Keep strong search,
nominal advance and actual prefix costs. Test persistent contact probes and
measured settle/hold contracts, including transient-success loss, rather than
feeding increasingly complex representations into an underperforming actor.
A candidate must retain search's solved cases and exploit its residual failures;
posterior activation or a better mean than weak SAC is insufficient.

This session does not establish calibrated hidden-parameter identification,
mature RL convergence, cross-seed robustness, industrial realism of bias ranges,
SCOUT/DAWN reproduction, independent algorithm gain, or site readiness.
