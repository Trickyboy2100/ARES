# STAP algorithm-level reproduction and scope

Primary source: [official STAP](https://github.com/agiachris/STAP), commit `0fe0dcbd5ede5ce4ee3d952642f70b5550a0813c` (MIT; retained LICENSE). No robot/site data.

| Original mechanism | Executed local mechanism | Evidence |
|---|---|---|
| Primitive sequence and legal action spaces | PICK_TOOL grasp x 70–105 mm; USE_TOOL hook y −75…−55 mm; actual stable-grasp boundary | Actual decisions and SkillResult in each trajectory |
| Policy-initialized CEM | Learned per-skill actors initialize Gaussian mean; dimensions scale population | `program_stap_kernel.py` |
| Dynamics rollout | Learned physical post-grasp state; actual measured state replaces prediction before second decision | Saved predicted states and actual observations |
| Skill Q product | exp(sum(log(Q_i))); Q estimates actual stable-prefix / terminal task success | Original and port outputs exactly matched |
| CEM updates | Clipping, elites, retained elites, population decay, momentum, unbiased std/minimum std, best across iterations | 12 exact fixtures, full candidate/state/value streams |
| Posterior replanning | Re-run suffix CEM at measured stable-grasp boundary | Explicit open-loop comparison, matched first-stage RNG in paired runs |

`official_kernel()` executes original AST algorithm nodes from original Planner, CEMPlanner and evaluate_trajectory, with minimal API adapters. Package imports/type-resolution and unbatched wrapper are replaced; algorithm statements are not rewritten. Inputs are explicitly batched. Fixtures cover 1/2 parameter dimensions, 2/3 stages, two seeds and two CEM settings. Equal-dimensional actions per stage only; ragged spaces and the complete original stack are not validated.

The actual original package import was attempted and failed on absent legacy `gym` (`OFFICIAL_IMPORT.txt`). We did not install an incompatible Python3.8/PyBullet stack or claim its benchmark/checkpoint reproduction. Original algorithm execution on deterministic fixtures plus a local physical-task port is the achieved scope.

## Physical port and learning limitations

Every compared method uses the same actual controller, native100-step horizon, simulator feedback, stable-prefix contract and legal action ranges. `ARES-ToolTarget-v0` adds a declared40mm target tolerance AND retains native PullCubeTool success. It is a named task variant, not the unmodified official benchmark. Known mass and target metadata are common observations. No future success/state enters a selector.

Two stages have one scalar parameter each. Stage0 Q predicts stable-grasp success. Stage1 Q predicts real terminal task success. Dynamics predicts the post-grasp state; terminal state after stage1 is unused by Q-product scoring and represented by identity. The policies are trained Q-maximizing initializers, not official pretrained task-agnostic policies. No original STAP benchmark performance claim.

Fitted Q is finite-horizon offline value learning: stage1 receives actual terminal labels; stage0 backs up the maximum learned stage1 value at the **actual** post-grasp state. Foresight is a state-based fixed-suffix analogue: its downstream predictor uses only actually executed auto-geometric suffix labels, and that unchanged suffix is used at deployment. It is not a VLA reproduction or a fully trained online Residual SAC/PPO baseline. Both receive the same dataset, but Foresight uses16 policy-matched labels rather than all64 suffix labels; this algorithmic distinction is disclosed.

Dataset64 comprises two scene seeds, two masses, two grasps, two targets and four hook policies. Multiple suffixes share essentially the same prefix;64 episodes are not64 independent prefix contexts. Single training seed31103. Each model version uses800 shared model optimizer steps,800 per first-stage value variant and400 total actor initialization steps=2800 optimizer steps. The residual-dynamics repair is an ordinary common-model inductive bias, not a new method. Candidate contract/generic auxiliary code exists but was NOT trained or evaluated: the required independent physical opportunity is unproven.

## Cost definitions

Historical `skill_model_queries` counts rolled-out skill stages including the policy initialization. It is **not** the exact number of neural forward evaluations. `analysis/EPISODES.csv` separately recomputes individual-sample critic, learned-dynamics and actor evaluations from saved CEM populations. For default16 samples/dimension and4 iterations, a complete posterior-replanning episode uses239 critic,97 dynamics and3 actor evaluations. Strong geometry uses3 yaw candidates for a selected hook or9 yaw/hook candidates for the auto suffix; actual search/planning wall times and candidate counts are retained. Fitted Q uses33 value-grid candidates per decision; it does not receive free physical search. One selected parameter sequence is actually executed; CEM candidates add no physical episodes.

The high-budget exploratory run used a continuous RNG stream across policies. It supports descriptive results only, not a clean replanning ablation. `absolute_paired16` and `residual_paired16` reset the same per-case RNG before each policy; their first decisions and prefixes are checked exactly before interpreting differences.
