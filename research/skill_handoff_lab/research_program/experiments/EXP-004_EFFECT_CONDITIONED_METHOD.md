# EXP-004｜从多阶段技能实际物理证据中选择一项原创RL机制

**状态：RUNNING；研究阶段S5；与EXP-003共享执行基线并行推进。2026-11-03已训练与物理验证候选，独立增益UNVERIFIED。**

## 不锁死为之前已失败的效果特征

当前49假设/6维几何兼容分数已在单训练seed、相同LN和alpha的对照
中得到2/12，且关闭反馈更新仍2/12；偏差推断不优于零先验。
因此不要简单延长它的训练或视为唯一论文候选；SCOUT已研究
动作效果信念，DAWN已研究ResidualRL的Critic归一化。

**论文主问题**是任务条件化的跨技能参数自适应，低层接触只是部分执行。

## 允许Codex根据EXP-003物理因果证据选择（而非同时堆）

**H-A：后续技能条件约束的动作权限与参数学习。**
把`next_skill_preconditions`转为实际参数合同，在当前与后续技能的
不同任务条件下选择合法且有效的参数/动作子空间。优先对照固定mask、
规则投影、相同mask普通RL、不同任务下的全维RL。
仅“禁止Z修正”不够原创，学习出的状态依赖选择必须真正有收益。

**H-B：物理结果条件化的后续可执行价值。**
学习`Q(o_k,xi_k,postcondition,Contract_next)`与实际下一技能
可达/成功的关系，并把真实释放/稳定/抓持后果通过SkillResult
反馈给下一技能动作。需对照普通两边界SAC/PPO、STAP技能Q/搜索、
Foresight后续成功预测及监督ranker。同一条件下必须证明超出
通用downstream-Q思想的具体算法/数据优势。

**H-C：部分可观测执行效果信念（有条件）。**
只有任务存在无法从当前合法观测判别、且可通过真实动作-结果
辨识的抓持偏差/接触状态，才研究POMDP belief/主动小探针；
对照通用history、RLS、SCOUT和全真值oracle。无隐藏状态就不要
为了理论深度强行建POMDP。

所有候选是可反证的学术假设，**没有自动原创性**。
先在有物理学习空间的多阶段任务上做少量A/B试验，
根据最强的实验证据确定唯一核心方法，并用匹配网络容量、
动作、obs、奖励、名义控制和额外物理/搜索成本的消融排除解释。

## 成熟论文方法必须给出的因果证据

- 真实前段SkillResult如何改变后段动作及真实任务价值；
- 不同目标/工具/可执行条件下，策略如何选择不同参数；
- 至少2个操作机制或明确不同下游合同的任务子群里，有独立未见
  条件的可重复收益；同任务多seed统计和rescue/regression；
- 不可只用任务更难、mask更多、训练样本更多、调好了stop或
  强基线不一致来解释增益；
- 近邻算法（RAPS/STAP/Foresight/Fabrica/SCOUT/DAWN等）不仅
  在Related Work引用，至少一个最接近功能对照应实际复现或
  制定诚实可比的实验条件。

## 当方法不成立时的行为

若后续合同Q与普通RL效果相同，承认该想法没有独立价值；
若隐藏状态并不可辨识，退出POMDP方向；
若强搜索解决所有局部案例，转向真实跨技能依赖；
保留所有负结果并自主迭代，不通过降低名义基线制造“论文正结果”。


## 2026-11-03 — Parallel native Peg physics and trainable skill-value mechanism

[Complete evidence and code](../evidence/EXP-004/2026-11-03/peg_value/README.md). Active EXP-004 now runs alongside EXP-003 common execution. Official GPU Peg OBB helper used stale CPU world pose; preserve literal baseline and adapt local collision geometry to actual GPU pose. Corrected official timing1/6, explicit shared retiming3/6. Frozen-control two-boundary panel:60 trials /10 natural scenes, strongest fixed6/10, measured union8/10; same-prefix rescues and regressions are real. Prototype: paired return-difference loss, same-data FQI/absolute/suffix-value/STAP-style enumeration, discordant sampling ablation, then ordinary masked LN-SAC with paired/absolute/shuffled critic gradients. Historical early closed-selection40/69 vs ordinary32/69 largely matched generic labels; later43/69 vs44/69. Fresh actual Tool all3/6; Peg all2/4. Independent algorithm gain UNVERIFIED. Retain actual models, all failed trials, source/hash/buffer evidence and exact-frame video. Costs:160 complete100-step episodes including6 invalid instrumentation terminal holds;75600 joint value updates and51600 actor/critic updates EACH. Native horizons and success unchanged; no sealed data or ARES-R edits. Next causal question is executable contact tracking/budget under the shared Peg controller; reconstructed target is correctly +50mm along goal x, so do not re-debug a presumed reversed frame or merely train another grasp score.


## 2026-11-03 — finite physical Options, acquisition revisions and extraction mechanism

Source: [evidence/EXP-004/2026-11-03/discrete_handoff/README.md](../evidence/EXP-004/2026-11-03/discrete_handoff/README.md); code `2ac433e`, base `1ea0965`. Independent algorithm gain remains **UNVERIFIED**.

- V0 continuous projected SAC has off-support Actor/Critic targets. V1 implements exact legal finite options, real SkillResult, duration, termination and remaining budget; DSAC/FQI/IQL/CQL/supervised/functional-STAP fixtures and physical trace checks pass. This is mature algorithm infrastructure, not originality.
- Actual offline→online study uses compatible60-episode Peg prior, fresh natural training scenes, two optimizer seeds and native100.38 model histories exactly reconstruct from their actual physical buffers.2458 full episodes/245800 steps include all training/evaluation/posthoc confirmation;250400 primary critic optimizer iterations exclude audit reconstruction and toy fixtures.192 pairs reexecute and charge both full prefixes.
- Expected-update acquisition was implemented, physically trained, then revised for wasted duplicate coverage, correlated suffix outcomes and current-task anchor distribution. Joint outcome prediction improves old-scene leave-one-scene-out counterpart Brier0.313→0.182; covariance changes7/16 actual acquisition rounds, but does not yield a reliable policy gain.
- Eight predeclared fresh development scenes: original joint and EES-style4/8 for both seeds; current-anchor revision2/8 and4/8;128-episode ordinary SAC5/8 and3/8. Same-history mature supervised/FQI5/8 for both. Best fixed6/8 is a hindsight comparator, not a training-selected policy. Across14 development scenes, fixed maximum10/14 versus finite measured union12/14 preserves a conditional-adaptation opportunity.
- A further actual intervention changes only extraction of frozen Q values: current-joint seed31301 Actor2→minQ4→meanQ5; mature seed31302 Actor3→minQ2→meanQ5.48 complete native rollouts confirm it. Mean-Q rescues3 and2 respectively with0 regressions; min-Q causes1 regression in mature seed31302. Actor approximation and clipped-Q aggregation are distinct. This is explicitly posthoc development diagnosis, not held-out performance or an independent RL contribution.
- The acquisition surrogate scores greedy decisions of one critic while deployment uses a learned Actor. Positive scores do not establish actual return gain; current-anchor seed31301 has observed update returns4→4→3→4→4→3→3→3→2 on the retrospective real branch panel. Retain this falsification and align any future mechanism with the actually deployed policy before expanding it.
- Permanent assets: finite Option runtime/learners; compatibility filter; exact optimizer/prefix audits; complete real branch evaluation; actor/min/mean diagnostic; trained checkpoints and RAW/English videos with frame-synchronized curves. EES and STAP comparisons are functional adaptations with disclosed limits, not claims of full original benchmark reproduction.
- No sealed data, hardware, ARES-R modification or restricted site data. New acquisition variants were frozen before reading expansion outcomes; later extraction confirmation is labeled posthoc. Next work is shared-baseline policy/value consistency and genuinely new development validation, not relabeling a standard extraction fix as novelty.


## 2026-11-04 — Natural coverage, value improvement and measured handoff targets

[Complete evidence](../evidence/EXP-004/2026-11-04/value_improvement/README.md). From60760b8, actual96 new natural initial states and822 native100 episodes/82200 steps;374400 primary critic iterations,268 checkpoints. Shared mature finite VIAC/USAC/BEE/IQL/FQI/CQL/STAP/conditional-scoring controls; no renaming mean-Q or empirical DP as novelty. Original3x2 training fixed24/32 vs union25/32; physical split/refreshed option adds one reproducible joint rescue, giving26/32 but no development union increase: both21/32 vs hindsight fixed19/32. All positives, regressions and exact prefixes retained.

Candidate substitutes physically measured suffix values at exact SkillResult into predecessor backups; complete-coverage revision changes only activation on full legal successor support. Offline3-option original14/32,13/32; revision15/32,17/32. Six true32-episode online runs: revision19/32,19/32; ordinary mean-critic SAC19/32,19/32; exact value improvement18/32,19/32. Candidate vs SAC1 rescue/1 regression, then0/0.48 additional actual final-policy rollouts exactly match their complete100-step physical branches; other32-scene model scores are explicitly logged physical-branch evaluation. This is catch-up, not independent gain.

New diagnosis: legal newly introduced actions lack historical execution support. Ordinary SAC terminal new-action Q exceeds1 on7/29 distinct old handoffs per seed; CQL removes this audited overflow. Same-data FQI15/32,18/32 versus CQL18/32,18/32, still not an independent method. IQL18/32,19/32. Neither calibration nor an isolated control rescue proves a paper contribution. Development now seen; complete-coverage/CQL are disclosed iterative revisions, not sealed validation.

RETAIN strong shared controller, explicit action-schema compatibility, finite RL controls, all real data/weights and shared split primitive. LOCAL_NEGATIVE measured-backup/coverage revision. Next investigate a physically legal broader predecessor parameter domain with explicit grip margins and the nominal option retained; require opportunity beyond strongest fixed/search control, keep conservative/support-aware baselines, and do not expand this failed backup unchanged. Method status remains UNVERIFIED. ARES-R/read-only and hardware/network isolation preserved.


### 2026-11-04 — Persistent closed-loop successor research

See [closed-loop evidence](../evidence/EXP-004/2026-11-04/closed_loop/README.md) and latest DECISIONS. The predecessor remains task-conditioned; successor now updates bounded parameters repeatedly under native100. Ordinary/scalar/budget matched training and local curriculum are real; mature retained SAC remains10/16 versus new final5/16,4/16,6/16. Deployment-label revision9/16,7/16,7/16 does not establish a budget-method gain. Six proposed recovery schedules give zero6/6,rule6/6,mined4/6 under exact native prefixes. Keep this closed-loop asset and investigate stable feedback learning, not more fixed-branch scoring. No sealed tests; independent gain and new beneficial operation capability remainUNVERIFIED.


## 2026-11-04 — Successful-behavior retention and controlled exploration

[Complete real evidence and reproducible artifacts](../evidence/EXP-004/2026-11-04/success_retention/README.md). Fixed actual predecessor, same Peg100/feedback actions. Four32-episode SAC branches followed by4000 replay updates:8/16,9/16,10/16,9/16; nominal9/16. Controlled SAC2 rescues/1 regression. Same-replay whole-episode nominal-success anchoring restores9/16 with0/0 on both behavior datasets; weight0.05 returns1/1. Posthoc strong phase/budget feedback12/16,3/0 is control progress, not RL originality. All trajectories/models/source SHA retained.505 full episodes/50500steps,41120 primary update iterations,1336 training successor decisions,3841 planner queries;9 exact native repeat trajectories.

Retain both nominal-retention and learned-recovery endpoints, stronger feedback reference and all negative variants. No task/runtime/horizon change or sealed-test expansion. Next test stronger-reference real training data as a shared learning anchor; do not expand probability heads or repeat the same coefficient search. See DECISIONS and STATE for the continuing experiment. **UNVERIFIED independent method gain.**


## 2026-11-04 — Intervention credit and changed continuation

[Full real evidence](../evidence/EXP-004/2026-11-04/intervention_credit/README.md). Fromcc6774c;616 complete native episodes/61600steps,9480 primary Actor/Critic iterations,4543 planner queries. New24dev nominal17, frozen rule18; fixed-pair SAC10/AWAC-style17/measured-gap16. Current-policy paired acquisition with SAC/AWAC/gap/direct preference all17; same-data memory consolidation both17. Budgets equal within groups only; one inherited neural initialization, no independent gain.125 measured-prefix/queue-verified contrasts; no PhysX restores.

RETAINED: real online intervention-credit trainer, same-data mature AWAC-style control, exact queue evidence and all weights. Concrete positive training contrast757300 action1 at59; prior action probability1.8e-6 attenuates expected-SAC gradient. Direct signed log-odds learns that first intervention but complete native reexecution still fails through later replans; first69 actual steps equal the successful witness. Thus failure is not merely missing action selection. Frozen continuation credit does not automatically justify modifying the shared continuation policy. Memory repetition adds no development gain. This is a local negative method result with a specific policy-improvement defect, not another prediction-head proposal.

Next continue this mechanism through continuation-preserving policy improvement versus mature trust-region and successful-witness imitation controls, sharing physical data/authority. Do not relabel pairwise preference, AWAC, Q-filter or KL as originality. Do not return to fixed branch scoring or tune the old strong rule. Preserve native100, actual prefix costs and all failed trajectories; no sealed/hardware/ARES-R edits.


## 2026-11-04 — Continuation-explicit Options and three learning revisions

[Real implementation, models, raw outcomes and review](../evidence/EXP-004/2026-11-04/continuation_option/README.md). Basec4219f2; code758ab2c. New event-terminated reference-continuation Option reproduces all100 physical steps of757300 success witness; phase-only commitment fails. This is mature temporal-interface capability, not originality. Action0 retains the active queue. Ordinary learners and frozen-rule control share the new interface; original frozen rule8/16 versusOption-wrapped rule7/16 shows the contract is not universally better.

Actual three training loops: same-data standard Option-SAC / successBC / measured preference+suffixKL; then joint continuation likelihood; then real SMDP termination-target credit.24 new natural training starts,16 initially new development starts. Final ordinary7/16,7/16,6/16; local4/16,4/16,6/16; joint3/16; SMDP-credit4/16. All learned final variants zero development rescues. Selected training native retakes candidate5/5,BC4/5,ordinary2/5; bothcandidateandBC truly learn758002/004, so local selection is no longer the only failure. Candidate overgeneralizes action5 to12/16 development states; measured terminal deltas and bootstrapped current-tail advantages must remain distinct. No independent gain, convergence, multiseed or sealed-test claim.

Costs484 fullnative episodes/48400steps,48 training episodes/24 exact native-prefix pairs,6400Critic and9600Actor iterations (including3200extra Actor-only),90savedmodels,3847plannerqueries. All old assets and all new failed variants retained. New runtime and learned training repairs are reusable; simple Option commitment,KL,joint preference or SMDP correction are not claimed original. Next isolate timing/state generalization with real reachable boundary interventions and checked current-policy tails; do not repeat unchanged losses or return to fixed-branch scoring. Predecessor remained real grasp1 for attribution; whole multi-skill adaptation and second-task research remain active. ARES-Rread-only; nohardware/sealed/restricteddata. **UNVERIFIED independent method gain.**


## 2026-11-05 — Own-policy learning, exact resume and target-origin falsification

[Evidence, models, complete trajectories and video](../evidence/EXP-004/2026-11-05/online_integrity/README.md). Froma8418f9. Question: can inherited Options train credibly with own-policy data, and is a3mm feedback parameter actually a local execution change? The named ARES_SCI_RESEARCH_TASKBOOK.md was absent; requested its location and proceeded with explicitly authorized work, without claiming it was read.

Implemented immutable source/seed manifests, isolated own-arm replay, atomic complete learning-state transactions, explicit incomplete-physics detection and idempotent resume. An actual collection-before-update interruption matches uninterrupted weights/all optimizers/RNG/replay and200native steps exactly; changedseed rejected. Arbitrary mid-PhysX restoration is not claimed. Restored real learned3-grasp predecessor decisions; retained native100 and frozen shared runtime/reference.

Two64-episode online arms,5120Actor/Critic updates each: ordinary final6/16 (1rescue/4regressions), mature successful-behavior BC9/16 (1/1); nominal9, frozen historical rule8 (1/2). Ordinary intermediate10/16 is retained but not substituted for predeclared final6. Initial shared inherited9/16, one neural initialization; no convergence or multiseed claim. Data are policy-specific, not a shared-data loss-isolation experiment. Actual replay audit958decisions including128front decisions, no dev/other-arm transitions.

Mechanism4natural scenes×3real grasps: zero5/12, refresh±3mm0/12, active-target+3mm2/12, active-target−3mm5/12. Additional no-offset attachment-refresh control0/12 excludes 'only the sign of3mm' as an adequate explanation. Current target changes5.74–21.98mm; relative target3mm reduces commanded-path disruption but adds0rescues.12zero repeats reproduce1200physical steps. Grasp affects downstream outcome (759501:0/1success,2failure); not yet a learned general handoff result.

344full episodes/34400steps,2501queries,10240main update iterations plus16resume-proof iterations. All failures/models retained; exact sources, selected public model/archive SHA, full local checkpoint inventory and one RAW/English synchronized video. Retain the online/resume foundation and opt-in operator as research assets; neither BC nor relative coordinates is originality. Next keep the current cohort immutable and compare the new action interface in a separately versioned online group; require a real residual before introducing another loss. Independent gain UNVERIFIED. No hardware/sealed/restricted-data/ARES-R changes.


## 2026-11-05 — Decoupled execution references: matched online learning and frozen confirmation

[Evidence and reproducible raw archives](../evidence/EXP-004/2026-11-05/decoupled_interface/README.md). Base cc0a0ff; core implementation ba7ad8c. Filled remaining_target × ordinary SAC / successful-behavior BC with64 own-policy episodes each, inherited model, same train/dev,3 learned real grasps and native100. Final seen-dev6/16 and8/16 versus historical measured6/16 and9/16: no advantage. Historical final64 unchanged. Equal additional64 BC episodes and a second128episode stochastic repeat retained all negative checkpoints; frozen dev selection chose64 for all four, while charging128episode budgets.

Frozen new32 natural conditions: measuredBC14/32,12/32 versus relativeBC17/32,18/32; nominal19/32; measured/relative frozen rules17/32,16/32. Relative reduces regressions5→2 and7→1, adds zero rescues. Two stochastic repeats share inherited neural initialization. The second16-condition cohort was declared after examining the first16; this is preliminary development confirmation, not sealed/fixed-sample significance. Own-policy datasets differ by design.

Mechanism: trained policies on759205/grasp1 share94 recorded physical steps. Continue succeeds; relative+3mm replans15points with6native steps remaining and fails. A fully reexecuted zero-offset replan also fails; four additional natural cases show unchanged complete outcomes. Replanning alone can break completion without target displacement; geometry versus retiming/velocity is not uniquely identified. No3mm-to21.98mm numerical-amplification claim.

Costs:512 new training episodes,3467effective decisions,40960Actor and40960Critic iterations (+40960unusedaux);1067complete native episodes,106700physical steps,8764planning queries. RAW/annotated101frame full state-playback video adds0physical steps. Complete raw/model/source SHA retained.

Retain representation-aware online training, model selection, all failures and real predecessor dependence. Next implement a standard progress-preserving local residual on unexecuted waypoints with shared IK/collision/dynamic checks; never compress15points into6 or extend native budget. This comparator is not yet implemented. Paper Problem/Related Work/Method/preliminary Results now updated; representation-retention is preliminary positive evidence, independent newRL and nominal superiority remain UNVERIFIED. NativeCharger200 second-task reuse prepared without incompatible replay mixing. No ARES-R/hardware/sealed/restricted-data changes.


## 2026-11-05 — atomic_handoff：可学救回成立，稳定净优势尚未成立

从9a66e94继承，完成真实own-policy训练320回合、5臂、两个独立NN初始化；共1002原生episode、101800物理步、8934规划查询。
H1单次atomic不足，H2重复阶段刷新可复现760213完整100步成功；实际首次接触87 vs98步。
761202共同63步前缀证明学到的局部动作必要且单独足够；冻结新762214同样成立，762213则有更宽的有效时间窗口。
新24确认：名义12、冻结规则15、queue12/12、macro10/12；后者分别1救回3退化、1救回1退化。两者都高估了762206的破坏性干预。
不将更多决策、标准参数化Option、83维状态增广或动作字典本身称为新算法。H3拼接10次全拒绝，尚无操作能力；第二任务Charger20016/16饱和，无迁移增益。
保留全部模型、失败、完整事务resume和历史分数。重点转为真实可执行续接下的干预价值校准与保持；当前确认集已看，不能复用为新测试。
证据：[atomic_handoff](../evidence/EXP-004/2026-11-05/atomic_handoff/README.md)。完整成本、SHA、RAW/英文同步视频与论文表已归档。


## Current cycle 2026-11-05 — rule_option: learned adoption retained; waiting-credit hypothesis not engaged

From9a30507, implemented and ran single-complete-Option deviation then frozen strong rule. New64 paired training conditions give3rescues/7regressions; all7 harmful outcomes have positive old behavior-Q gaps. Completed512 own-policy training episodes across two iterations, two independent initializations and actual entry-credit/rehearsal controls. Shared500-update pretraining underfit the three positive training examples; common2000-update initialization was corrected before further matched training.

New24 fixed-final64 confirmation: rule16, fixed first proposal16 (4rescues/4regressions), shared initialization19/18, sequenceRL19/18, supervision18/19, ordinary rehearsal19. Each RL model has exactly its initializer's per-scene success outcomes. Thus paired-data conditional adoption is retained as an effective learned baseline; independent online-RL gain is not supported. All nonzero updates in the ablation seed occur at the first gate decision, so full waiting credit and entry-only credit produce identical112 complete training/development trajectories. Do not claim a new stopping-time algorithm from this comparison.

Tool's early-contact adapter had a real continuation defect: declined intervention discarded the live nominal caller, and early modes could intervene before the event. Fixed in a new version without overwriting history. Eight nominal100-step traces and three declined-intervention traces are exactly retained. Corrected early none/feedback/press=3/3/2 of8,0rescues; no second-task RL gain or blind transfer training. Whole cycle:1528 native episodes,152800physical steps,14644planner queries,40named weight checkpoints. Detailed evidence, actual scripts, all models and rawSHA: [rule_option](../evidence/EXP-004/2026-11-05/rule_option/README.md).

Decision: retain the stronger learned initializer and rehearsal controls, execution contracts, own-policy data and Tool repair. Stop expanding the inactive waiting-credit mechanism. Next investigate real predecessor–successor ordering on NEW training initial conditions with this retained learned successor, then a matched ordinary two-stage learner; never reuse the current confirmation as unseen data. No independent algorithm or formal safety claim.


## 2026-11-06 — actual joint continuation learning (adc86dc continuation)

Three joint learning iterations;448 own episodes;1163 total native episodes/116300steps/9857queries. MC mixing did not preserve13/16; shared execution critic7/16 vs flat9/16; paired whole-return PG and SAC both9/16. Frozen prior RL21/32 vs initializer20/rule19 is preliminary, not independent robust gain.

| Arm | Own episodes | Initial → middle → final |
|---|---:|---|
| identity_flat_32001 | 64 | 9 → 8 → 9 /16 |
| identity_semantic_32001 | 64 | 9 → 7 → 7 /16 |
| paired_pg_32101 | 32 | 9 → 9 → 9 /16 |
| paired_sac_32101 | 32 | 9 → 9 → 9 /16 |
| return_mix_31901 | 64 | 13 → 12 → 11 /16 |
| return_mix_31902 | 64 | 13 → 12 → 11 /16 |
| sac_31901 | 64 | 13 → 11 → 11 /16 |
| sac_31902 | 64 | 13 → 12 → 10 /16 |

Shared physical authority and frozen-final selection within each comparison. First revision has two new-network/exploration seeds sharing pretrained weights; later revisions single seed. New data are development/confirmation, not sealed tests. No pooled cross-cohort gain. Actual two-component rescue764712 is reproduced by either component alone. Action alias equivalence is physically verified but enforcing it does not improve policy. Whole-return PG avoids injecting fixed-suffix credit into changed continuation, yet matches ordinary SAC and mostly preserves initial behavior. No convergence or independent novelty claim.

Evidence: [evidence/EXP-004/2026-11-06/joint_continuation/README.md](../evidence/EXP-004/2026-11-06/joint_continuation/README.md). Next: Test omitted legal physical Options at real handoffs to determine whether the frozen two-proposal interface leaves repeatable recovery opportunity. Share any expanded authority across warm baseline and ordinary joint RL; do not expand unsupported MC-mix, alias-critic or paired-PG mechanisms. Preserve native Peg100 and sealed isolation.
