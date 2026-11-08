# 唯一论文主张—证据矩阵（不因实验数量自动更新摘要）

当前工作题目：
**Task-Conditioned Executable Skill-Parameter Reinforcement Learning
for Adaptive Contact-Rich Robotic Manipulation**。
研究领域/标题可以根据真正的正机制收缩，不能据未实现功能扩张。

| 欲写成论文的主张 | 现有支持 | 必要的新增验证 | 状态 |
|---|---|---|---|
| 名义规划与物理接触结果存在可测差距 | 历史P2、Stack、Charger真实物理案例 | 新任务下强名义控制与可重复失败分类 | PARTIAL |
| 任务条件化合法动作能改变完整物理结果 | Pyramid 9候选和Charger同后验参数试验 | 新独立场景、同动作复跑、因果机会与强规划比较 | PARTIAL |
| **我们提出的RL新机制** 比普通RL/规划更有效 | 尚无 | 同动作/同obs/同数据预算、独立seed与关键消融 | **UNVERIFIED** |
| 执行效果表示的作用是独立的 | 初步物理效果记录，历史输入消融无收益 | 匹配history，真实效果残差参与动作决策；新条件 | **UNVERIFIED** |
| 方法在不同任务机制和未见条件泛化 | 原有两个独立环境各有普通基线结果 | 新方法跨第二接触任务、OOD/传感误差、复现实验 | NOT_TESTED |
| JAKA/ARES-R适配可用 | Isaac研究初步基础、EG2仍有mimic问题 | 正确抓取物理验收，模拟接口与全任务实验 | NOT_TESTED |

禁止将局部参数例图、oracle上界或已看开发集包装成新方法因果证据。
论文用表/图应由真实实验生成，引用与来源见EVIDENCE_LEDGER/PRIOR_ART。

**撰稿与研究同步**：现在更新Problem、Related Work、Method和初步结果，
逐条标明已测、待验证及负证据；不以独立新算法收益作为写作门槛。投稿前追加严谨统计、
鲁棒性/泛化和完整工程限制；不凭“模型/视频/PDF交付完成”宣布可投稿。

## 2026-11-05：执行表示分解的具体主张

| 主张 | 证据与限定 | 状态 |
|---|---|---|
| 实测参考刷新与局部残差是不同操作 | cc0a0ff严格前缀探针；3mm与21.98mm含参考重建差，非数值倍率 | SUPPORTED_MECHANISM |
| 继续/刷新/相对active-target残差可分离实现 | 同强规划、真实抓持、原生100步；0继续当前队列；局部目标仍触发完整重规划 | IMPLEMENTED |
| 分离接口改善在线RL | remaining_target×SAC/成功BC终点6/16、8/16；旧格6/16、9/16，未见优势 | NEGATIVE_AT_64 |
| 多阶段适应 | 前段三抓持参数与后段反馈均有真实物理决策；跨任务复用尚未验证 | PARTIAL |
| 新自然条件中的能力保持 | 冻结模型、新32初态：同学习器14/12→17/18；退化5→2、7→1；0救回，名义19最强。两随机重复共享神经初始化，后16条件为自适应扩展 | PRELIMINARY_RETENTION |
| 重规划可在零目标偏移下破坏完成 | 759205相同94步前缀：继续成功、零偏移重规划失败；另4自然条件未改变成败，不推断普遍发生率 | SUPPORTED_COUNTEREXAMPLE |
| 保持剩余队列进度的残差参照 | 尚未实现或训练；须共享IK/碰撞/动力学检查与原生预算 | NEXT_IMPLEMENTATION |

标准SAC、成功BC、Option续接和task-frame residual均不单独主张原创。
潜在论文贡献是可检验的执行表示分解及其对真实多技能学习的影响，最终范围由结果确定。

## 2026-11-05 atomic_handoff：新增与修订

| 主张 | 实测证据 | 状态 |
|---|---|---|
| 单次原子刷新足以恢复旧规则能力 | 760213单次与顺序均失败；重复phase成功 | FALSIFIED_SINGLE_ACTION |
| 时间权限影响实际可执行性 | 76步共同前缀，接触87 vs98；标准参数Option复现100步见证 | SUPPORTED_MECHANISM |
| 学习器获得真实有效救回 | 761202及冻结762214取消干预失败、单次干预成功；762213存在后续替代介入窗口 | SUPPORTED_CASES |
| 新表示带来稳定净性能优势 | 新24：候选10/12、名义12、强规则15；2独立NN | NOT_SUPPORTED |
| 进度保持修正形成可用能力 | 10次全拒绝，8次有限差分加速度检查；并非实际电机加速度判据 | PARTIAL_NEGATIVE |
| 新的多阶段协同 | 三抓持真实训练；确认唯一非grasp1案例两侧消融均成功 | UNVERIFIED |
| 第二任务学习迁移 | 原生Charger16/16成熟控制，小样本饱和，未训练候选 | NOT_TESTED_TRANSFER |

旧分数、旧结论及固定final64终点均保留。当前贡献材料是可复现执行语义、因果物理机制、真实学习救回与泛化限制，不宣称新SAC损失。


## 2026-11-05 rule_option

| Claim | Actual evidence | Status |
|---|---|---|
| Conditional complete-Option adoption improves on the frozen rule in the new cohort |24 natural conditions:19/18 vs16; fixed-first16 with4rescues/4regressions; improvement already in supervised initialization | PRELIMINARY_LEARNED_INTERFACE |
| Online relative policy-gradient adds independent benefit |Same per-scene confirmation outcomes as initialization; supervision18/19 and rehearsal19 | NOT_SUPPORTED |
| Waiting-decision credit causes the gain |All nonzero ablation updates have one decision;112 training/dev trajectories identical | MECHANISM_NOT_ENGAGED |
| Old behavior Q is reliable for the new continuation |All7 observed harmful Options have positive old Q gaps; Q gate8/16 dev | REJECTED_TRANSFER_OF_VALUE |
| Rule rejection is equivalent to queue continuation |763039:exact79-step prefix; four-step zero-action deviation changes success to failure | FALSE_EXECUTION_ASSUMPTION |
| Learned front–rear cooperation improves |Frontend frozen; fixed grasp1 flips2 confirmation outcomes, net equal | UNVERIFIED_JOINT_GAIN |
| Tool recovery transfers |Corrected continuation retains nominal; none3/8 feedback3/8 press2/8 | CONTROL_FIX_ONLY |

No population-level safety or SCI-level superiority follows from these small exploratory cohorts. All positive/negative models and natural scene groups remain visible.


## 2026-11-06 claim-specific update

| Claim | Status | Evidence boundary |
|---|---|---|
| Real learned predecessor + recurrent successor | IMPLEMENTED / TRAINED |448 own episodes, current SkillResult, no object reset|
| Conditional physical grasp effect | OBSERVED |16 design scenes; stronger successor reduces opportunity|
| Alias Q inconsistency | VERIFIED LOCAL |four values, two full physical alias pairs; correction has no gain|
| Joint online independent advantage | UNVERIFIED |three revisions do not improve warm initialization|
| Old learned baseline generalization | PRELIMINARY |32 fresh natural scenes;19/20/21 rule/init/oldRL, small effects|
| Second-task transfer | NOT RUN |historical Tool/Charger review only|

[Complete evidence](../evidence/EXP-004/2026-11-06/joint_continuation/README.md).


### 2026-11-07 initialization erratum (historical scores unchanged)

Direct checkpoint comparison shows joint_continuation sac_31901 and sac_31902 INITIAL Actor, Critic and target tensors are identical: the shared Gate constructor resets Torch RNG to31801 after the requested seed. The two runs have distinct exploration RNG, **not independent neural initialization**. Earlier independent-new-network wording is superseded. Their actual outcomes and negative evidence remain unchanged. The next full-catalogue learner reseeds after loading shared assets and verifies different hidden/Critic weights while keeping cross-arm initialization matched.


## 2026-11-07 completed-cycle claims

| Claim | Evidence | Status |
|---|---|---|
| Omitted legal physical opportunity exists |16 design states,256 reset executions; finite14 vs best fixed12 |LOCAL_PHYSICAL_EVIDENCE|
| Expanded catalogue caused learned development gain |restricted deployed proposal support also21/24 |NOT_ESTABLISHED|
| Stable ordinary online learning asset |two independent neural seeds21/24; confirmation chart_state_32201 19/32; old_rl 22/32; rule 20/32; sac_32201 19/32; sac_32202 22/32; supervised_32201 18/32; warm 21/32 |PRELIMINARY_BASELINE|
| Reference/offset factor improves policy |20/24 vs flat21 |NOT_SUPPORTED|
| State-conditioned target chart improves policy |19/24 vs flat21; confirmation in evidence |NOT_SUPPORTED_IN_CURRENT_COMPARISON|
| Mean-Q deployment is a stronger default |rear-only13/24 vs Actor21 |REJECTED_LOCAL|
| Necessary front/rear synergy |765010 separately rescued by grasp; no factorial necessity |UNVERIFIED|
| Independent novel RL contribution |mature precedents; no consistent candidate advantage |UNVERIFIED|

No unchanged historical scores or sealed tests were relabelled.

| Claim | Evidence | Status |
|---|---|---|
| Additional deployed capability under expanded authority |Frozen SAC32202 loses765625/765627 when masked to old proposals; also avoids regression765604 |LOCAL_EXECUTION_DEPENDENCE|

The above post-hoc evidence updates catalogue-dependence for new conditions only; the earlier seen24 support ablation still ties21/24. It is neither novel coordinate-method gain nor a retrained restricted-policy comparison.


## 2026-11-08 continuation-retention claims

| Claim | Actual evidence | Status |
|---|---|---|
| Reference-bound physical comparisons and actual own-policy training |189 shared +515 refresh;1536 own episodes;64/128 checkpoints; independent initializations |IMPLEMENTED_AND_EXECUTED|
| QP finite witness preservation |580/570 final stored nodes, zero choice violations,5120/5120 accepted updates each |SUPPORTED_FINITE_NODES_ONLY|
| QP improves complete native tasks over same-budget matureRL |17/16 versus17/17SAC on32 new scenes; no rescues forQP |NOT_SUPPORTED|
| Frozen first-localZ-minus capability |24/32,9 rescue/3 regression againstwarm18; selected before outcomes |SUPPORTED_SCOPED_CONTROL_RESULT_NOT_RL_GAIN|
| Strongest existing learned reference retained |PriorSAC21/32 under commonfront; different historical data budget |REFERENCE_ONLY_NOT_MATCHED_BUDGET|
| Multiple necessary rear interventions demonstrated |All6 positive branches reach native success before release |NOT_ESTABLISHED; success maintenance still matters|
| Real second-task predecessor dependence |Opposite15/30 outcomes on2 natural roots,8 complete repeats |SCOPED_PHYSICAL_EVIDENCE|
| Second-task rear adaptation or learned transfer |Matched feedback=nominal;9 legal probes0rescue; no new network |UNVERIFIED|
| Independent SCI RL method advantage/safety theorem |No positive controlled performance or theorem |UNVERIFIED|
