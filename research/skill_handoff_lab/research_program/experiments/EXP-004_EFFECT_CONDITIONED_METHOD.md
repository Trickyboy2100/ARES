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
