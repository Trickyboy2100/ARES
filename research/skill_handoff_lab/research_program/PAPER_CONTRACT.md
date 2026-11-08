# 论文科学合同：一篇论文、一个核心问题

## 1. 稳定不变的研究目标

**任务条件化的可执行技能参数强化学习**：在已存在合理运动规划和名义
操作技能的系统中，针对新工件、目标或真实接触偏差，学习怎样在合法合同内
调整技能参数，并基于物理执行反馈改善完整任务的成功、可靠性与泛化。

工业背景与甲方主题《多模态具身感知与基于强化学习的仿人机械臂控制策略研究》
相符；本论文主要贡献位于**RL控制/操作技能适应**，不要求同时训练LLM、
VLA或完成整套现场双臂系统。ARES-R SkillRuntime、SceneSnapshot、cuRobo
是只读架构参考；仿真Panda的结果不能冒充JAKA/现场部署结果。

## 2. 拟解决的可反证问题

强规划器能够产出运动学可行的名义动作，但实际工具抓持、接触/插入、滑移、
跟踪、有限执行预算等变化可能使物理结果偏离预期。关键问题不是重新学习
自由空间关节轨迹，而是**既有技能执行过程中哪些反馈能让合法参数选择更有效**。

针对一个技能或接触事件，采用
- \(c\)：任务/物体/目标/技能合同和剩余预算；
- \(o_t\)：执行时真正可获得的机器人/工件观测；
- \(u_t^{nom}\)：强名义控制器的动作；
- \(a_t\)：小维、可执行的技能修正或参数；
- \(\Pi_{\mathcal C}\)：物理/动作许可投影；
- \(e_{t+1}\)：执行后真实物体、工具、接触等状态变化。

标准控制残差可写为\(u_t=\Pi_{\mathcal C}(u_t^{nom}+\delta_\theta(o_t,c,h_t))\)。
当前规划型技能的实际实现则是：策略选择参数，执行器更新当前队列/目标并执行；
目标偏移不等于该加法式中的电机残差。分析必须包括实际队列、参考、进度与预算。
这里的 residual、任务条件、历史和投影均有相关先例，**不是独立新颖性**。

## 3. 论文潜在的三个贡献槽位（均为待验证）

1. **问题与接口**：可复用名义技能中，任务条件与接触后果怎样定义合法参数
   的真实可执行性和下游价值；该接口应能跨至少两个不同物理操作复用。
2. **核心技术贡献**：动作/执行表示、技能适应结构、学习机制或其有意义组合，
   能够在与标准构件明确区分后，以公平对照支持可靠性、适应或效率主张。
   不要求每个组件全新；标准残差、SAC或实现修复不能单独改名为原创。
3. **证据**：在公平动作权限、任务分布和物理预算下，多个种子/新场景上的
   全任务成功或预声明的样本效率正结果，机制消融、失败归因和泛化成立。

如果具体新方法不是原创或性能无优势，就不宣称槽位2已完成；保存系统性
负结果和可用技术基础，并据物理实验重新设计要解决的操作机制。

## 4. 必须严格排除的错误结论

- 自建反馈被修复后成功增加 ≠ RL增益；普通PPO/SAC成功 ≠ 新RL方法。
- 参数输出与策略梯度改变 ≠ 工件物理结果提升；单案例救回 ≠ 总体改进。
- 事后选择成功并集 ≠ 任何部署策略的得分；已看开发集 ≠ sealed test。
- 规划器success ≠ 接触成功；几何margin预测好 ≠ 强基线后的操作正结果。
- 状态中已有全部物体速度/位姿时，拼接过去位移不一定新增信息。
- 标准动作效果预测/SCOUT式在线信念、STAP/Foresight下游价值、DAWN
  Critic预热、动作基元、task-frame、门控残差均已有研究；不得换名后认原创。
- Panda/SAPIEN、Isaac/JAKA和真实金隅现场各自有独立验证边界。
- 任何实验不能以弱化名义控制、修改官方成功判据或取样有利初态制造增益。

## 5. 不变的论文主评价

首要：**完整物理任务终态成功**（保留原生官方判据）。
辅助：配对救回/退化、成功条件下耗时、失败超时、接触进展/保持、物理违规、
交互+规划+数据成本、未见目标与物理条件的泛化。

可以提前指定样本效率为主贡献，但不能看完sealed test后再从失败成功率转
为有利指标。至少把标准反馈/强规划、普通同权限ResidualRL、监督或参数
搜索、最接近公开算法原型放进同一证据链。

## 6. 论文完成态（不能把必要条件替换成漂亮PDF）

- 主要技术主张有可复现证据，并通过同动作/观察/预算消融定位来源；
- 强名义/重规划和成熟残差RL对照，跨训练种子与独立场景组有配对不确定性；
- 第二种物理机制/任务与合理的未见条件，证据覆盖任务适应范围；
- 非特权感知或误差/延迟评估至少有一项，若宣称多模态须真实验证；
- JAKA应用为独立增强证据；仿真/场地/实机绝不混写；
- 论文主张、表、图、源码/场景/模型哈希一致，尚未实现标PROPOSED。
## 7. 论文主目标防偏条款（2026-11-01）

**原研究框架仍成立且不可被单个实验重命名**：
任务/结构化感知与技能合同 → 已有规划和可复用技能 →
**task-conditioned RL的可执行技能参数自适应** → 真实物理执行/
SkillResult回读 → 后续技能重决策 → 完整任务成功与新条件泛化。

早期`11_PAPER_FIRST_20261028/LATEX/PAPER_SKILL_HANDOFF.tex`研究的是
真实`PlaceA/Release/Verify→PickC/PlaceC`的两个参数边界，
在同一物理episode完成；当前PlugCharger只在抓持/接近之后学习
接触残差，**还没有恢复多技能决策机制**。它是训练器和动作表示
研究，不是整个论文题目。早期16环境并行PhysX训练同样是
技术能力，不要求每个新benchmark使用并行16环境。

按论文主张验收：至少一个主任务真的包含≥2个有效物理技能
参数边界，前段参数在实际物理中改变后段可执行性，后段必须
根据真实完成状态作动作决策；另一个不同物理任务验证方法的
可复用性。不能把规划waypoint视为技能完成，不瞬移物体，
不能把手工停止合同/动作Y-only投影/LN伪装成新RL贡献。

**理论选择并不先验确定**：Option/SMDP是技能级序贯决策基础；
POMDP只有在真实隐藏接触/工具状态造成观测别名、历史动作
能辨识且给未来决策带来好处时才值得成为论文理论主干。
基线合格就冻结共同的名义控制，不能无限修规则、把研究
锁死成单一插入任务的伺服控制，也不能削弱规则以凸显RL。

按具体主张分别记录证据成熟度。独立算法收益尚未证实仍记为
`UNVERIFIED_INDEPENDENT_ALGORITHM_GAIN`，但不是研究、撰稿或第二任务准备的
通行门槛。表示与执行语义、物理机制、学习收益和泛化分别验证；正式投稿的
性能主张仍需多种子、独立条件和机制消融。

2026-11-05：3 mm与21.98 mm之差包含实测附着/目标参考重建，不能表述为
已证实的七倍数值放大。当前比较保持三种真实抓持前驱决策与Peg原生100步。

2026-11-05 atomic_handoff：将**执行参考、时间权限、实际学习救回、稳定净收益**分开评价。新24确认支持可学真实救回，不支持超过冻结强规则；阶段重开与参数化Option为成熟构件。三抓持继续训练但当前未证明新增跨技能协同，Charger小型全成功研究不是方法迁移。保持单篇多阶段技能参数RL问题与原生主评价不变。


## 2026-11-05 claim-specific update: rule-relative executable adoption

The stable problem remains task-conditioned, physically executable multi-stage skill adaptation. New paired-data initialization supports conditional adoption above a strong rule (19/18 versus16 on24 new natural conditions), but explains all confirmed online-RL success outcomes. Full waiting-credit and entry-only updates were equivalent on every nonzero training event in the ablation seed. Therefore neither waiting credit, baseline gating, Monte Carlo labels nor standard clipped policy gradients are claimed independently novel. Preserve this learned baseline and the complete execution contract; future joint learning must beat it rather than recounting the same initialization benefit. Predecessor weights were frozen this cycle; new cross-stage RL advantage and second-task learned transfer remain unresolved. These are substantive Method/Experiments materials with explicit limitations, not a publication-readiness claim.


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

Evidence: [evidence/EXP-004/2026-11-06/joint_continuation/README.md](evidence/EXP-004/2026-11-06/joint_continuation/README.md). Next: Test omitted legal physical Options at real handoffs to determine whether the frozen two-proposal interface leaves repeatable recovery opportunity. Share any expanded authority across warm baseline and ordinary joint RL; do not expand unsupported MC-mix, alias-critic or paired-PG mechanisms. Preserve native Peg100 and sealed isolation.


### 2026-11-07 claim refinement, stable paper problem

Expanded executable authority is an opportunity diagnosis, not a contribution by itself. Preserve task-conditioned real predecessor and recurrent successor learning. Current independent action-coordinate candidate benefit is unsupported after two trained revisions; ordinarySAC is the retained shared baseline. Same model's Actor/greedy-Q extraction has different physical behavior, so a value's continuation policy and execution semantics must accompany any proposed improvement. Native final success, paired rescue/regression, all physical/data cost and frozen new conditions remain primary. Writing proceeds on implemented methods and scoped evidence, not contingent on a VERIFIED label.


## 2026-11-08 — continuation retention: actual method iterations and stronger frozen control

The single-paper question and native outcome standard remain unchanged. A complete implementation, finite empirical action-retention property, and actual physical performance are separate claims. Current projected-witness improvement is unsupported on new conditions; preserve the stronger fixed24/32 and inheritedSAC21/32 references rather than treating the new17/32SAC as the strongest available learner. Both fixed-control history and historical data budgets are disclosed. Charger has repeatable predecessor crossover but no demonstrated successor-parameter rescue in9 bounded probes. Do not claim learned transfer, safe policy improvement, joint synergy or publication readiness. Writing and method development continue on actual evidence.
