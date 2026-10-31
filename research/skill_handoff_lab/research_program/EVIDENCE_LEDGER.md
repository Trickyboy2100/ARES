# 已有证据总账（只记来源支持的结果，附局限）

**研究快照截至基座提交 `c013f7e33fe65c3a76c3ef2dfc1dedc6d4816a67`。**
这里不是训练任务队列。新实验后可追加完整、准确的链接并更新状态。
所有数值使用自己对应的seed/case/success/horizon；不同任务或控制器不相减。

| 研究集 / 来源 | 数值与范围 | 支持什么 / 不支持什么 |
|---|---|---|
| [旧 JAKA PhysX v6 Geometry](../deliverables/client_sci_20261027/05_METHOD_AND_EXPERIMENT/TRAINING_DECISION.md) | 反馈8/32，普通Residual三种子13/18/23；P2分层反馈5/16、RL6/6/7 | 说明普通RL在自建未见几何有初步帮助；主要来自P1，不是新机制 |
| [R3执行资格似然](../deliverables/client_sci_20261027/08_ITERATIVE_RESEARCH/R3_RESEARCH_DECISION.md) | 强反馈26/32，普通PPO22/25/20，候选24/24/18 | Actor改变≠稳定增益 |
| [公开Stack新64](../deliverables/client_sci_20261027/10_METHOD_EVOLUTION_20261028/RESULTS.json) | 名义60；Residual63/62/64；参数PPO62/63/60；paired60；固定/CEM62 | PPO基础有正结果；动作权限不同、develop已看，不能宣称新算法 |
| [Pyramid机会与闭环](../deliverables/client_sci_20261027/11_PAPER_FIRST_20261028/OPPORTUNITY_SUMMARY.json) | 64设计名义52、固定59、九组合oracle62；64开发普通PPO51/53/56、风险候选51/54/55 | 前后技能参数有真实物理作用；风险候选均值+0，oracle不是policy |
| [Pyramid价值/接触诊断](../deliverables/client_sci_20261027/12_SCI_OPPORTUNITY_20261029/README.md) | 576物理试验，前段失败33、后段未完成36；风险13/735；开发名义/监督/FQ/合同30/32 | 旧任务学习空间不足、风险标签稀疏；数值非独立场景数 |
| [PullCubeTool](../deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029/README.md) | 官方0/4→增强3/4；另一新8例几何前瞻7/8 | 强几何/执行器改进大；目前不值得刷RL，不是新方法增益 |
| [PlugCharger](../deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029/README.md) | 12设计场景×9参数，固定5/12、有限成功并集11/12（分支有波动）；新8开发例固定2、几何2、监督3、FQ1、margin-FQ1 | 局部动作可改变终态，离线价值学习未胜；独立RL新机制UNVERIFIED |
| [Charger真实配对案例](../deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029/metrics/VIDEO_CASES.json) | 同抬起后态，50mm与75mm接近：误差22.05→3.66mm，失败→成功 | 证明某个参数在某一案例有效；非总体因果/泛化收益 |

## 原则

- 既有 `53/57` 等条件成功与 `53/64` 全任务成功，不能混成同一指标。
- 12训练×9分支是**108条执行**，不是108独立任务初态；同样如此，
  物理tick、RL有效决策数、PPO更新次数各自分账。
- c013中“388回放/56,400完整记录物理步”不是388独立训练场景，也不是
  新的RL在线训练预算；三模型是监督/Fitted Q训练。
- 所谓“未见开发集”在结果被查看后已经用于研究决策，**之后不再是密封测试**。
  独立 sealed split 必须在方法冻结后才运行。
- Pose replay/渲染不能称新PhysX训练；实际成功来自源运行/独立审核。
- 新方法已证实独立性能增益：**否**。不能将学术工作标题写成已完成创新。

## 新证据应如何添加

追加一行：`experiment id | source snapshot SHA | scene group/split | native task &
horizon | fair baselines & action/obs | seeds/cases & metric | outcome |
uncertainty & limitation | exact raw path/hash | science decision`。
更正旧条目要保留更正依据，不抹去真实负结果。

## EXP-001｜2026-10-31 原生强控制与重复性

[完整精简证据与源码/原始SHA索引](evidence/EXP-001/2026-10-31/README.md)：原生PlugCharger200，14初态、74完整执行+105步已记录异常，合计14905控制步、0优化步。18跨进程配对成败一致、17记录状态一致；设计4例原积分+保持1、有界反馈+保持4；冻结新8例1→5，5救回/1退化。事后局部轴向保持修正4例2→4，不是新确认集。保存真实非Git轨迹与一组标注的姿态回放视频；无现场数据。支持控制器修复，不支持独立RL创新；科学决定D-006，活动实验仍EXP-001。

## EXP-001 independent confirmation and visual mechanism, 2026-10-31
[Source-backed evidence](evidence/EXP-001/confirmation-v2-2026-10-31/README.md): 24 cold-process episodes on 8 new scenes, 2/8 integral vs 7/8 original servo vs 7/8 local-depth servo. Four additional inspected grasp-angle diagnostics all fail. 5600 actual steps, zero optimizer updates. Protocol RAW/ANNOTATED/TRACE/CAPTION generated for two historical diagnostic cases; replay adds zero physics. No new RL claim. EXP-002 opportunity not yet established beyond the strong suffix.

| [EXP-001独立确认55861de](evidence/EXP-001/confirmation-v2-2026-10-31/README.md) | 新8初态积分2/8、servo7/8、depth-hold7/8；7/7进入后段成功；同成功7例首次到位平均快18.71步 | 当前插入后段确定性控制饱和；方法RL更新0。另25/5等历史样本不可拼为训练新证据。 |

## 2026-10-31｜EXP-002真实双轨探索、在线SAC与反馈消融

[主证据](evidence/EXP-002/2026-10-31/README.md) · [逐例SHA](evidence/EXP-002/2026-10-31/analysis/EPISODES.json) · [成本](evidence/EXP-002/2026-10-31/analysis/SUMMARY.json) · [成对比较](evidence/EXP-002/2026-10-31/analysis/EVAL_COMPARISON.json) · [模型](evidence/EXP-002/2026-10-31/models/INDEX.json)。

原生新4scene3/4；自定义质量/摩擦12/12由强控制成功。观测偏差新4scene×3条件：geometry4、RLS6、search10、SAC1、effect SAC3、prior-only3、同SAC零Z7，分母均12。零Z六救回零退化；effect/prior-only逐例完全同成败。所有当前比较的记录前段边界相同，无前段失败；不是不可见PhysX内部状态等同证明。

真实训练包含普通SAC、效果特征SAC、既有critic-LN对照；每轮1024技能决策、896次actor/critic更新，训练物理步/尾段/原生全任务成本分别记录。LN alpha=.1协议偏差与补齐的alpha=.02分开报告，实际结果从模型训练manifest解析。崩溃旧run的四episode800步也计入，重复恢复文件不重复计数。完整累积成本以SUMMARY为准，不把重渲染或CPU分析计为物理实验。

方法收益：UNVERIFIED。没有复现SCOUT/DAWN/STAP论文成绩；最近邻源码/许可已核对，LayerNorm不是新方法。完整视频与英文圈注为实测姿态重播，保留源轨迹、投影、帧PTS和浏览器同步QA。下一项EXP-003，重点参数权限和强RL学习能力，不以扩大弱SAC或小样本反馈特征网络代替机制研究。

最终补齐对照：critic-LN alpha0.02普通SAC10/12，alpha0.1偏差组4/12；同设置effect+LN2/12，prior-only2/12。共五组真实训练，每组1024技能决策、896次actor/critic更新。结果仍不支持候选特征的独立算法收益。
