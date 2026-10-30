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
