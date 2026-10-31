# EXP-002 最近邻源码核对与可反证实验设计

核对日期：2026-10-31。状态：**论文/公开源码检查完成；外部方法训练与成绩复现均未执行**。
本文件不产生任何物理步、优化器更新或新方法增益。没有运行或复制外部实现。

## 1. 已核对的证据与许可边界

| 方法 | 本次实际检查 | 适用边界 |
|---|---|---|
| STAP | [官方仓库固定版本](https://github.com/agiachris/STAP/tree/0fe0dcbd5ede5ce4ee3d952642f70b5550a0813c)，`stap/planners/cem.py`、`utils.py`、`base.py`及LICENSE | MIT许可已读。是技能动力学与价值驱动的序列参数规划；不能把下游价值+参数CEM重新命名为创新。 |
| DAWN | [固定版本](https://github.com/Guozheng-Ma/DAWN/tree/71122b7fa89568bc2d49831fed8cb1b01e5a91e8)，`nets/dawn.py`、`online_dawn/pi_dec_diffusion_maniskill2_dawn.py`、LICENSE | **该版本LICENSE为0字节**，不能因GitHub有License标签即视为可复用许可。当前仅检查思想和代码，不复制实现。训练器依赖ManiSkill2，不能称已在本机ManiSkill3复现。 |
| SCOUT | [固定版本](https://github.com/liy1shu/SCOUT/tree/cdb079ccf9f40c5ed120261f665945aeff3a6473)，README、`scout/models/scout.py`、`dynamics.py`、`training/trainer.py` | 树中未找到项目顶层LICENSE；仅发现`third_party/metaworld/LICENSE`，后者不授权整个项目。项目网页的CC BY-SA也不能替代源码许可。 |
| Foresight Residual RL | [原文v1](https://arxiv.org/html/2607.16506v1)，重点算法1/2与技能边界定义 | 论文链接的官方项目页本次返回404，未确认官方代码/许可，不虚构仓库。不要误用同名导航项目`ut-amrl/foresight_public`。 |
| 接触模式自适应 | [Sidhik等，RAS2024作者公开稿](https://homepages.inf.ed.ac.uk/msridhar/Papers/ras24_changingContactManipulate.pdf)及[出版记录](https://www.research.ed.ac.uk/en/publications/an-adaptive-framework-for-trajectory-following-in-changing-contac/) | 预测误差驱动接触模式/控制适应有明确先例；本次未做其控制器复现。下文RLS是自行实现的标准基线设计，不声称为该文算法。 |

版本通过GitHub API `commits/HEAD`解析，再按SHA读取raw源文件。没有下载权重、安装依赖、运行机器人SDK或占用GPU。

## 2. 机制核对：什么已经存在

**STAP。** `cem.py`在技能动力学中展开候选动作序列，调用`evaluate_trajectory`，按成功评分选精英并更新均值/方差。其随机/CEM参数规划是真正应纳入的竞争思想；对比时必须区分学习模型查询和调用真实模拟器的特权搜索成本。[源码](https://github.com/agiachris/STAP/blob/0fe0dcbd5ede5ce4ee3d952642f70b5550a0813c/stap/planners/cem.py#L224)

**Foresight。** 离线采集前阶段终态，运行后续策略估计概率，训练视觉预测器；逆序训练残差策略，将后续概率乘到局部成功奖励上。单纯增加一个“后续可行性/价值头”与奖励乘子不构成区别。ARES若只在接触过程中适应，尚未证明跨阶段收益；需另验证合法抓持选择影响后续接触和完整成功。[算法1/2](https://arxiv.org/html/2607.16506v1)

**SCOUT。** 固定源码中，`inner_update`对动作条件动力学预测与实际结果编码的MSE求隐变量q梯度；`adapt`只更新q；策略为q条件化FiLM GMM。`trainer.py`外环使用目标动作的GMM负对数似然。**不是普通在线SAC的同义词**；将SAC与历史预测网络拼接也不能自称区别于SCOUT。[模型](https://github.com/liy1shu/SCOUT/blob/cdb079ccf9f40c5ed120261f665945aeff3a6473/scout/models/scout.py#L73)、[训练器](https://github.com/liy1shu/SCOUT/blob/cdb079ccf9f40c5ed120261f665945aeff3a6473/scout/training/trainer.py)

**DAWN。** `LayerNormCritic`明确使用LayerNorm；训练器允许零残差名义数据预热、可选critic-only预热、不同残差探索策略和SAC更新。因此“预热+LN”应强化普通残差基线，不能作为ARES贡献；源文件中额外选项不等于论文主结果都使用了它们。[网络](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/nets/dawn.py#L92)、[训练器](https://github.com/Guozheng-Ma/DAWN/blob/71122b7fa89568bc2d49831fed8cb1b01e5a91e8/online_dawn/pi_dec_diffusion_maniskill2_dawn.py#L102)

## 3. 当前最有信息量的实验，而非新增名称

以下是**预先提出的设计，不是本次已完成结果**；真实运行和计数以本轮主证据为准。

### A. 先分清可控制与不可辨识

1. 新原生初态保持官方200步、成功判据和分布；比较最强深度保持控制、抓持后几何再规划，以及联合抓持/后续IK筛选。分别记抓持失败、IK不可行、自由空间跟踪、接触停滞、预算耗尽。
2. 自建接触变化族独立命名；仅episode初始化改变已声明物理条件。使用常见/挑战混合分布，不能仅挑强控制失败样本作为总体成绩。
3. 每个相同初态×相同参数至少做独立重复以估计物理波动，再与不同参数作用差比较。记录请求、裁剪、实际关节/TCP增量及真实工具增量。只记录成功并集不足以证明可学习性。
4. 隐藏摩擦/跟踪因素可供分析员记录，但不给任何部署策略读取；若全部动力学已在观察中，历史优势不能解释为新增信息。将完整状态、当前可观测状态、等长动作—结果历史分开比较。

### B. 强RLS/局部模型控制基线

在任务坐标拟合 `delta_tool = B * phi + noise`，其中`phi`含**实际施加**的局部动作、常数项及接触状态；姿态用局部旋转向量。RLS逐步更新B和协方差，采用相同有界合法动作和剩余预算。短视控制枚举零残差、侧向正负修正、有限推进/回退，通过预测几何误差、接触停滞与动作代价选择动作。正则化、遗忘因子和模式切换阈值只在训练设计集调节。

必须保留“不更新B”的几何模型和“更新B但不用历史网络”的自适应模型两项。若RLS已经救回全部可救任务，则形成有效控制结果；不因此宣称RL机会已成立。若参数矩阵病态，检查动作缺乏激励；不能把不可辨识的系数画成确定因果。

### C. 只有真实条件排序存在后才训练

最小训练比较为：强规则/再规划、RLS短视控制、同动作Residual SAC、等长历史Residual SAC、监督参数评分和一项候选机制。普通SAC可使用名义数据预热/LN，全部组共享数据预算、奖励、技能终止规则、动作界限和可观测量。对原生抓持前段耗时不能免费重置到完美接触边界；可单独训练suffix，但主成绩必须完整任务重跑。

候选待验证机制是**执行效果的合同相关辨识**：仅对影响当前技能合法推进及后续保持条件的物理效果维度建模，并把预测不确定性纳入参数价值学习；既不将全状态历史自动视为新信息，也不将失败概率头自动视为贡献。需区分效果预测损失、合同投影、额外历史及额外数据各自贡献。只有超过同历史RL、RLS和SCOUT式效果适应原型，且去掉合同相关性确实退化，才讨论独立方法增量。

立即可反证：若几何状态已完整解释最优参数，停止隐信念方向；若固定参数同样好，停止任务条件化主张；若仅接触阶段得分提升但抓持/全任务退化，不算主指标增益；若优势仅来自更多模拟器查询，按搜索/数据成本收益报告。

## 4. 何时更换具体任务

若原生PlugCharger主要是抓持/后续IK失败，而接触又被强规则饱和，应优先评估有可观测接触结果且存在物理耦合的工具抓持—受力操作。Drawer的隐藏关节轴与摩擦、工具抓持位置影响施力方向等均是候选，**尚未在本文件执行**；“增加随机偏差”本身不建立论文价值。评估时仍保留公开原生成功标准和成熟几何控制，不能通过删掉反馈/降低控制器预算制造RL空间。

结论：本次已排除“后续价值”“效果误差隐变量”“预热LN”“在线局部辨识”四种简单换名创新。下一步价值来自真实物理配对能否显示强模型仍未覆盖的、可辨识的参数决策空间，而不是先固定网络名称。
