# 持久接触参数与有效执行likelihood：理论边界

这是实现核对与已有思想定位，不是新定理或已证实算法优势。相关先例包括[H-PPO，IJCAI2019](https://www.ijcai.org/proceedings/2019/0316.pdf)、HACMan++的primitive参数mask，以及本项目v6真实有效似然pilot（56/64对57/64，未达到推广标准）。H-PPO并非本文相同的stateful执行器，不应声称已复现它；同样不能把一般的无效参数边缘化叫首次提出。

更早的[FiGAR: Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning，ICLR2017](https://arxiv.org/abs/1702.06054)已经联合选择动作与重复时长。因此固定周期或接触事件触发的参数保持，不是首次提出动作持续性、时间抽象或可变时长动作；本轮也未复现FiGAR。我们需要独立检验的是：在当前持续接触执行器中，正确的有效执行似然是否比同接口普通PPO产生可重复物理收益，而非将已有时间抽象重新命名。

该工作本轮补核[正式PDF](https://arxiv.org/pdf/1702.06054)首页及§2：印刷作者名为Sahil Sharma、Aravind S. Lakshminarayanan、Balaraman Ravindran；以该印刷版本列参考。其将动作和重复次数分因子，并在保持时长内累积、折扣回报；当前ARES仍逐控制步计账，不声称移植其SMDP训练。

## 1. 必须先固定实际执行语义

设采样前状态包含当前物理观察、已有held parameter、刷新计时器；mask m(s)在采样前确定。R3真实源码中，正常模式使用XY，接触偏置/yaw/法向只在对应阶段允许刷新时使用；自动撤回/环绕期间连XY也不执行新采样，可能出现全零mask。未刷新的维度保留上一合法参数，不能将“本步仍采样但未用”记成物理执行。全零mask的Actor新采样梯度为0，但Critic仍可更新；该事实需记录覆盖率。若contact事件来自上一个已完成物理步，可作本次mask；不能用本动作执行后的接触或未来任务成功回填mask。

actor、critic及rollout应能获得足够的held value/timer信息；否则可以诚实研究部分可观测策略，但不能无依据宣称输入是完整Markov state。所有对照享有相同持久参数、刷新规则、物理合法范围与观测，学习机制只改变目标中无效采样维的处理。

正式R3增加了reset工件XY和peak displacement以观察20mm触发条件，共87D/5D；早期84D/5D只属smoke。最终仍以每run源码/manifest为权威。当前PPO的entropy coefficient为0，所以尽管distribution实现也mask entropy，这一支不影响当前训练loss；不能虚构当前方法因改变entropy正则而获益，也无须为此新增无意义训练。

当前PPO仍按60Hz控制步采样和计账，不是仅在Option边界采样的SMDP训练。保持/再接触时长已通过每个物理控制步的奖励与折扣进入普通轨迹；不能因参数持续多步就宣称实现了新的SMDP估计器或学习型技能终止。阶段切换仍依共享合同规则，恢复启动/路径helper也不是学出的高层切换网络。

## 2. 一步执行动作分布

对独立Gaussian proposal，若inactive维确实不影响当前转移及奖励，则积分它们给出：

`log pi_exec(a_active | s) = sum_i m_i(s) log Normal(a_i; mu_i(s), sigma_i(s))`。

未刷新值是状态记忆对应的Dirac分量，不应把旧held value塞进当前Gaussian算概率。PPO ratio的old/new使用**同一条rollout保存的pre-action mask**；评估时不能按更新后policy、后验contact或重新生成timer重算mask。entropy若按执行维度定义也须明确，与baseline保持相同系数及报告active-count分布。

对未裁剪policy-gradient score，在状态/有效动作条件下，无用独立采样的score条件均值为0；边缘化可去除这部分条件噪声。这是标准条件期望/Rao-Blackwell直觉，**不意味着PPO clipped surrogate无偏或一定有更小方差/更高任务成功**。PPO clipping、优势归一化和样本更新本来就引入额外近似，不能直接搬用未裁剪结论。

## 3. 与物理约束和旧负结果分开

参数refresh mask不是collision/soft-limit投影本身。命令裁剪、antiwindup、IK投影仍可能使proposal与实际工具动作不同；至少记录proposal、clipped/reference-applied、held value、mask、真实tool/object状态与physics ticks。不能因此声称完整约束投影后的解析密度已被精确处理。

v6曾在hold分支将无效XY边缘化，真实pilot只多1/64而无独立优势。当前候选若使用持久接触/yaw参数，与旧hold接口不同，但一般统计思想不新。需要证明有益执行方式、同执行器标准PPO、去事件触发/固定周期、完整与有效likelihood的独立消融，额外交互逐项计入。

## 4. 最小验证记录

- mask覆盖率、每维实际刷新次数、Actor score梯度与参数变化；m=0时该维distribution-output梯度应为0，共享躯干仍可有其他维梯度。
- 未刷新dim改变proposal而不改变当步applied action的确定性单测；相同cold start的真实机器人/物体轨迹，而非只看NN权重。
- 与相同stateful接口的普通PPO同数据/容量/奖励/参数范围；先验主要指标完整物理成功，不事后以某个微小时间差替代。
- 跨训练seed/新scene配对的真实结果；仅梯度正确、训练return上涨或接触位置不同都不自动证明SCI新机制。
