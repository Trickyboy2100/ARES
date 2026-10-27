# 科学问题、竞争解释与可反证主张

研究题目（暂定）：以合法技能参数和真实配对后果检验残差干预的收益—退化权衡。

## 当前主张边界
本轮不提出新PPO似然，不扩大旧R3。首先验证控制权限，再比较同源强反馈、固定调参、普通Residual PPO、参数化接口PPO及真实配对差分选择。完成真实新开发评价后，新算法独立增益仍为NOT_ESTABLISHED；参数PPO和paired选择均未稳定胜过普通Residual PPO及固定调参。

M0三个参数为放置XY±15mm与放置高度±12mm，仅在相应技能阶段实际进入TCP控制。普通Residual PPO使用同原生控制器的XYZ+夹爪修正，权限更宽；二者差异只能支持动作接口结论，不能冒充新RL损失。核心差分机制还需相对同参数权限的普通PPO和固定参数显示独立收益。

M1只在首次进入放置边界做一次KEEP/低12mm/高12mm选择，明确是contextual bandit。它比每步连续XYZ参数PPO权限更窄，决策频率也不同；两者不是严格同动作空间消融。paired与absolute预测器及其消融才是同候选、同频率、同容量的直接机制比较。同频率边界RL尚未补齐，负结果阶段不据此声称算法优劣。输入为当前robot/object/goal、此前物理变化和剩余预算；不输入未来success或case ID。完整配对后缀仅在独立训练条件中构造终态差分标签。每条真实冷重置分支单独收费；各分支实际执行50步，所以本轮不把较早出现成功谓词冒充执行时间缩短。

全部是显式state-based模拟基线，部署非privileged感知与多模态泛化尚未验证。ARES-R仅为只读系统背景，未使用现场点云、SDK、机器人或受限资料。

## 最接近的先例（2026-10-28核对）
| 原始来源 | 已存在的构件 | 本轮必须区别的证据 |
|---|---|---|
| [RAPS, NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html) | 参数化原语接口 | 参数化本身不新；须证明具体物理权限与强基线后的增量 |
| [STAP, ICRA2023](https://sites.google.com/stanford.edu/stap/home) | downstream Q、技能依赖、参数搜索与不确定性 | 单一边界ranker不是新的长程Q规划；必须比较绝对结果价值与差分标签 |
| [Fabrica, CoRL2025](https://proceedings.mlr.press/v305/tian25a.html) | task-frame、计划残差、接触装配 | 不能以规划+残差组合命名创新。已检查公开fabrica_task_assemble.py中路径变换与residual action进入实际动作的代码 |
| [Foresight Residual RL, IROS2026](https://arxiv.org/abs/2607.16506) | 终态handoff的后续成功预测与残差训练 | 本轮是冷重置配对差分和弃权的可学性检验，尚无长程handoff贡献证明 |
| [SCOUT, CoRL2026](https://arxiv.org/abs/2609.36107) | 动作—结果反馈更新动力学belief | 仅把上一步物理位移接入输入不等于SCOUT式适应；需no-effect消融 |
| [Action Space Design, ICML2026](https://proceedings.mlr.press/v306/feng26ab.html) | 动作表示改变可学性与控制稳定性 | 动作接口改进不能称新的RL算法 |
| [RecoveryChaining公开代码](https://github.com/shivamvats/recovery-chaining) | 价值估计、恢复技能组合 | 本轮尚无重复恢复选项，不能称已完成recovery chaining；已检查learn_recoveries.py价值模型调用 |
| [U-GRA原始链接](https://www.mdpi.com/2075-1702/14/8/924) | 任务书给出的twin-Q disagreement gating线索 | 当前出版社429，未独立完成正文/代码核验，不能臆断与其有原创区别 |

只核对摘要/项目页的文献不视作已复现算法；本轮不借新颖性表宣称首次。未实际复现的竞争方法只列为竞争解释。

## 可反证条件
1. 固定调低放置高度即可达到相同收益：应归为强控制基线，不是学习新方法。
2. 相同控制权限普通PPO达到相同或更好结果：参数化是接口工作，不是差分机制增量。
3. paired ranker与同容量绝对结果预测相同：差分学习没有独立机制证据。
4. 弃权降低退化但也失去所有救回：报告Pareto权衡，不称无条件改进。
5. Peg无法在原生100步内形成可信强基线：先修物理跟踪/接触问题，不能把零基线当论文胜利。
6. 三训练种子共享验证case，统计不得把192个模型-case结果称192个独立场景。封存test仅在方法固定且有重复方向后使用。
