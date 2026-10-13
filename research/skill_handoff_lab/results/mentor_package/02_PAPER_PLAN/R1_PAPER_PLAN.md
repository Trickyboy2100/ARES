# 论文架构：以机制诊断决定RL主线

状态：拟验证，非已证实贡献。原有CASH cost-bandit结果仅历史planning-level结果，不作为在线技能链结论。

研究问题：复杂空间与姿态约束下，高层RL能否在有限规划预算或真实执行偏差下改善合法准备/恢复决策？若充分预算规划同样有效，应明确将计算预算或执行不确定性作为问题，而非声称RL天然更优。

章节：1工业技能边界问题；2参数化原子动作/技能价值/恢复/主动感知相关工作；3真实Skill Contract与证据等级；4经机制实验选定的Option-RL；5任务、同预算基线与训练协议；6主结果/泛化/消融/失败原因；7局限与结论。

任务族：E0开放持物物理验证（当前为isolated-arm smoke）；E1柜口水平约束同起点Prepare；E2狭窄入口及可恢复执行偏差；E3未见工位/尺寸/位姿及有依据的观测不确定性。E1–E3参数须在正式运行前冻结；调试real-aligned案例单独留存，不反复选seed。

训练/验证/测试按场景组和任务族隔离，同场景候选不可跨集合。所有策略同观测、同候选、同规划预算；报告规划调用数、Option时间、物理步数、训练交互、失败与UNKNOWN。规划超时/软件错误/代理冲突与真实物理失败分别统计。接触抓取不在当前范围。

|方法|任务成功率|恢复率|规划调用|执行成本|训练交互|
|---|---|---|---|---|---|
|固定规则|NOT_TESTED|NOT_TESTED|NOT_TESTED|NOT_TESTED|0|
|ARES-R启发式/Beam|NOT_TESTED|NOT_TESTED|NOT_TESTED|NOT_TESTED|0|
|等预算/充分预算搜索|NOT_TESTED|NOT_TESTED|NOT_TESTED|NOT_TESTED|0|
|监督方法|NOT_TESTED|NOT_TESTED|NOT_TESTED|NOT_TESTED|NOT_TESTED|
|标准PPO|仅软件smoke|NOT_TESTED|NOT_TESTED|NOT_TESTED|4096 smoke|
|机制匹配RL|NOT_TESTED|NOT_TESTED|NOT_TESTED|NOT_TESTED|NOT_TESTED|

潜在贡献需证明：工业约束下可恢复性/可接续性确实影响长程完成；学习方法相对强基线在预算、执行成本或泛化上有可重复优势；消融证明条件信息及合法选项的作用。若不成立，报告负结果并换机制，不增加无必要的世界模型/MPC。
