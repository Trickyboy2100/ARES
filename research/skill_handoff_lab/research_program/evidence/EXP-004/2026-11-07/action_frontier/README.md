# Action frontier → own-policy joint Option learning

基点c50123e；原生PegInsertionSide-v1/100步/真实抓持/共享强规划器。训练与物理结果已实际完成，非计划。所有本轮新自然场景均为设计、训练或开发确认，未开启封存测试。

[打开完整审查页](REVIEW.html) · [论文/源码对照](PRIOR_ART.md) · [机制与反例](THEORY.md) · [完整复算](LEARNING_SUMMARY.json) · [原始归档与逐文件SHA](RAW_MANIFEST.json)

## 本轮实质推进

1. 256次原生重执行：16个新设计初态，前段三抓持、后段完整现有物理Option。Warm8/16、设计集最佳固定localZ−12/16、有限实测成功并集14/16。只有2例超出最佳固定，不是6例独立方法空间，更不是可部署oracle。
2. 扩展共享执行接口，完成5个128回合own-policy训练：普通SAC两个真正独立网络种子、固定参考/偏移坐标候选、状态相关实际目标坐标修订、同信息终局监督评分。640次真实在线训练；每臂另消费同一256回合设计replay，不能算零数据成本。每臂5120次更新；不是旧transition改名。
3. 固定坐标候选20/24未超普通SAC21/24；根据同方向跨参考成败反转修改Actor坐标，使动作关系由实际目标几何决定，再训练128回合，仅19/24。两次假设均有真实梯度训练、行为与负结果，不继续无依据扩训。
4. 普通SAC两种子均21/24、相同两次救回且无退化。新32自然初态用冻结final128确认如下，未按确认结果选checkpoint。

## 新自然条件的冻结确认

| Fixed policy | Native success | Rescue | Regression | Planner calls |
|---|---:|---:|---:|---:|
| confirm_chart_state_32201 | 19/32 | 1 | 3 | 276 |
| confirm_old_rl | 22/32 | 1 | 0 | 266 |
| confirm_rule | 20/32 | 0 | 1 | 259 |
| confirm_sac_32201 | 19/32 | 1 | 3 | 267 |
| confirm_sac_32202 | 22/32 | 3 | 2 | 278 |
| confirm_supervised_32201 | 18/32 | 3 | 6 | 272 |
| confirm_warm | 21/32 | 0 | 0 | 266 |

基准为同初态warm；两个网络在相同32场景上评价，不能当64独立场景。历史oldRL有不同历史训练预算，只是保留资产对照；不将其分差解释为等预算因果收益。所有新方法共享执行权限/观测/控制器/原生终止标准；各自采集自身数据。冻结rule不在本轮调规则。完整配对场景及描述性bootstrap区间见JSON；小样本不作总体安全保证。

## 固定开发集与真实学习曲线

| Fixed policy | Native success | Rescue | Regression | Planner calls |
|---|---:|---:|---:|---:|
| sac_32201@0 | 19/24 | 0 | 0 | 199 |
| dev_rule | 18/24 | 1 | 2 | 196 |
| dev_old_rl | 20/24 | 1 | 0 | 200 |
| dev_fixed5 | 16/24 | 2 | 5 | 186 |
| sac_32201@128 | 21/24 | 2 | 0 | 198 |
| sac_32202@128 | 21/24 | 2 | 0 | 200 |
| factor_32201@128 | 20/24 | 1 | 0 | 199 |
| chart_state_32201@128 | 19/24 | 1 | 1 | 207 |
| supervised_32201@128 | 16/24 | 1 | 4 | 203 |

| Learner | 0 | 32 | 64 | 128 | Own train successes | Actual decisions | Updates | Own planning calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chart_state_32201 | 19/24 | 16/24 | 18/24 | 19/24 | 68/128 | 779 | 5120 | 1116 |
| factor_32201 | 19/24 | 17/24 | 19/24 | 20/24 | 62/128 | 768 | 5120 | 1120 |
| sac_32201 | 19/24 | 17/24 | 19/24 | 21/24 | 65/128 | 779 | 5120 | 1129 |
| sac_32202 | 19/24 | 19/24 | 20/24 | 21/24 | 66/128 | 755 | 5120 | 1135 |
| supervised_32201 | 19/24 | 20/24 | 17/24 | 16/24 | 64/128 | 693 | 5120 | 1141 |


765400–423在本轮研究中已看，仅为开发。新确认765600–631在所有模型冻结后才打开，不是封存测试。设计765000–015、own-training765100–227，历史及保留seed碰撞核对见SEED_CHECK。各方法的探索行为分数与贪心部署分数分开。

## 机制证据与被排除的解释

- 同signed Y+换active/refreshed参考，10/16成败反转；Z−为12/16。参考含实际附着/目标重建，不是已证实的3mm数值放大。413个返回规划目标的坐标推导误差≤4.36e−8m；另1次路径参数化失败保留。几何正确不保证学习收益。
- 状态坐标修订的核加权相反结果比例.258，均匀.385；576动作对来自16场景，不是576独立物理条件。这项proxy改善没有变成完整任务优势。
- 765416相同物理前缀68步：普通SAC选择localY+并成功，修订选择rule0、错过必要早期修正而失败。失败不只是“最后动作太多”。
- 765423同前缀71步：SAC的localY+成功；历史oldRL也救回，不能称独占新能力。765402在93步已经成功，94步Z−不记为救回原因。
- 将普通SAC最终策略限制到历史提议支持，仍21/24；213个实际边界也满足旧Gate更严格mask。因此新动作目录有物理机会，但当前开发提升并未证明必须依靠扩大后的部署权限。不是限制接口重新训练的对照。
- 冻结同一SAC模型：Actor21/24；全Mean-Q13/24；只换后段Mean-Q13/24；只换前段Mean-Q19/24。后段提取失配足以造成8次退化。Q是含shaped reward/熵的策略价值，不是成功概率；贪心提取也改变后续分布，不能仅称数值校准误差。
- 三抓持在训练中均有实际选择，但固定grasp设计7/8/7、并集9；765010换抓持本身可救回。没有证明必须联合改变前后参数的协同收益。

## 算法与近邻边界

普通离散SMDP-SAC、成功BC、受控探索、完整Option与Mean-Q都是成熟构件。实际Option持续时间入replay，gamma=1；熵按决策计，不宣称时间划分不变性。候选仅修改Actor residual-logit坐标；固定矩阵和state kernel不是已证实的新算法。状态核改变有限网络函数类，虽参数数相等，不能声称全局表达类相同。

已读PEARL/ViSkill/PolicyDecorator/AGILE公开源码，MP-DQN/ActionBranching/ActionRepresentations等原论文；确切SHA、许可证情况和借鉴边界在PRIOR_ART。AGILE已有状态相关动作关系，因此一般“根据状态共享动作”不能作为原创。没有新STAP官方任务排行榜或此目录完整STAP移植比较，不宣称胜过它。

## 可进入论文的内容与下一项实验

可以进入Method：真实前驱—SkillResult—后继Option接口、数据/恢复/成本定义和两种可反证坐标机制。可以进入Experiments：上述机会矩阵、真实在线曲线、固定确认、参考反转、策略提取消融和失败视频。独立候选优势仍未成立；不把基础设施或监督预热计作原创。

下一项应并列保留预热、历史RL、普通SAC与完整Option，按逐场景证据确定参照，而非继续扩训坐标核：从其冻结策略在新确认中剩余失败/退化，检验后段合法动作价值与实际续接匹配。首先在新的训练初态、共同实际前缀下检验Actor选择和Q偏好是否对应不同Option完成合同；若数据支持，再做同信息、同预算的策略一致改进。已有Mean-Q单独提取失败不能自动证明任一新损失有效。前驱真实作用继续保留，但不扩大没有联合增量证据的抓持网格。

## 成本、原始材料与复现

本轮唯一原生执行 **1882回合/188200物理步/16027规划调用**。规划累计77.255s，仿真回合累计11657.3s；并发运行，不能当实际墙钟历时。网络优化CPU，PhysX单张3080。GPU峰值本轮未连续测量，不用快照冒充峰值。两次采样启动故障200步计入，未进入有效学习；每一原始文件的final官方success与100步已复算。

原始运行目录：`/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/action_frontier_20261107`。发布RAW_*.tar.gz包含完整新模型/优化器/RNG/replay/START事务/训练评测轨迹/运行脚本/日志/源码快照；RAW_MANIFEST逐成员与归档SHA已回读核验。保留CURRENT和0/32/64/128检查点。历史模型由GLOBAL_ASSET_MANIFEST和既有发布证据定位，不能当本轮新训练。未执行的static-chart及额外factor/supervised种子脚本只是预备入口，不计为完成。

复算：仓库tools/analyze_full_option_learning.py --root RUN；plot_full_frontier.py --root RUN --out FIGURES；analyze_full_option_rescues.py；review_full_option.py。原生重跑采用归档run_*.sh和已有隔离handoff_offline.sh；不要对已有输出目录覆盖重跑。训练完整事务resume已测，精确随机状态与优化器续接对齐。

保留的两组开发视频：[正例RAW](video_765423/RAW.mp4)、[英文标注](video_765423/ANNOTATED.mp4)、[逐帧曲线](video_765423/SYNC_VIEWER.html)；[负例RAW](video_765416/RAW.mp4)、[英文标注](video_765416/ANNOTATED.mp4)、[逐帧曲线](video_765416/SYNC_VIEWER.html)。均为101帧已测状态重放，零新增物理，不作为第二次成功证据。每张科学图另存PNG/PDF。

初始化勘误：历史31901/31902因Gate重设Torch种子而初始网络相同，仅探索种子不同。本轮在加载后重设种子，实际checkpoint验证32201/32202 hidden/Q不同；历史分数全部保留。


## Post-confirmation mechanism check: deployed action support

Both frozen SAC seeds are reexecuted on all32 now-seen confirmation scenes with only current historical proposal physical IDs plus rule allowed. No model, horizon, reward, planner or initial state changes. These64 additional native episodes test execution dependence, not new independent generalization or a retrained restricted-learning comparison.

| Network seed | Full catalogue | Old proposal support | Full-only successful scenes | Restricted-only successful scenes |
|---|---:|---:|---|---|
| 32201 | 19/32 | 20/32 | 765625 | 765608, 765626 |
| 32202 | 22/32 | 21/32 | 765625, 765627 | 765604 |

The full-policy benefits and regressions must both be retained. Any full-only success supports necessity of the extra executed choices for this frozen policy on that state, not impossibility of every other old-support policy. This does not turn the failed coordinate candidates into an independent algorithm contribution. Exact physical prefixes, raw paths and complete boundary events are in POSTCONFIRM_SUPPORT_EFFECT.json.


### Evidence-based retained capability and unresolved reliability

The frozen second SAC network uses an additional localZ− Option to rescue765625 and765627. Removing expanded action support loses both; one harmful refreshedY− intervention in765604 is also removed. Full22/32 versus restricted21/32 is therefore a capability/retention tradeoff, not robust dominance. The first seed similarly gains765625 but breaks765608/765626 through localZ+, yielding19 versus restricted20. These are post-hoc same-model physical controls on now-seen conditions, not a new independent test. Ordinary SAC owns these gains; neither coordinate candidate earns credit. The task interface has a measured new deployed capability, while reliable selection is still unresolved.

新条件成功视频：[完整RAW](video_765625/RAW.mp4)、[英文标注](video_765625/ANNOTATED.mp4)、[逐帧曲线](video_765625/SYNC_VIEWER.html)，普通SAC32202对warm，同前缀64步。
