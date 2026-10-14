# ARES × ARES-R：导师三项成果总入口

最新阶段：[R1中文简报](MENTOR_3PART_BRIEF_R1.md)——真实1→4→16 PhysX在线PPO、同起点Prepare规划比较与实际带物执行。当前尚未证明任何机制必须使用RL；下方保留2026-10-13历史成果，不能将其planning-only训练误读为完整工业物理任务。

## 历史阶段：2026-10-13

## 一句话结论

从“不带物诊断路径、未训练网络”推进到**190上真实带物cuRobo规划、独立轨迹验收、真实数据Ranker/PPO/CASH联合效果头训练，以及实际Isaac Sim 5.1静态截图**。但安全结论限于声明的研究底盘模型；CASH没有超过强贪心；Isaac物理任务及阵列尚未完成。旧负结果全部保留。

## 成果1：方法初版

[方法说明](01_METHOD/CASH_METHOD_V0.md)与[框架图](01_METHOD/CASH_METHOD_V0.png)。工业技能局部成功不一定可接续，拟通过下一技能合同条件化的候选集决策选择合法终态/抓取/IK/准备动作。真实硬约束与底层规划保持独立，successor-effect用真实轨迹指标监督。当前已训练的是**PREPLACE一步规划成本pilot**，不是三阶段RL或物理成功学习。

## 成果2：论文架构

[SCI研究计划](02_PAPER_PLAN/SCI_PAPER_PLAN.md)：宽空间带物运输→几何验收后的柜口/槽口→换站多阶段链。强基线包括ARES-R实际规则、贪心、等预算搜索、Ranker、PPO；CASH做下一技能条件与效果头消融。贡献均标为拟验证；只有出现强基线不能解释的可接续性/预算差异后，才值得扩大学习实验。

## 成果3：真实初步结果

- 完整1.15m底盘代理：16/16端点拒绝，固定base_link净距−17.8mm。不能当学习差异。
- 历史CLEAR的1.0m下部代理：首批跨IK 8次，2条路径通过独立验收，3条因轨迹姿态拒绝，3次无返回路径记UNKNOWN；并非证明任务不可行。
- 同IK分支延拓：4/4真实带物运输路径通过模型内验收；正常/反向滚转关节路径约0.553/0.633rad，TCP均运输约100mm。
- 预冻结6目标组×2分支：12/12路径通过模型内世界碰撞、自碰撞、持物碰撞、软限位和姿态验收。所有候选均可行，仅存在成本差异。
- 训练3组、验证1组、测试2组；种子17和23；每种子普通PPO/CASH各2048次真实缓存交互。效果头实际联合更新，模型与曲线保存。

| 策略 | 测试平均关节路径rad | 每次选择规划预算 | 结论 |
|---|---:|---:|---|
| 随机期望 | 0.604477 | 1 | 真实两候选均值 |
| 固定NORMAL/关节贪心 | 0.564790 | 1 | 强基线已解释差异 |
| Ranker/PPO/CASH | 0.564790 | 1 | 与强基线持平，不是方法优势 |
| 完整两候选搜索上界 | 0.564790 | 2 | 不冒充等预算对照 |

[实际Isaac 5.1截图：STATIC_RENDER](03_PRELIMINARY_RESULTS/fig01_isaac_scene.png)、[真实持物路径](03_PRELIMINARY_RESULTS/fig02_held_trajectories.png)、[候选比较](03_PRELIMINARY_RESULTS/fig03_candidate_comparison.png)、[实际学习曲线](03_PRELIMINARY_RESULTS/fig04_real_learning.png)、[Ranker/PPO日志图](03_PRELIMINARY_RESULTS/fig05_ranker_ppo_logs.png)。[GIF](03_PRELIMINARY_RESULTS/held_path_FK_REPLAY_NOT_PHYSICS.gif)是**原始cuRobo轨迹的CPU FK回放**，不是Isaac物理/训练视频。原CPU USD预览另存，不混为Isaac截图。

[原始数据及SHA清单](../e0_20261013/data_manifest.json)、[CASH检查点](../e0_20261013/learning_seed17_v2/cash_aux.pt)、[逐场景评测](../e0_20261013/learning_seed17_v2/evaluation.json)、[环境/训练说明](03_PRELIMINARY_RESULTS/TRAINING_ENV_REPORT.md)、[根因对照](03_PRELIMINARY_RESULTS/P2_1_ROOT_CAUSE.md)。24套request/result/验收与执行源码SHA已逐项核验。

## 问题与下一步

当前学习是离线候选评估bandit，不同候选拥有不同持物起始构型；还没有同一当前状态到这些终态的真实prepare转移，不能直接当作在线技能衔接策略。真实训练链路已运行，不代表因果完整的SkillHandoffEnv已验收。

底盘上部CAD/安装接触、真实托盘抓取标定尚待解决；此次固定附着并不验证接触抓取。190单GPU12请求约197秒含两种附着/姿态模型的冷启动，实际solve中位6.81秒；不能乘4宣称四卡吞吐。216未获授权，不连接；190环境配置已授权，但驱动/重启未授权。

190个人目录已部署Isaac 5.1最小渲染环境，560驱动实际启动并输出1280×960截图，227.75秒正常退出；不等于获得官方580测试驱动支持。旧USD的BODY坐标尚未验收，托盘动态三角网格发生PhysX凸包回退，不宣称物理碰撞模型已合格。详见[仿真报告](03_PRELIMINARY_RESULTS/SIMULATION_BOOTSTRAP.md)。

下一轮最值得投入的是：先用CAD/坐标证据解决底盘代理，再冻结柜口入口及合法prepare/站位，检验局部贪心与整链目标的真实矛盾。Isaac按V0单机器人→V1 4×4 scene cloning独立物理→V2基线→V3学习推进；当前仅静态渲染PASS，V0–V4均未验收。不要继续只扩网络或在全可行E0任务上刷曲线。
