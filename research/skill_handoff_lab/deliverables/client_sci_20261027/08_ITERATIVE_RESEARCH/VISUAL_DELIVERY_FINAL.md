# 实际物理与在线训练可视化交付

## 入口与唯一媒体版本

交互入口：`index.html`。非Git原始相机、模型、状态与完整运行日志位于研究根目录`runs/client_sci_closed_loop_20261027/`。

可移植打包仅选择`paired_videos/comparisons_final/`，不要同时打包保留的旧`comparisons/`展示版本。

| 最终媒体相对目录 | 内容与选择规则 |
|---|---|
| `R0_case_000/paired_actual_physx.mp4` | 历史seed41，P1首个反馈失败/残差成功；非新方法 |
| `R0_case_001/paired_actual_physx.mp4` | 历史seed41，P2首个反馈失败/残差成功 |
| `R0_case_003/paired_actual_physx.mp4` | 历史seed41，P2首个共同失败 |
| `R0_case_009/paired_actual_physx.mp4` | 历史seed41，P2首个共同成功 |
| `R1_case_008/paired_actual_physx.mp4` | 新R1首个共同失败；两策略均25/32 |
| `R1_case_000/paired_actual_physx.mp4` | 新R1首个共同成功 |
| `R2_case_005/paired_actual_physx.mp4` | 新R2首个共同失败；两策略均24/32，反馈25/32 |
| `R3_case_005/paired_actual_physx.mp4` | 预指定seed31，首个执行资格版本救援；仅案例机制示例 |
| `R3_online_training_12288/actual_online_training_16env.mp4` | 独立真实在线训练重执行，12288额外交互，非正式性能模型录像 |

前8项为1920×1080、H264、30fps真实闭环配对；最后一项为1920×1080、H264、12fps、15.834s，按相机记录物理时间重采样并明确标记结尾展示保持。所有原始相机帧/轨迹保留，没有用状态回放冒充物理执行。

## R3结果边界

预声明训练seed31、相同32开发案例、相同87维状态/5维动作/24s合同：普通PPO22/32，执行有效维度似然PPO24/32；3个救援、1个退化。相机只跟预指定案例5，不能从无相机的案例10伪造退化视频；案例10及全部32结果保留在开发矩阵和CSV。

三seed完整结果是普通PPO22/25/20，候选24/24/18。候选均值68.75%±10.83pp（样本SD），普通69.79%±7.86pp；强反馈26/32。不存在可重复的整体提升。相同32案例重复用于不同训练seed，不计作96个独立场景；这是开发集，不是密封最终测试。

这里的“救援/rescue”仅表示同案例中一方法失败、另一方法成功的统计配对结果，不等于策略学会了恢复技能。

## 真实训练与矩阵画面

`figures/R3_online_training_16env_montage.png`及同名PDF：单个Isaac应用中16个独立物理环境在同一tick1536的16相机视角拼版，不是单个相机全景，也不是16次独立应用、静态模型复制或固定动画。各环境关节、工件、阶段及速度原始状态随帧保留。

`figures/R3_online_training_wide.png`：同一训练应用的真正单相机全景；保持原物理2.5m环境间距，因此机器人显示较小。

`figures/R3_optimizer_before_after.png`及PDF：第1/3次真实优化前后RGB与参数张量哈希。优化期间物理tick/全部记录状态保持完全一致，优化后网络权重变化；后续物理步才使用更新后的策略。

原始训练`r3/visual_training_qualified_seed31_12288_retry1/`：真实12288交互、1536全局向量物理tick、三次rollout优化、288小批量optimizer更新；120Hz只读训练sidecar维度1536×16×44。记录位置/姿态/速度/动作/实际参考/控制器状态，不将`env.evaluation`改为true来借用评估轨迹。最终CPU加载checkpoint的排序state_dict张量哈希与最后优化后哈希一致。

首轮失败保留于`r3/visual_training_qualified_seed31_12288/`：满首rollout4096交互后，原始审计保存模型时遇录制闭包序列化错误；0次optimizer更新。仅修复录制包装器，在调用原算法期间移除包装后finally恢复，未跳过原审计、改算法/任务/物理。4096交互额外计费；512向量tick/8192env-physics步为明确计数换算，含reset的总物理tick未知不猜测。详见该目录`OBSERVED_FAILURE_BUDGET.json`。

## 图表与打印

`FigR_method`为7in紧凑源图，当前真实87D/5D共享执行器与执行有效维度似然进入PPO比值；熵系数0，不将熵写入PPO比值。无学习型高层切换、Q头或硬件安全声明。

`FigR_online_learning`与`FigR_development_cases`为7.2in打印源图；含全部完成正式训练的种子、固定刷新消融、强反馈及共同失败。旧84D诊断和12288训练可视化重执行不混入正式学习曲线。曲线来自真实已结束训练episode，点为全部episode、线为预声明8episode滑动均值，不生成假数据或跨seed置信带。

`ITERATIVE_FIGURE_PROVENANCE.json`绑定522个正式已结束episode、512条实际独立验收开发案例及源SHA。表`ONLINE_TRAINING_EPISODES.csv`和`DEVELOPMENT_QUALIFIED_CASES.csv`可以重画；接触/角速度/实际工具yaw与指令yaw分开记录。

## 验证与限制

`VIDEO_QA.json`：8组配对、10个独立录制进程的模型/源/物理方法AST/相机/媒体/初态一致性验证。

`TRAINING_VIDEO_QA.json`及`TRAINING_VIDEO_PROVENANCE.json`：额外真实训练的输入输出SHA、源AST一致性、CPU checkpoint加载、真实优化前后权重与物理状态、120Hz训练轨迹和媒体规格。

外观美化前后物理/几何属性身份一致；额外render不推进physics。P1/P2为自建自由工件接触任务，不等于真实料盘CAD、实机抓取、现场柜体放置或完整机械臂碰撞/硬件安全认证。上述验收是采样几何/任务/工具检查，非连续CCD或全自碰撞证明。没有读取最终测试以挑选新超参数或视频seed。
