# 两条可执行研究方案：暂不定死主方法

## 判断

当前A更贴近已报告的现场问题（Center/Transit/Preland、IK及原生J3拒绝、换站场景重绑），但10月原始数据尚不在设备，不能说实际学习数据已经充分。B具备更充分的可复现软件/PhysX基础（既有16env、真实PPO与残差、过程检查），但WideCabinet不是现场任务，缺最后放置几何/接触合同。两者都不应依据二级19条统计直接训练；无需先证明非RL不能解决才开展研究。

## A：技能边界/准备/有限规划预算Option-RL

**状态**：当前实测q与active TCP/附着物、左臂HOLD_CURRENT、已观察的工位几何与不确定性、scene/base/observation/calibration版本、姿态与软限位裕量、剩余预算、名义技能阶段。PICK阶段不读取尚未观测的PLACE障碍。

**动作**：Nominal规划调用、已真实实现的Prepare过渡目标/合法IK分支、合法standoff/入口方案、有限预算内Replan；换站只调用已有安全技能并强制失效旧scene。Reobserve仅当已有可用观察流程与合理观测代价时作为Option，不生成虚假新观测。

**转移/奖励**：OracleExecutor复用真实cuRobo请求及持久服务；Prepare必须从同一当前状态实际规划到准备态，再评估下游，不能把不同起点缓存冒充选择。SMDP按真实Option持续时间累计成本；完整链独立/native验收成功奖励，真实调用/时间/路径成本扣分。UNKNOWN保留并报告数据不足，不能填成失败；planning-only与物理执行两级分开。

**约束**：同版本world、TCP、物体、水平/yaw、native joint/velocity/acceleration、场景重绑和全链碰撞；硬mask只采用当前已知可验证条件。使用已有CASH Gymnasium接口，不另建框架。先标准MaskablePPO/SMDP Actor-Critic；候选集效果头仅在真实转移足够后加入，不能以预测loss冒充机器人RL。

**对照/实验**：原ARES-R TransferChain/Beam、固定Prepare、局部贪心、等预算搜索、Ranker、RL；相同起点/约束/调用预算，按场景组split。原301软限位拒绝是资格回归样本，不是正执行标签。最小试验取得三种Prepare的成对完整链（固定8–12场景，有限seed，不挑有利样本）。

**当前可运行性**：FK/IK/Oracle软件可复用，原档导入后能先进行planning级回放；实际在线学习转移不齐。四次固定seed历史模型IK探测在reported Preland/target各得2个soft-bounded端点解，其中1个Preland解贴近J1软边界；只说明旧模型假设下非显然不可达，不证明路径、当前工具、碰撞或10月问题已解决。

## B：从实际Preland开始的受约束局部操作/恢复RL

**状态**：真实可观察q/qdot、标定TCP/对象姿态估计、已知目标/开口几何、水平与前向yaw合同、接触传感（如实际可用）、时间/剩余深度/误差与有限恢复标记。特权PhysX真值仅用于reward/诊断或明确标注critic，不当成部署Actor输入。

**动作**：有限Cartesian XY/Z残差；若合同允许才加入小yaw修正，roll/pitch始终保持物体水平；名义下降/对齐控制＋硬约束。70mm垂直Preland偏移不证明最终路径必为纯-Z，先核验原任务接口与接触顺序，不沿用旧+X柜口进入。

**奖励/终止**：基于实际物体（非只TCP点）的放置/姿态/深度/接触与时间，独立过程碰撞/穿透；成功、过大接触、限位、workspace、timeout分开。可靠附着可以先保留，但不宣称接触抓取/释放已经实现。

**实现**：复用现有IsaacLab16env、PPO、残差/合同策略和过程审计；新增site profile、双臂外观/左保持及真实物体/工位。先1→4→16独立物理实例，不启动16应用。旧checkpoint是研究实现资产，不能不改观测/任务就称现场策略。先名义反馈/标准PPO/普通残差/同容量blind/full合同，公平预算和预声明多seed。

**增量与边界**：拟检验技能合同条件化跨真实几何、物体/持物偏差和必要恢复；残差叠加及工业装配RL不是首次思想。[Residual RL](https://arxiv.org/abs/1812.03201)已有控制与学习叠加，[IndustReal](https://arxiv.org/abs/2305.17110)已有接触装配训练与动作积分；不能把基础结构当新贡献。

**当前可运行性**：16env物理与训练代码最成熟；actual Preland、CAD/ear/attachment/开口未知，不能直接作为现场接触任务训练。缺数据期间可继续明确标记PARAMETERIZED_RESEARCH的通用任务开发，旧结果保留、不描述为真实任务解决。

## 推荐执行顺序

先补原档做A的小规模一致约束机制诊断；同时按下份工作单把B物理对象和坐标层修好。依据真实转移/接触证据选择论文主线，而非预押注CASH/SAC/MPC或不停刷宽口PPO。
