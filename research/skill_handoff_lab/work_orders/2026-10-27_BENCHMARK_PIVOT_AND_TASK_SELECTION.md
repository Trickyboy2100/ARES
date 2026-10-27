# ARES 科研补充工作单｜转向公开 manipulation benchmark，保留 JAKA/ARES-R 机器人与技能资产

2026-10-27。附属于 \`2026-10-27_AUTONOMOUS_CLOSED_LOOP_SCI_RESEARCH_AND_VISUAL_DELIVERY.md\`，在不覆盖原工作单/运行结果的前提下执行。本补充明确允许且鼓励**替换 P1/P2 自建推送任务**。用户偏好保留当前移动底盘+JAKA双臂工业机器人构型，但**不要求所有算法实验必须使用该机器人**，也不要求保留当前工件/接触任务。

## 先做研究决策：两轨而非二选一

- **公开 Benchmark 轨道**：在一套**未改动机器人、任务动力学与官方成功判据**的公开 benchmark 上，通过其标准训练/评价流程证明方法的通用性、与公开PPO/SAC/规划/技能基线做可比验证。可独立使用默认 Panda/Sawyer/Franka 等作为算法验证；不能把其成功率宣称JAKA工业结果。
- **ARES-JAKA 轨道**：用已有 Isaac Sim 5.1 + Isaac Lab 2.3.0 自定义JAKA模型/控制/Skill Contract，选择2种来自公开任务概念的**真实不同操作**迁移物理任务，证明方法可嵌入ARES-R已有工具链，所有自定义任务的结果标为 \`ARES-JAKA adapted tasks\`，**不混入官方 benchmark 排名**。移动底盘第一阶段可固定在停稳工位、后续设计合法的站位/重绑定条件；不要强制长程导航成为第一篇的训练瓶颈。
- 两轨共享**论文方法定义与输入/输出语义**（任务条件、object-centric pose、技能参数、合法动作/成功审计），不要求直接共享同一个神经网络权重；除非真实验证，否则不宣称跨机器人零样本迁移。
- 如果仅能合理完成一轨，优先形成标准 benchmark 的扎实强基线和新 RL 机制，再补 ARES-JAKA 的可运行一例；不得将两引擎重建当作新方法创新。

## 候选清单（权威代码/文档，按当前可用性排序）

1. **Isaac Lab 2.3.0 内置机械臂任务**（原有软件栈优先，用于JAKA）：
  - https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html
  - \`Isaac-Lift-Cube-Franka-v0\`、\`Isaac-Stack-Cube-Franka-v0\`、\`Isaac-Open-Drawer-Franka-v0\`，以及对应IK/Direct变体按当地实际注册检查。原官方是Franka；改JAKA后是自定义任务。
  - 先跑原任务随机/名义控制/短 PPO baseline，再谨慎适配6-DOF JAKA+夹爪/底盘，与真实机器人关节限位及夹爪物理模型一致；独立检查抓取/物理碰撞/工位可达。
2. **ManiSkill3（算法方法开发强推荐）**：
  - https://github.com/mani-skill/ManiSkill
  - https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html
  - https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/
  - https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_robots.html
  - 自带GPU并行、标准PPO/SAC基线、状态/视觉观察、真实物理任务。先 smoke \`PickCube-v1\`（校验训练器），正式候选 \`StackCube-v1\`（准备/抓取/堆叠/释放），\`PegInsertionSide-v1\`（对齐/插入），\`OpenCabinetDrawer-v1\`（抓住把手/拉开），必要时 \`PushT-v1\`。每任务明确是否具有可调技能参数、是否有可重复合法纠偏动作与强对照。用原机器人跑的才是标准benchmark数据。
  - 主仿真器是SAPIEN，**不能把SAPIEN结果叫PhysX**，也不能把其包直接覆盖Isaac51环境。隔离独立env/Docker、pin commit、记录 LICENSE/资产许可（ManiSkill资产有CC BY-NC 4.0限制，向甲方交付的使用方式需复核）。
  - ManiSkill确实可以基于URDF定义自定义Agent，但需要重新定义控制器、关键帧、传感、接触摩擦，不要首先把所有任务迁JAKA。报告名称不得用官方 leaderboard 规则描述JAKA变体。
3. **Meta-World（可选低成本多任务/迁移验证）**：https://github.com/Farama-Foundation/Metaworld ; https://farama-foundation.github.io/Metaworld/ 。默认Sawyer，MT10/MT50验证共享多任务策略，ML10/ML45验证新任务少样本适应；仅训练在MT任务不等于证明held-out任务迁移，\`MetaRL\`与zero-shot严格区分。使用官方success定义。
4. **RoboCasa365（未来长程、技能组合加强轨道）**：https://github.com/robocasa/robocasa ; https://robocasa.ai/docs/build/html/introduction/overview.html 。2026版本65原子+300复合，共365，官方2026-10已更新复合任务逐帧技能/阶段标注，适合研究高层Skill Contract/Outcome/Option，但主流baseline多是IL/VLA而非现成等价在线RL。成本大，除非现有工作方法特别依赖长程组合，否则不作为近期正方法的唯一环境。
5. **RoboTwin 2.0（未来双臂与VLA补强）**：https://github.com/RoboTwin-Platform/RoboTwin ；50项双臂任务、默认Aloha类硬件、SAPIEN。与JAKA双臂构型相关，但原机器人和执行合同不同。移植JAKA后不再是未经改变的官方同分布成绩。
6. **LIBERO（未来语言任务、VLA及长期学习）**：https://github.com/Lifelong-Robot-Learning/LIBERO ，有130任务，主要IL/VLA/ lifelong知识迁移，不能直接假设有充分调优的在线RL控制基线。VLA方法论文可用，本轮纯小规模在线RL首选并非它。
7. **RLBench（可选，但环境迁移更重）**：https://github.com/stepjam/RLBench ，广泛多任务/少样本/机器人可换，但依赖CoppeliaSim+PyRep，且官方说明改机械臂后不保证任务可解，不建议比ManiSkill优先安装。
8. **Isaac Lab-Arena（关注，不作为本轮依赖）**：https://github.com/isaac-sim/IsaacLab-Arena ，组件化任务/机器人/场景基准编写有价值，但截至2026-10其main偏向Isaac Lab3.0并标alpha、不兼容即装即用。先复用场景分解设计思路，避免升级污染现有2.3.0主实验。

## 强制一轮最小 Benchmark Scout，而不是一开始安装全部资源

在与现有Isaac训练器隔离的工作目录中完成下述**有实际运行的**可行性选择：
- Scout-1：确认IsaacLab2.3在本机可注册的Lift/Stack/Drawer任务，分别1/4 env cold reset/实际物理step、baseline、相机和video、成功谓词，找到JAKA移植是否有抓爪/Reach不可行障碍；
- Scout-2：优先安装隔离ManiSkill3依赖并运行原版\`PickCube-v1\` smoke，随后\`StackCube-v1\`、\`PegInsertionSide-v1\`（OpenCabinetDrawer视资源）；评估GPU吞吐、观察/动作形状、官方reward/term、可用scripted baseline和官方PPO/SAC checkpoint/训练脚本。记录seed、commit、许可证、真实视频。
- Scout-3：如果主研究目标转向跨任务学习，测试Meta-World官方MT10任务的创建与训练入口；若目标是复杂Skill Option长程链，优先阅读RoboCasa365复合子任务接口。**不用因“任务很多”就在没有学习机制时搬全库。**

输出 \`BENCHMARK_SCOUT_DECISION.md\` 包含候选打分：①独立方法增量机会；②公开标准可比性；③成功非饱和但真实可解；④已有强基线；⑤JAKA改造代价；⑥算法/视频训练时间；⑦多模态/Skill接口。**请选择一个公共主基准 + 一个JAKA自定义任务族，最多增加一个验证辅助基准。**

## 学术实验题目与方法建议（可自主改进）

首选三类机器人操作：
- **Stack / Place**：合法准备位姿、抓取/抬升/放置顺序、目标偏差对后续效果；
- **Peg Insertion**：进入前的姿态/入口条件、动态纠偏、任务成功与真正插入深度；
- **Drawer Opening**：工具接触区域、沿关节约束运动、开启幅度/成功与失败后再接近。

本论文方法可以形成：\`Task & observation → nominal skill → legal task-relative parameters → RL correction/skill-level outcome conditioning → physical execution → reobservation\`。重点不是任务名，而是从一个任务族迁移到不同操作时有相同科学机制：合同约束、合法动作参数调整和执行结果价值。严格根据实验改问题，不把RAPS、Fabrica、STAP、HACMan++已有的构件包装成首次创新。

最低比较：官方适合的PPO/SAC、充分调优的名义规则/脚本或规划强基线、普通ResidualRL、同合法动作范围的新方法；多训练seed，同场景pair视频与未见目标/对象参数；使用官方成功率与自定义工业独立terminal验收并**分别报告**，不能混成同一数值。

确保新公开 benchmark **确实有研究空间**：探索合理的未见任务对象/目标/入口变化和长程动作耦合，不事后修改官方任务算成原版排行榜成绩。无论控制基线多强都不能刻意削弱；若全方法饱和则另选公开不同任务，而不是让任务作弊。

## 对上一轮闭环任务书的直接修改

- R0先获得公开benchmark smoke + 任务画面，原v6/v7保留正负样例作历史；
- R1不必只查P2 case4/31，允许优先分析StackCube/PegInsertion/Drawer实际失败与动作可控性；
- R2必须在选定的公共benchmark上有至少一个实际有用的RL实验，同时并行JAKA task adaptation smoke；
- R3完成真实多seed/方法消融、同case video/filmstrip、官方benchmark表与JAKA表各自独立；
- 客户学术图文方案和论文种子随任务变化更新；**不要再把P2推板当成未来唯一创新靶标**。
- 在初步方法证据仍不确定时，先选择能够多轮闭环运行的任务和动作自由度，再扩算力；不要为“高概率出正结果”制造新优势。

**执行边界**：ARES-R只读，受控现场数据不入public Git，JAKA仿真只接受真实物理可执行动作；公开实验录像注明仿真器/任务版本/机器人、case seed、模型SHA、真实评价结果。环境分支可推研究报告，但大模型/原始视频/不再有用ckpt按前工作单留非Git与清理。
