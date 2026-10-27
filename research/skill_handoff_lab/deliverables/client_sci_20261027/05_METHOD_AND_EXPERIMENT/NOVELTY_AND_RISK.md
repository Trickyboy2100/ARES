# 最接近工作的机制重叠与论文风险（2026-10-27）

## 结论先行

**目前可交付的是可复现应用研究初稿，不是已证实新算法或可保证录用的稿件。** 普通残差在历史未见几何上有真实正结果，但P2只有6/6/7对反馈5（各16例），大部分宏平均提升来自P1。统一任务坐标、加入几何条件、PPO残差、额外预测头都已有先例，不能独立承担原创性。新pilot优先检验“实际持续接触结果是否能改变后续参数选择并带来同预算P2增益”；没有独立消融时只标PROTOTYPE/PROPOSED。

以下每项固定四行：准确来源；机制及实际代码；重叠/可迁移；差异、代价与许可风险。只分析八篇最近邻，不重新无界调研。代码阅读不代表已运行对方基准或复现其指标；不复制论文图、摘要、模型或数据。

### 1. RAPS

1. **Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives**，Dalal / Pathak / Salakhutdinov，NeurIPS 2021。[论文](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html)。
2. 手工动作基元类型和参数构成RL动作接口；[官方代码](https://github.com/mihdalal/raps) `rlkit/rlkit/envs/primitives_wrappers.py` 的 `SawyerXYZEnvMetaworldPrimitives` / `RobosuitePrimitives` 明确映射基元ID、参数索引并执行多物理步；Kitchen/MetaWorld/robosuite使用不同RL实现。
3. “RL调合法技能参数、复用既有技能”直接重叠；可迁移接口与基元执行日志，不可称首次参数化技能。ARES只复用思想，现有Isaac/JAKA代码独立实现。
4. 我们拟研究自由工件持续接触结果驱动的后续参数，不是更换技能动作接口；必须证实额外收益。移植整套旧MuJoCo/图像训练成本高；仓库MIT不自动覆盖所有子模块/资产，未下载其checkpoint或资产。

### 2. STAP

1. **STAP: Sequencing Task-Agnostic Policies**，Agia / Migimatsu / Wu / Bohg，ICRA 2023，DOI 10.1109/ICRA48891.2023.10160220。[论文与方法](https://sites.google.com/stanford.edu/stap/home)。
2. 独立训练技能Actor/Q、转移动力学及不确定性；联合成功以Q乘积估计，规划优化参数。[代码](https://github.com/agiachris/STAP) `stap/planners/{cem.py,utils.py}` 的 `evaluate_trajectory` 与elite更新真实使用后续价值而非闲置头。
3. 后续技能价值、避免短视与技能组合直接重叠；可借鉴等预算CEM/搜索、Q校准和分布外诊断，不能将“引入下一技能成功”称首次。
4. 本轮若仅持续控制残差，不是STAP式技能序列规划；若拟引入suffix值，须证明在Actor更新/选择中的独立作用。官方MIT；其PyBullet资产/已训技能不能当ARES实验证据；整套模型迁移成本中高。

### 3. Residual Skill Policies / ReSkill

1. **Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics**，Rana / Xu / Tidd / Milford / Sünderhauf；**CoRL 2022会议，PMLR 205于2023出版**，2095–2104。[正式来源](https://proceedings.mlr.press/v205/rana23a.html)。
2. 状态条件skill prior引导潜技能探索，低层残差适应新任务；[代码](https://github.com/krishanrana/reskill) `reskill/train_reskill_agent.py` 将decoder动作与residual相加，两个PPO实际更新；`reskill/rl/agents/ppo.py`保存GAE/截断bootstrap。
3. 条件化技能与低层残差已有完整前例；可迁移动作/残差公平消融思想，不能将普通ResidualPPO命名为全新方法。
4. 我们不训练演示VAE/prior；若有增量应是接触结果校准下的参数/后缀机制及新物理证据，而非“名义+残差”。Fetch/MuJoCo旧环境与数据成本中高；代码许可及外部演示数据需逐项确认，本轮未导入。

### 4. HACMan++

1. **HACMan++: Spatially-Grounded Motion Primitives for Manipulation**，Jiang / Wu / Zhou / Paxton / Held，RSS 2024。[官方方法页](https://sgmp-rss2024.github.io/)。
2. 对点云位置、基元和连续参数联合建模；[代码](https://github.com/JiangBowen0008/HACManPP) `hacman/algos/location_policy.py` 计算per-point/per-primitive Q并采样或argmax，`multi_td3.py`等训练相关critic，另有多仿真后端。
3. “选择接触点+动作基元参数”已被研究；ARES若拓展接触点不能只把手工接触位置参数当创新。可借鉴空间落点可解释图和动作可控性配对测试。
4. 本轮仍是state-based紧凑控制，不是已验证点云策略或零样本实机；未运行其基准。复现完整视觉/点云栈成本高；仓库/第三方子模块资产许可需单独确认，未复制其模型/图。

### 5. GRN

1. **Learning Geometric Reasoning Networks For Robot Task And Motion Planning**，Smail Ait Bouhsain / Rachid Alami / Thierry Siméon，ICLR 2025。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/561dd71d04695c85e1c0f6a0fbe0bd85-Abstract-Conference.html)。
2. 用IK、grasp obstruction、action/grasp feasibility模块构造可解释几何预测；[代码](https://github.com/Smail8/geometric_reasoning_networks) `Network.py`中的 `IKModule/GOModule/AGFModule/GRN.forward`真实传递预测，`train_grn.py`加载模块并训练图模型。
3. 执行资格/IK/阻挡预测不是ARES原创；可用于规划候选排序对照，不能把分类准确率当接触RL成功率或物理安全保证。
4. GRN主要是监督几何可行性而非持续接触反馈控制；我们若研究物体动态结果，需有独立的物理机制差异。数据/预训模块与机器人适配成本中；官方仓库和模型卡要求按LICENSE核对，本轮未引入网络/数据。

### 6. Fabrica

1. **Fabrica: Dual-Arm Assembly of General Multi-Part Objects via Integrated Planning and Learning**，Tian等，CoRL 2025，PMLR305:406–419。[正式论文](https://proceedings.mlr.press/v305/tian25a.html)。
2. 分层装配规划结合几何等变残差接触策略；[官方代码](https://github.com/yunshengtian/Fabrica) `learning/isaacgymenvs/tasks/fabrica/fabrica_task_assemble.py` 的 `_preprocess_actions`执行path-frame变换和名义残差相加；`fabrica_fixplug_task_assemble.py`执行目标控制与post-physics刷新。
3. “规划+task-frame+几何泛化+残差”高度重叠，是本轮最强原创性风险；统一坐标只能作为工程基线。可借鉴接近/接触分工与路径归一化，但不得宣称首次。
4. 本任务是自由工件推送，不是论文的双臂已持件装配。其某些变体直接同步plug pose，**不能移植为自由工件训练或抓取成功证据**。旧IsaacGym/AutoMate及SDF适配成本高；主仓MIT，learning另有LICENSE及资产条款。

### 7. RecoveryChaining

1. **RecoveryChaining: Learning Local Recovery Policies for Robust Manipulation**，Vats / Jha / Likhachev / Kroemer / Romeres，IROS 2025，DOI10.1109/IROS60139.2025.11245856。[论文](https://merl.com/publications/TR2025-152)。
2. 从真实名义失败状态训练恢复，混合动作含继续调用整段名义策略；其后缀实际执行产生稀疏回报，Lazy版本用高precision分类器节约rollout。[代码](https://github.com/shivamvats/recovery-chaining) `scripts/learn_recoveries.py`收集失败、训练/执行recovery、fit nominal value并调用nominal skills。
3. “后缀执行值、恢复后交回名义技能”已有直接前例；恢复机会不能由软件异常或不可能状态伪造。可借鉴失败库及额外rollout预算核算。
4. ARES目前hold/proceed没有证明恢复优势；不应称已复现RecoveryChaining。若走恢复，需真实可恢复接触状态及等预算比较。旧robosuite和稀疏任务复现成本中高；代码/资产许可需明确，论文IEEE版权不授权复制图文。

### 8. TASP

1. **Task and Skill Planning: Hierarchical Robot Planning with Black-Box Skills**，Hedegaard / Wei / Yang / Jaafar / Tellex / Konidaris / Shah，ICRA 2026；采用[Brown作者正式PDF](https://cs.brown.edu/people/gdk/pubs/tasp_icra.pdf)及[arXiv v3](https://arxiv.org/abs/2504.17901v3)。旧v1题名不同，不能混用版本。
2. 将异构黑盒技能包为CIP，以kinematic envelope和head/tail motion plans桥接，ATAM式分层细化/回溯组合；论文给出双臂和移动操作实验。公开项目入口 `benned-h.github.io/tasp/`本轮无法访问，**未核验其关键代码路径/许可证，不臆造代码复现**。
3. 黑盒技能库+规划衔接和Skill Contract系统设计直接重叠；可迁移head/tail接口，但工业集成本身不能代替新的RL方法贡献。
4. 本轮不是TASP planner复现，也未验证双臂黑盒长程泛化；差异应落在可学习持续接触参数及独立对照。代码/资产可得性和许可是复现风险，暂只阅读论文，不将其结果列作我们的baseline实测。

## 可发表性判断与明确反证

| 候选差异 | 当前定位 | 成立至少需要 |
|---|---|---|
| 普通残差比反馈好 | VERIFIED历史初证，不是原创算法 | P2重点、多种子、新scene冻结及强反馈 |
| 统一task-frame | 工程/归纳偏置基线 | 反馈和RL共同修正，B1/B2同预算 |
| 任务合同字段拼接 | 条件化接口，非自动创新 | 相同信息/容量的有效条件、假条件和去条件对照 |
| 执行结果进入后续Actor | PROTOTYPE/PROPOSED | 同起点真实动作改变、真实后缀回报、去outcome/去next对照 |
| 后续头loss降低 | 不构成策略贡献 | 移除头后行为/任务指标存在配对差，额外交互计入预算 |
| 共享接触技能泛化 | 待验证应用贡献 | 至少两种非饱和物理接触条件，场景隔离，非标签重命名 |

本轮若B3没有对B2形成配对P2收益，则保留负结果，不扩五种子、不写“显著超越”。可交甲方可信方案、编译初稿和真实pilot；投稿前至少补足机制独立作用、多种子P2主表、去outcome/去next消融及新的未见几何/动力学评价。不能凭文字“执行结果感知”避免与上述最近邻的重叠。

## 本轮实际pilot后的更新

修正后的正式v7有界pilot已完成：10个训练模型、163,840次正式在线交互；另保留并计入两次索引修正前调试训练32,768次，**本轮总训练196,608次，不把验证运行再次累加**。27项完成运行的独立验收齐全，主图/表由`06_RUNS_MANIFEST/NEW_PILOT_RESULTS.csv`与原始run index自动生成。

- 新P2验证各模型同32例：强反馈B0 **30/32**；B1 **30/29/30**；B2 **30/30/30**；B3 **30/30/30**。B3与B2三个种子的成功集合相同，不能凭均值包装方法改善。
- 同checkpoint的8维历史遮蔽、仅测得响应历史遮蔽、部分next-goal遮蔽均保持相同30个成功案例；独立no-next训练seed31也30/32。这不证明后续信息普遍无用，只说明本pilot没有其独立贡献证据；共享56维仍包含当前状态反馈，partial mask未删除控制器已知任务方向。
- 同冷启动、连续物理执行下历史遮蔽引起了微小参考/物体轨迹差异，但未救回或丢失成功案例；这不是性能增益，也没有将数值敏感性与单独动力学因素完全分解。
- 固定接触目标−4mm **30/32**、+4mm **29/32**；后者丢失case17，没有救回案例。它们是有实际物理成本的参数可控性探针，不是免费三次搜索或新的成功标签。

因此**G3未满足，不扩五种子、不读取封存最终测试来挑方法**。保留真实训练能力、参数接口、独立验收、漂亮真实场景与已有几何泛化正证据；本次甲方交付是可靠研究方案和可编译论文初稿，尚非拥有经验证新方法优势的可投稿定稿。下一步应先增强真实接触/姿态可控性和验证非饱和后缀机制，而非把一段历史拼接改名成原创算法。

新验证组与历史v6b旧Test-Geometry不是同一场景分布；不能将新30/32与旧P2的6/16直接相减声称“方法提升”。
