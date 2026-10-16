# 研究主线重选：公开机制→ARES接触组合任务

2026-10-16，执行基线4ea1461，继承c98d112；不回滚历史。10月Preland是独立工程回归，不再自动等同论文科学问题。本轮不继续WideCabinet或六例缓存PPO。

## 精读记录与复现边界

已读取下列论文的方法/实验段、官方README及所列关键函数；代码commit和文件SHA见本地`runs/skill_composition_20261016_r0/REFERENCE_SOURCE_INDEX.json`。这里的复现成本是针对当前ARES环境的工程判断，不是已测训练耗时。未下载上游大数据、checkpoint或整套旧运行时；不能把阅读代码写成已复现其论文成绩。

|工作/准确来源|机制与实际代码|任务、基线和公开资源|ARES迁移、重叠与成本/限制|
|---|---|---|---|
|[STAP: Sequencing Task-Agnostic Policies](https://arxiv.org/html/2210.12250)，ICRA2023|`stap/planners/cem.py::plan`用技能动力学滚动候选；`planners/utils.py::evaluate_trajectory`将各技能critic的成功概率组合为整链概率；不是额外预测头独立降loss|PyBullet Pick/Place/Push/Pull；DAF、随机/CEM及不确定性规划；官方README列约10GB技能/动力学/checkpoint下载|借鉴完整链评估、相同预算搜索与技能接口；不移植旧PyBullet/SCOD全栈。技能Q、后续价值、组合成功概率均已有先例。MIT；旧Python3.8，完整运行成本中高|
|[Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation](https://proceedings.mlr.press/v229/chen23e.html)，CoRL2023|论文§4.2使用终态10帧预测transition feasibility，向前初始化、向后奖励微调；§4.3用于实际切换。`policy_sequencing/transition_value_trainer.py`读取真实成功/失败HDF5，训练BCE分类器；`scripts/bi_optimization.py`组织训练|BlockAssembly、ToolPositioning；IsaacGym Preview3/4，Allegro；官方Drive checkpoint。README称每子任务约1–2天，非我们实测|借鉴真实前段终态初始化后段、速度/短历史、整链验收；不能宣称首次用可行性决定切换。旧手模型与IsaacGym需大改，不装作IsaacLab原生移植。根LICENSE为Apache2.0，README末尾却写MIT，记录冲突不擅自统一|
|[Value-Informed Skill Chaining for Policy Learning of Long-Horizon Tasks with Surgical Robot](https://arxiv.org/html/2307.16503)，IROS2023|`viskill/agents/sc_sac.py::update_critic`在子任务边界用下一子任务actor/critic构建Bellman目标，`update_actor_and_alpha`实际最大化该Q；末技能固定。不是只拼接合同|SurRoL BiPegTransfer/BiPegBoard/MatchBoardPanel；DEX、GCBC、T-STAR；论文5种子，子技能示教约200条、约2M步；本次未核到单独公开权重下载并验收|可迁移跨边界价值目标、固定底层技能与可学习子目标；不直接移植手术模型、MPI/HER整个环境。MIT，需SurRoL子模块；接续价值本身不是我们的创新|
|[Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html)，NeurIPS2021|`rlkit/rlkit/envs/primitives_wrappers.py::act`解码primitive索引及其参数，再真实执行；包装器累积底层奖励。论文讨论长短动作比较应同时记墙钟和底层交互|MetaWorld、Kitchen、Robosuite，Dreamer/PPO/RAD等与raw/VICES/SPiRL比较；本次不声称已运行作者checkpoint|借鉴小型参数化动作库与实际持续时间记账，不宣称动作基元新颖。根MIT，vendored子项目/资产另有许可；旧MuJoCo/Gym依赖不整体安装|
|[RecoveryChaining: Learning Local Recovery Policies for Robust Manipulation](https://merl.com/publications/TR2025-152)，IROS2025（2024预印本）|论文§V中名义后缀Option被真实执行以生成终止奖励；Lazy版本用高精度分类器减少昂贵执行。`belief_srs/skills/skill_chain.py`按interval累计折扣，`recovery_skill.py`训练PPO，`subtask_wrapper.py`重置实际失败状态|Robosuite PickPlace/Shelf/ClutteredShelf，Nominal、离线precondition、primitive recovery；代码有采失败/恢复入口。未下载验收权重|B路线最贴近现有名义技能；恢复机会必须来自真实接触/滑移，不是软件错误。根目录未发现LICENSE，不直接复制代码；旧环境和路径硬编码需重写。混合恢复动作/交回名义控制已有先例|
|[Skill-based Model-based Reinforcement Learning](https://arxiv.org/html/2207.07560)，CoRL2022（SkiMo）|`skimo_agent.py::estimate_value/plan`在技能潜空间展开动力学、奖励和末端critic，以CEM选择技能；离线技能与模型联合训练|Maze/Kitchen/CALVIN；Dreamer、TD-MPC、SPiRL及组合；`pretrained_models.md`公开预训练入口|只借鉴模型误差与长程代价评测。当前没有证据需要世界模型，暂不移植。MuJoCo2.1/Python3.9/多子模块；根目录未发现统一LICENSE，成本高|
|[Compose by Focus: Scene Graph-based Atomic Skills](https://arxiv.org/html/2509.16053)，2025预印本|论文是聚焦场景图+语言条件diffusion/BC，并非技能RL。`tabletop_skill_compose/*`分别给原子/组合物理任务，`examples/motionplanning/panda/solutions/*`给名义脚本|CubeOutAndIn、ToolsUsage、ObstacleAvoidance等。实读Cube组合`evaluate`将success固定False且dense reward=0；不能直接视作开箱即用RL环境。源码还在类定义时加载CLIP|借鉴多物体持续状态、工具/绕障和物理名义轨迹，不引入VLM前置依赖。Apache2.0，第三方资产另查。README称ManiSkill2但代码含新GPU API，需逐项兼容核对|
|[ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI](https://arxiv.org/abs/2410.00425)，RSS2025（2024预印本）|`mani_skill/envs/tasks/tabletop/push_cube.py`建立自由动态工件，以物体真实位置/高度验收；`examples/baselines/ppo/ppo.py`正确处理终态观测和并行评估|GPU异质场景、状态/视觉PPO/SAC等；官方文档有数据/基线入口，本次未安装或报告其标准benchmark结果|直接借鉴自由工件、reset、独立物体判据及训练预算协议。代码Apache2.0；README资产CC BY-NC4.0，不能因代码许可推断资产同许可。采用自己参数化几何，避免平台/资产迁移|
|[RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots](https://github.com/robocasa/robocasa)，RSS2024；RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots，ICLR2026（官方当前README）|`environments/kitchen/kitchen.py`组织场景，`utils/subtask_labeling/base.py`明确原子阶段、连续片段、DONE及边界校验，不把静态场景数量当任务成功数|多阶段厨房、pretrain/target场景和对象split、示教/子任务标注；资产下载约10GB（README），不在190直接下载|借鉴任务族与场景/物体隔离、子任务标注及人工复核。代码MIT，当前README资产/数据CC BY4.0；不同版本仍需核查。作为任务设计参考，不作为新主平台|
|[Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0)，已部署软件|实读`direct/factory/factory_env.py`的控制信号、深度/XY成功与结束评估；`direct/forge/forge_env.py`的真实力观测、噪声与随机化。使用DirectRLEnv/InteractiveScene而非16个独立进程|Factory PegInsert/GearMesh/NutThread及Forge；官方RL配置/资产入口；本轮复用已部署5.1+2.3，不宣称复现原论文成绩|复用接触物理、并行scene cloning、reset/日志和标准RL。代码BSD3，NVIDIA及第三方资产另有条款；已有JAKA模型不必重装所有机器人/环境|

## 主线选择与有界探索

**优先A：非抓取接触技能组合中的执行结果与后续可行性。** 原因不是“RL不可替代”，而是自由工件的姿态、速度、接触侧与前段终态会跨阶段保留；静态末端IK/避障规划不能当作物体接触结果标签。已有JAKA、受限Cartesian控制与真实PhysX可立即复用。首轮先用连续PPO建立可训练基线；尚不是新CASH方法，也不是离散Option算法。

新建两个ARES任务，不冒称公开benchmark：P1工作台两段工件推送（物理/训练验证）；P2准备/对准→槽口送入（实际自由工件与导向接触）。保留旧双臂/现场回归，不将它们混入新任务统计。工件不是焊接物，策略不能重写其状态。成功由工件位置、速度、支撑高度、阶段历史及机器人约束共同决定。

**紧密备选B：**从P2真实卡接触/方向或姿态偏差中采集仍可恢复的状态，加入撤回、改变接触侧与重新送入；必须先用真实名义后缀检查恢复空间。根本几何不可能、超时与软件异常分开。

**备选C：**若接触推送路线仍不稳定，使用已部署Factory/Forge机制建设局部装配，先单环境标准配置，再替换JAKA；不回到宽柜口刷步数，不让某个Preland故障决定是否研究。

## 不能提前称为创新的内容

技能价值、终态可行性、短历史、名义Option恢复、残差控制和合同拼接均已有充分先例。候选增量必须体现为真实策略行为与公平任务指标，例如：在同一观测/动作/训练预算下，用经真实后缀执行校准的接续价值影响实际交接/恢复选择。仅训练预测头、仅换任务名、仅得到PPO权重更新都不成立。

当前边界：控制器、接触和任务可完成性仍需实测；任务随机化、正式scene-group train/validation/test、原规划/搜索基线及多种子/消融未完成前，不预填性能优势。具体运行结果另按不可覆盖目录保存，不把本文件当实验成果替代品。

## R2后依据真实结果修正主线

16环境新接触任务与三种子PPO已完成。冻结G1/G2各模型约62/64例都到达第二阶段，主要差异在最后推进和稳定停住，而非第一阶段是否可达。故不能把本轮结果解释为已经验证A的非短视准备/接续机制。短期主问题收敛为**接触状态持续下的阶段切换与稳定完成RL（A/C交界，更接近局部接触控制C）**，保留同一工件状态的阶段接口；B恢复需另做真实可恢复性试验。

阶段回报混合seed31与全阶段MC最终权重相同，未隔离出阶段贡献；不将该原型包装为论文主创新。下一项实验先检查旋转/角速度观测、真实终止稳定性和足够强的接触控制，再用同一起点、不同合法交接/恢复和真实后缀执行验证机制。该调整依据真实失败分布，不是回到RL必要性论证，也不继续宽柜口饱和训练。
