# ARES × ARES-R：工业原子技能衔接学习研究总体路线与 Codex 执行计划

**日期：2026-10-13**。**状态：研究设计和工作单；非实验结果，非机器人执行授权。**

本文件建在**独立规划分支** \`research/skill-handoff-method-roadmap-20261013\`。用户 3080 Ubuntu 的 ARES 工作分支已有未推送本地 commit \`2b13fb9\`；不要移动/强推/覆盖主研究分支。先在 Ubuntu 原研究分支检查并正常 push 本地提交，再从此规划分支 cherry-pick 本文件提交（或者只按工作单执行）。原 ARES-R 一律只读。

## A. 已有证据与限制，不得宣称更多

当前用户报告（不是本工作单在 Ubuntu 上重新运行的结果）：

- ARES pinned assets 与 ARES-R 配置 SHA **16/16 匹配**；两个 FK 实现的 48 个测例最大位置误差约 \`2.54e-16 m\`；URDF/现场限位存在有效交集。
- 本机 L0 Python 3.13.5 + NumPy 2.5.4 + SciPy 1.18.1 可运行；旧 MuJoCo 简化环境仅完成 headless 编译和 20 个 physics steps，不能作为真实机器人碰撞真值。
- 驱动 kernel NVIDIA \`580.173.02\`，userspace NVML \`580.178\`，\`nvidia-smi\` mismatch，CUDA 不可用。磁盘 89% 使用、swap 接近耗尽。Isaac Sim 4.5/5.0 Docker 和 5.1 RC 镜像/安装存在，但没有运行/验收。
- USD WORLD↔BODY、实机最新完整 TCP、2026-10 当前现场 soft limits、相机真实 USD 外参未验。ARES cuRobo YAML 还含旧主机绝对路径。
- **现有证据不能推断出任何 RL 优势**。此前归档 smoke 只有 planner 状态，无成对候选的已验证下一技能标签。

需读的现有代码：

- ARES: \`isaac_sim/simforge/core/{kinematics,planning}.py\`、\`demos/tray_grasp_cycle/demo.py\`、\`robot/{jaka_minicobo_gripper.urdf,jaka_minicobo_curobo.yml}\`。
- ARES-R: \`src/ares_r/manipulation/transfer_chain.py\`、\`src/ares_r/task_runtime.py\`、\`src/ares_r/skills/{contracts,registry,runtime}.py\`、\`src/ares_r/motion/{runtime_goal_ik,tcp_orientation,scene_aware_motion}.py\`、\`config/{robot_collision_model,robot_world,jaka_mini2_motion.site,named_poses}.json\`。
- ARES-R research: \`research/skill_handoff_smoke/{README.md,audit_plans.py,headroom.py}\`。

**关键代码局限：** 当前 ARES-R \`TransferChainPlanner\` 用 joint-continuity / min-singular-value / joint-limit 手工评分、有限 beam 后才用 cuRobo 校验；PLACE 部分在 \`PLACE_SCENE_REBIND\` 下记录 PENDING，而非全链实际可执行证明。ARES Isaac \`tray_grasp_cycle\` 的 FixedJoint/carrier 和 cuRobo 失败后的单帧跳转**不能**产生接触/碰撞真实成功标签。早期 \`center\` 是**BODY TCP 语义位置**，绝不是唯一历史 6 轴关节角。

## B. 严格分清学术问题

研究问题（核心）：

> 在机器人和场景变化、且前一原子技能有多个合法终态的情况下，学习一个**由下游启动条件、任务剩余目标和机器人能力共同约束**的技能终态/准备选择策略，提高完整操作链的可接续性和效率，而不是重训全部底层动作。

严格区分：

1. **必须传哪些数据**（真实 joints、TCP/姿态、物体 attach、tool hash、SceneSnapshot、effect、设备状态、调用版本）→ 强类型 Skill Contract、显式约束、不学习。
2. **终态选哪个、下一个技能准备什么、何时需要观察/计算** → 可以学习的高层决策。
3. **free-space 从哪走、路径与物体是否碰撞** → cuRobo + 独立验证；强化学习无权改写硬碰撞事实。
4. **有真实抓取/放置接触效果吗** → Isaac/MuJoCo 完整动力学/实机验证；不能用瞬移或 FixedJoint 假想标签代替。

数学上可用带时间抽象的 Option/SMDP：\`s_i\` 是交接时刻已知上下文；\`a_i=(terminal_candidate, preparation_choice)\`，硬 mask 定义合法候选；执行 skill 后经过 \`Δt_i\` 得到下一状态/观测。奖励来自**独立下游结果**与成本，不来自被比较启发式的手工分数。

## C. 三条候选方法与已有论文差异

### A. 下游前提条件化的可行终态选择（最小优先）

输入：当前 q/TCP、obj/dest 相对姿态、抓取对称性、工具/持物几何、场景、下一技能 initiation/preconditions、机器人能力、任务余下阶段。候选：NORMAL/FLIPPED × 少量 center yaw/IK × 可行收臂终态。先让模型学 \`p(next_skill_feasible | state,candidate,next_contract)\` 和下游执行成本；可选小规模离散 masked Q。

**为什么低成本**：现有 \`transfer_chain.py\` 和真实 URDF/IK 可直接生成候选；多数组件已存在。**为何仍有论文风险**：STAP 已优化技能 Q 联合可行性，Text2Motion 已考虑几何依赖，GTP-FA (2026) 已考虑 grasp→plan 和失败归因。只做候选 Q-排序不足以独占创新。需要重点检验**工业 typed precondition + 多终态等价姿态 + 任务条件 + 启动状态与执行约束 + 跨场景/任务泛化**是否构成可靠增量。

必选对照：现有 \`transfer_chain._edge_cost\`、贪心、等预算 beam/DP、监督成功率排序器、STAP-style joint feasibility ranking、oracle（仅诊断上界）。

### B. 技能启动集合条件化的 Bridge Option（中成本）

如果经证据发现：仅终态候选选择不能解决 initiation mismatch，则训练一个短时 bridge policy，把当前已实现的安全终态迁移到下一个技能的可启动集合。可用目标条件化 SAC 或离散微动作 MaskablePPO，动力学来自可验证 MuJoCo/Isaac，且必须保留运动/碰撞硬检查。

**边界**：Adversarial Skill Chaining 已做终态分布匹配；2025 年 \`Curriculum-guided skill learning for long-horizon robot manipulation tasks\` 已显式使用后续技能启动分类器、价值函数和 transition policy。创新只能在**工业合同驱动的可迁移 Bridge Option、关节分支与持物/定位约束、跨设备/跨目标泛化**上找。

### C. 不确定性下的技能效果模型 + 事件触发滚动规划（高成本）

学习 \`p(s_{i+1}, outcome, duration | s_i,skill_i,params_i)\`；当 AMR 换站/附着变化/目标短时遮挡时，选择准备、观察、重规划的时机。可使用简化 SMDP model-based RL、CEM/beam rollout 和 uncertainty-aware receding horizon。SkiMo 学习 skill dynamics+skill-space CEM，TAMPURA 已研究任务与运动不确定性；要证明新颖性需有**场景版本失效与异步依赖、观察价值、跨机器人技能合同**的证据。不要第一篇三四区同时做 B+C。

**推荐顺序：A →（看 smoke 是否有 bridge mismatch）B →（数据足够再做）C。**

核心阅读：
- Sutton et al. 1999 Options: https://doi.org/10.1016/S0004-3702(99)00052-1
- STAP ICRA 2023: https://arxiv.org/abs/2210.12250 ; https://github.com/agiachris/STAP
- Adversarial Skill Chaining CoRL 2021/PMLR 2022: https://proceedings.mlr.press/v164/lee22a.html ; https://github.com/clvrai/skill-chaining
- RAPS NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html ; https://github.com/mihdalal/raps
- SkiMo CoRL 2022/PMLR 2023: https://proceedings.mlr.press/v205/shi23a.html ; https://github.com/clvrai/skimo
- Curriculum-guided skill learning, Robotics and Autonomous Systems 192 (2025) 105032: https://doi.org/10.1016/j.robot.2025.105032
- Text2Motion: https://arxiv.org/abs/2303.12153
- GTP-FA (2026 preprint; code availability unconfirmed): https://arxiv.org/abs/2606.03385
- MaskablePPO: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/docs/modules/ppo_mask.md

**开源复现纪律：** STAP 参考技能 Q/采样规划设计；Skill Chaining 参考 termination/initiation；SkiMo 参考 skill-dynamics/rollout。它们各有旧依赖，不要直接 pip 安装到现在 Python 3.13 的 L0 环境，也不要整体覆盖 ARES 框架。

## D. 统一研究基座分层

- **L0 CPU 运动学与情景层**：ARES pinned URDF、真实双臂 BODY 基座变换、与 ARES-R site soft limits 的交集，解析抓法/center 多IK、端点限位/姿态以及轻量独立几何；不宣称物理成功。
- **L1 GPU 真实规划层**：CUDA 恢复后用 ARES-R \`SceneAwareMotionService/cuRobo\` 及 collision/dense/level validator 检查关键非接触段；每个 PICK/PLACE 都有**各自的 scene 和基座变换**。仅 planner SUCCESS 不等于 native packager/pass、也不等于 grasp success。
- **L2 物理仿真层**：重建并验证的模型；旧 MuJoCo 简化物理 20 步仅是 host health。若 Isaac 4.5 headless/10GB 显存不够，保留 MuJoCo 本机 URDF/MJCF 对齐模型，Isaac 用来可视化/少量验证。不启动无界多环境物理 RL。
- **L3 实机**：最后阶段少量受监督运行，现场独立授权；不会在这台 3080 的研究脚本里直接发送硬件命令。

数据对象：

\`\`\`text
Scenario { scenario_id, task_family, seed, robot_revision, tool_revision,
           T_body_base[], pick_scene_digest, place_scene_digest,
           object/destination relative pose, obstacles, attachment_geometry,
           next_skill_requirements, site_soft_limits, provider_revision }
Candidate { candidate_id, symmetry_id, center_tcp_goal, center_yaw,
            ik_branch/q_terminal, preparation_policy, hard_validity_mask }
Transition { scenario_id, candidate_id, start_state, observed_terminal_state,
             successor_state, actual_duration, source_revisions,
             success_geometric, success_planning, success_physics, success_real,
             failure_class, unknown_reason, planner_calls, hashes }
\`\`\`

必须把 \`UNKNOWN_TIMEOUT/ERROR/OOM\` 与 \`FAIL_COLLISION/LIMIT/IK/ORIENTATION\` 分开；一个 source hash/scene ID 对应的重复规划次数不能冒充独立随机样本；同场景所有候选共用生成布局与评价预算。缓存键必须包含 scene/robot/tool/site-policy/candidate/planner revisions。

## E. 8 个可交付阶段（每阶段都有代码、数据和结果，而不是永远审计）

| 阶段 | 主要工作 | 可交付、验收依据 | 是否必须 GPU |
|---|---|---|---|
| **P0 主机与对齐收口** | 保存本地 \`2b13fb9\`、GPU mismatch 只读归因、磁盘/Swap 审计、USD BODY 映射与最新 site cfg 缺项表 | \`GPU_DIAGNOSIS.md\`、\`ALIGNMENT_DELTA.md\`；当前资产/FK PASS 保留 | 否 |
| **P1 实验合同与 reset** | 建 \`SkillHandoffScenario/Candidate/Transition\`、gymnasium-like \`reset/step\`、显式 semantic effect/scene rebinding；先只用 CPU | \`scenario_schema.json\`、\`generate_scenarios.py\`、test 不变性/多 seed/replay；能输出两任务族 | 否 |
| **P2 几何问题存在性** | 6 场景×4候选 sanity；扩大至 30–50 个独立场景；保证候选真实 URDF/双臂/持物近似几何/软限位一致 | \`candidate_feasibility.jsonl\`；对同场景不同候选的后续状态、失败原因进行对照 | 否，可 CPU |
| **P3 cuRobo 冻结场景 Oracle** | GPU 恢复后从实际候选小集合做 FAST→FALLBACK、dense level/collision、native limit 等价检查；单 planner owner，限时/缓存 | \`oracle_l1.jsonl\`、CUROBO timing 与硬门分层结果；不接实机 | 是 |
| **P4 Headroom + 强基线** | fixed、greedy、当前 ARES-R heuristic、equal-call-budget DP/beam、STAP-like value rank、oracle 上界 | \`baseline_comparison.csv\`、bootstrap CI、失败 attribution 与 headroom 结论 | 原则否（读缓存） |
| **P5 学习前的强空假设** | 训练监督 ranker、contextual bandit，held-out scene/task split；完全相同信息/候选/规划预算 | \`ranker_results.json\`、\`bandit_results.json\` 与泛化；若完全不弱于 RL 可作为论文主方法方向调整 | 否 |
| **P6 真正多步 SMDP-RL** | 加 2–3 次按实际新观测决策（选抓法→center终态→换站后选择再观察/准备），用 MaskablePPO 或自实现 masked Double DQN；5 seeds | \`train.py\`、\`evaluate.py\`、5 seeds+held-out、学习曲线、对 bandit/DP 的差异及消融 | CPU 优先，GPU 可选 |
| **P7 仿真/实机/论文闭环** | 在 L2 物理仿真验证不由几何代理夸大效果；少量受监督 ARES-R 以及另一机器人适配；论文方法、消融、局限 | 路径轨迹/任务验证、置信区间、跨场景/任务、论文图表及独立可复现实验包 | 根据后端 |

这些是建议顺序和工作包，不是已经完成的事实。**P0 GPU 修复与 P1/P2 CPU 研究可以并行**。P0 不应成为 L0 研究启动的阻塞门。

## F. P1/P2/P3 的关键任务约束

### P1：让实验可以自动复现而不是只能手工拼场景

1. \`research/skill_handoff_lab/bench/contracts.py\`：typed \`Scenario, Candidate, Transition, Outcome\` 与枚举/版本。
2. \`bench/generator.py\`：seeded 目标位姿、障碍、两站位/搬运物随机生成；T1 \`tray→groove\`，T2 \`tray→cabinet-preland\`。后者来自十月 archive 的 Epic object=4、70mm preland，不能和更早 object=3 groove 混为一谈。
3. \`bench/geometry.py\`：复用 ARES-R \`RuntimeGoalIK\` 或等价 pinned ARES FK/IK、碰撞件；不能拿 ARES 旧 MuJoCo 简化体当真值。
4. \`bench/successor.py\`：下一 Skill 的启动/终止 predicate 明确可检验；特别是 center 为 Cartesian 区域、多 IK/可变 yaw；附着物已创建，place 在新场景判断。
5. \`bench/env.py\`：\`reset(seed,scenario_id)\` 和 \`step(candidate)\` 分明；第一版可单步，但这是 contextual candidate selection，不能宣称充分 RL。
6. 验收：同 seed/hash 必定同场景，改变一个 terminal 候选不改变场景；缺字段/旧场景/碰撞无效必须拒绝；无任何真实控制器导入。

### P2：验证研究问题在真实几何下是否真的存在

- 先用 6×4 样例确保 pipeline；然后至少 30–50 **独立**情景，每情景 4–8 合法候选（不要硬编码数字为论文结果）。
- 关键区别：前置技能可成功且终态合法，但后续 successor 的初始集合/轨迹可行性不同。真正同场景 paired comparison，不把不同场景成功率直接比较。
- 验证 actual q_terminal、soft joint limits、双臂与 chassis、payload geometry、水平姿态、next skill semantics；不能用仅 TCP xyz 点验证。
- 标签层：\`GEOMETRIC\`、\`PLANNED\`、\`PHYSICS\`、\`REAL\` 分开；P2 的 CPU 结果不能叫 cuRobo oracle。
- 若候选完全没有差异，先调整具工业依据的任务布局/末端约束而不是伪造失败。若差异存在但手工 heuristic 一样最优，研究价值转向给定预算的加速或后续不确定性。
- 统计：每场景 oracle 可达 upper bound 与 heuristic 的差值，bootstrap 以 scenario 为单位重采样；统计 invalid/timeout 单独报告。

### P3：GPU 与 cuRobo

1. 只读检查 kernel-module、NVML、\`dkms status\`、\`dpkg -l\`、\`ldconfig\`、boot/kernel logs；不直接 \`apt purge\` 或 \`rmmod\`。
2. 用户同意并保存会话后，若是升级留下旧已加载模块，可在受控窗口重启；若重启无效再查重复安装/DKMS/混用 runfile，出计划要求确认。不要边跑任务边改内核驱动。
3. GPU PASS 后先跑 cuRobo 单个 FK/IK/碰撞/路径示例；不要先起 Isaac GUI。工作盘 89% 用量，先只读找 10GB+ 大文件清单，指定可删除范围后再清理。
4. 从 \`bench\` 导出任务、已知几何以及两工位独立 scene，使用 **同一个 real-tool/site-soft-limit/level/collision** 约束；序列以 frozen source hash 绑定。
5. 只做候选中最有价值的少量 cuRobo solve，FAST→FALLBACK 至多一次；结果如果 \`OOM/TIMEOUT\` → \`UNKNOWN\`，不能视为硬不可行。

## G. P4–P6：学习实验协议

### 核心指标与公平性

- \`HSR\`：前一技能完成后，下一技能**可实际启动**的比例；\`planning-HSR\` 与 \`physical-HSR\` 不得混报。
- \`full_task_success\`：所有技能效果已验证，是真实仿真或物理结果，不是仅一段轨迹成功。
- \`task_time\`、\`cuRobo_calls\`、规划消耗、准备被作废次数、失败原因、持物水平误差、最小净距。
- 同一 \`scenario_id\` 的所有候选、不同 seed 重复、同一父场景微扰应放在**同一个 train/val/test 组**；场景 ID、候选排序结果、未来真实 obstacles 不得进入 policy 观测。
- equal compute：对 greedy/heuristic/DP/ranker/RL 提供相同候选列表、合法 mask、观测、规划预算，评估也报告推理/准备时长。Oracle 消耗全候选真值，是上界而非同预算可部署 policy。
- held-out 包含 (a) 不同布局和未知障碍，(b) 新目标姿态高度，(c) 新技能组合/目标设备，(d) 第二 embodiment（后期）。
- 初步 \`5 seeds\`、scenario-level bootstrap 95% CI；正式数量以变异/计算预算/先导试验确定。

### RL 状态与动作

**不能只让 RL 作一次终态选择并立即结算**，那是候选排序或 bandit。多步 SMDP 例子：

\`\`\`text
D0 观察 PICK → 选 symmetry / 预抓取终态
D1 抓起并获真实 q+attach 结果 → 选 center 终态/姿态分支与准备策略
   （下层 cuRobo 求实轨迹并执行相应高层 option）
D2 AMR 到 PLACE（旧 scene 失效）→ 新观察 → 选是否补充局部观察或启动合法 place 方案
D3 验证后续 Skill success, cost
\`\`\`

三次决策都只允许在当前可观测信息上做；\`a_i=(valid_terminal_id, prep_choice)\`，\`mask(s_i)\` 来自确定性规则、资源合同与当前硬门。

**奖励示例（需实验标定）：**
\`R = 1.0*I(verified full task success) - λ_time*(task_seconds/ref_time) - λ_plans*planner_calls - λ_reobserve*reobserves\`。不得把 \`transfer_chain\` 的启发式评分当回报/真值；不把碰撞当可交互探索惩罚，非法动作直接 masked/rejected。半马尔可夫按选项持续时间 \`γ^(duration/reference_step_time)\` 折扣。

先训练/评价：
- 监督 \`p(success|state,candidate,next_skill)\` ranker;
- contextual bandit；
- MaskablePPO (SB3-Contrib; \`action_masks()\` 和正确 mask-evaluation callback) 或自实现 masked DQN；
- 再视结果做带 learned skill dynamics 的简化 MPC。**不要直接将官方旧 SkiMo/MuJoCo2.1 或 Skill Chaining+Unity 环境原封不动安装进新的 Python3.13 L0 venv。**

成功的科学证据不是「PPO 曲线增加」，而是（1）有真实候选敏感性/可改进 headroom；（2）在相同 planner budget 和任务观测条件下，学习策略泛化强于简单启发式与监督单步选择；（3）如果宣称 RL 有必要，还需证明其多步延迟决策优于 bandit。

### 失败/转向条款

- 若 \`oracle≈heuristic\`：仅学习排序的上限不足，转**减规划调用成本、观测/计算调度**或更真实 initiation mismatch；不能为论文故意削弱 heuristic。
- 若 \`supervised ranker≈RL\`：论文定位为下游条件化可行性学习，而不是强行称 RL 方法；RL 可作为消融。
- 若 terminal choice 无法克服 initiation mismatch：切 B 的 Bridge Option；局部过渡必须通过真实约束。
- 若场景变动/观测延迟主导：切 C 的 partial-observation / effect model 研究，仍使用同一 Bench。
- 若仿真好、实机差：优先修机器人 TCP/手眼、真实载荷和碰撞模型，不将 domain gap 写成 RL 成功。

## H. 论文路线

目标甲方应用型三/四区 SCI 的最小完整证据：

1. 真实工业任务数据/场景与技能 API，可通过独立仓库重放；
2. **一种明确提出并实现的交接选择方法**，而非 KG+LLM+cuRobo+PPO 模块罗列；
3. 相同 robot/site model/任务输入下，与手工启发式、监督候选排序、等预算全链规划对比；
4. 随机种子/场景数明确、参数/失效归因/消融/新工位泛化、至少有限真实现场验证；
5. 不伪称首次研究 skill chaining 或首次下游可行性；
6. 明确资产/代码版本、仿真固定附件 vs 接触真实世界差异。

写作素材应从第一天保存：\`fig1_system_arch\`，\`fig2_terminal_candidate_and_initiation_region\`，\`fig3_paired_scenario\`，\`fig4_training_learning_curve\`，\`fig5_generalization\`，\`table1_main\`，\`table2_ablation\`，\`table3_real\`。

## I. 工作顺序总结（给 Codex）

1. 先完成 **P1/P2 CPU**，让实验真正生成候选与对照结果，同时单独审计 GPU 修复；不要被 NVML 失配卡住全部工作。
2. 这周优先交付**至少 6 个可复现情景、每个 ≥4 候选的真实 URDF/FK/限位/碰撞分层结果**，再启动 P3 cuRobo。
3. 每轮只提交一个工作包及证据，不擅自继续到实机。
4. 本文件不是命令授权；任何新实机运动、宿主机大规模包操作/重启需用户现场批准。
