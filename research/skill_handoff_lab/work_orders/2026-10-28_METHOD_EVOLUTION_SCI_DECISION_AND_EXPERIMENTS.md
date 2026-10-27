# ARES 甲方 SCI｜研究方法演进与下一轮实验任务（2026-10-28）

> 研究决策文件 + Codex 自主执行起点。新分支从 `6af76af6f8f1f85cde612a1fb1cb2a573c93cc60` 建立；不改历史结果、ARES-R生产代码、真实机器人或受限现场数据。允许自主改变物理任务与 RL 算法，但需要连贯研究假设、真实执行、强基线和证据链。此文件不是“保证论文正结果”的承诺。

## 0. 当前可靠科学结论（先理解，不再重新刷 P2 无效似然）

- **ARES 自建 P2/R3**：32 个共同开发案例，强反馈26/32；同87D/5D、24s、32,768交互/seed的普通 PPO：22、25、20/32；执行有效维度似然 PPO：24、24、18/32。后者确实改变真实 Actor 更新，但总体没有稳定成功率增益；不能继续把该模块包装成核心科学贡献，封存最终测试不可用于选方法。
- **ManiSkill3 StackCube-v1**：原生50-step完整成功判据，同一64开发案例：结构化反馈58/64；普通 Residual PPO（31/37/43）61、62、60/64，每种子256k在线交互。相对反馈均为正（+3、+4、+2 例），但多数差距很小；三个种子各救回**相同5例**且分别造成2、1、3例原本成功案例退化；强制 residual=0 重回58/64。原 joint PPO1.024M为0/64、另 Cartesian PPO256k为0/64，但由于动作/探索接口和超参不等价，不能当成“已经证明RL只能通过残差”的证据。
- **重要可研究空间诊断**：就当前已经观测的 nominal 与 residual 配对而言，两个策略的成功并集为63/64，故即便完美按case切换，最佳也仅能从各 PPO 的60–62提升到63/64，这是**当前两个既定策略的可见oracle并集**，不是全体算法的理论上限，更不能在64例数据上训练gate后称未见泛化。仅六例 nominal 失败，当前Stack不适合作为唯一正方法主测试。
- **ManiSkill3 PegInsertionSide-v1**：已完成本机真实GPU物理planner/obs适配探针；4例中规划器延长诊断完成4例，原生100step内success-ever仅1例。其他例在约108–161步才完成；超过官方预算不能记官方成功。没有充分调试且成规模的原生强基线，也没有已验证Peg RL增益。
- **Isaac Lab/JAKA**：JAKA + EG2 物理跑720步，但限位/mimic和速度验收FAIL；未完成JAKA抓取，不可借Franka结果叫工业实机/ JAKA成功。ARES-R的SceneSnapshot / SkillRuntime / cuRobo 保持只读系统背景，论文一开始不用逼自己完成全部移动底盘+双臂任务。

**阶段判断：研究基础正进展已成立；原创新算法独立收益尚未成立。本文件不新增训练，本轮方法讨论的成功率变化 = +0 pp。**

## 1. 重定科学问题（先不要命名网络）

> 给定已能完成多数常规操作的名义技能/反馈控制，直接残差强化学习既能救回一部分失败，也会损害原本成功的执行。面对新对象、目标、几何/接触条件，如何以**真实可控的技能参数和执行结果**为依据，学习**何时/如何使用修正**，同时减少无必要的干预与完整任务回归，并提高未见条件的任务成功率及效率？

主张不应是“PPO加个gate”“加入历史就适应”或“首次利用后续成功”。RAPS、STAP、ViSkill、Residual Skill Policies、Fabrica、RecoveryChaining、Foresight Residual RL (arXiv 2607.16506)、SCOUT (CoRL2026)、U-GRA (Machines2026)、Demystifying Action Space Design (ICML2026) 等已经涉及参数化、后续技能价值、动作反馈、残差门控、坐标空间。需先独立对照，并以新任务/消融结果确认本工作的具体价值。严禁声称首次提出已存在的通用构件。

## 2. 推荐的科学演化：行动效应→差分任务价值→审慎选择（候选，不是已经发明）

先按最小可行路线；其中任一机制若只是重复论文先例且不产生独立任务增益，及时放弃，不必实现全家桶。

### M0｜真实可控的参数化 Skill（不可跳过）

将已有规则/规划技能视为名义 `u_nom`，对每个具体 Skill 定义合法参数合同 `C_k` 与物理参数 `xi_k`，而不是任意扩大末端坐标：

- Stack：pregrasp XY/standoff、lift/pre-place高度或偏置、放置速度、释放时间、retreat等选择其中1–3个真实可执行参数；必要时沿用20Hz/50 step与原任务成功逻辑，不能把增加episode长度当原版收益。
- Peg：抓取预对位、入口位置和 yaw、轴向推进、重接近与短段回撤等；必须真正使官方100step内可执行且可达，不穿透peg/插孔。
- 每个参数真实作用于相应末端/抓爪控制，记录 `request → clipped → applied → object/grasp effect`。Baseline拥有同样合法控制权限；`xi` 为空转/被裁剪掉不得算策略起效。
- 原生 ManiSkill 任务不改官方物理、reward、horizon、success；若建立更困难的同任务分布，独立命名 OOD/industrial variant，不冒充官方排行榜。

任务条件 `o_k=(robot, object/goal relative pose/velocity, phase, C_k, C_{k+1}, previous observed effect, remaining budget)`。部署Actor不输入未来success、仿真privileged真值（除显式 state-based baseline）或后验全轨迹标签。物理执行后 `e_k=features(s_{k+1})-features(s_k)` 与实际接触/成功谓词构成观察反馈。

初版用既有 residual PPO/SAC 处理连续合法参数；**任务失败仍采用官方终态评估**，附实际耗时、失败类别。先通过同初态物理探针证明某参数对后续结果具有双向可控效应（不仅负作用），再扩训练。

### M1｜配对 outcome difference：学习“何时值得偏离名义”（先验证可学性）

训练环境中对独立采样的状态/任务条件 `c` 分别从相同合法reset、独立冷启动真实执行名义策略 `pi_0` 与候选参数/残差策略 `pi_theta`，得到真实完整任务 `Y_0,Y_theta` 和代价 `T_0,T_theta`，构建正/负/零**配对增益**标签：

`D(c,xi) = Y(c,xi)-Y(c,xi0) - lambda*(T(c,xi)-T(c,xi0))`

这与只训练绝对成功预测不同；目的是避免已能成功的case被不必要修正而退化，增强跨任务决策的**效用比较**。对于有多阶段决定的episode，在合法 Skill 边界进行结果观察→下一技能参数选择；如只有一次决定，应如实称 contextual bandit，不伪称长程SMDP。

可用一个 calibrated pairwise-advantage / value model `D_phi(o,xi)` 与不确定性估计；策略在估计真实收益不足或不确定时可以`abstain`并执行名义技能，在有充分实证的情况下选择有成本约束的修正参数。对使用的任务Q值/下游预测必须真正进入动作选择或Actor训练，不能只是一个不影响policy的预测loss。

**不规定必须用哪个gate/critic、也不宣称其原创。** 原始R3的inactive-dimension PPO梯度方法已验证没有稳定优势，不以它作M1唯一实现。先比较更简单的监督paired ranker、固定阈值gate、Q-uncertainty gate和等预算搜索（STAP/GRN/U-GRA/SCOUT等最接近想法优先），如果一个simple ranker足够解决，论文不应硬包装成多步RL原创。

注意：双跑需要**显式计入2倍物理交互成本**；不能从测试case训练gate；训练时可使用完整后缀标签，部署决策只能看当前观测；不存在跨已分叉轨迹“强行保持后续状态相同”的因果说法。

### M2｜有效 Skill 修正/恢复选项（仅在M0证实物理机会后）

可选`KEEP_NOMINAL`、`PARAM_CORRECTION`、`REAPPROACH`等少数受实际物理许可的动作，执行成功前应有合法终止和状态回读；是否将离散选项与连续参数交给PPO/SAC还是模型辅助MPC，按真实消融决策。

目标是：与普通ResidualRL比减少退化与昂贵重试，而不是声称绝对安全、形式验证或在线概率保证。要比较同动作空间规则控制、普通ResidualPPO、单次paired ranker、简单skill-level contextual bandit、至少一种search/CEM；若存在连续重试才需SMDP预算等价。

## 3. 核心奖励与训练目标

默认主目标：独立 `terminal_success`（原生Stack/Peg success）+ 带完整失败的实际执行时间 + 规划/重试次数及所需物理成本。可以在新方法和普通RL对照中共同引入正确终止处理的potential-based shaping，但不可通过奖励补丁与新动作权限一起报算法收益。

推荐将“对名义成功执行的退化”作为**配对评测与离线/辅助约束**，不是在普通单轨迹PPO奖励中偷看另一路未实际执行的nominal结局。一个示例研究目标是：
`max_pi E[Y_pi - lambda*T_pi - mu*N_retry] with penalty for estimated P(Y_nom=1,Y_pi=0)`
其中真实退化项仅在paired训练集可标注；部署用有校准风险的预测而非真值oracle。最后报告success和regression/rescue分别多少；统计种子、cases/scene group、配对效应与95%CI。

## 4. 实验推进路线与继续/收缩条件

### E0（立刻，证据检查 + 物理机会）
- 旧Stack64案例全对照热图和5个rescues/1–3 regressions原型视频，明确63/64是**现有两个策略的oracle并集**；不从这64案例拟合后给出“未见gate提升”。
- 用512–2k **全新独立train conditions**的随机化物理探针（可按实际成本缩减）验证Stack release/placement参数以及Peg approach/yaw/correction参数在合法条件下导致正负的后续任务效果；控制器和原生终态不改变；指定train/validation/密封test scene seeds与group哈希。
- **Peg原100-step是否存在已充分调试、能够在预算内完成的强规则/planner+feedback基线？** 已知4例 planner在108–161步完成不能当native pass。可优化合法技能路径/阶段时间，但不得修改任务原生100step；如强基线始终为0，需要重检动作设计，别拿0做论文胜利证据。若任务本身不合适，StackPyramid/Drawer等可作为更有机会的任务族。

### E1（第一轮方法实验）
- 锁定一个非饱和主任务，动作权限/强规则/评价协议**同时固定**。同seed/预算训练普通ResidualPPO（合合同的可调整参数）和M0物理参数策略，比较真实物体state效果与终态。
- 如果收益来自新增动作权限，而普通Residual用同样动作也能达到一样结果，则写“更有效的动作接口”，不是虚构新RL损失。
- 至少3 seeds×64–128新的开发案例，保留case-level rescue/regression/失败类型，播 paired视频。如果方向稳定，优先验证第二任务是否共用参数条件化框架。

### E2（差分增益/后续状态实验）
- 在独立训练物理轨迹采集成对nominal/candidate outcome，训练候选保守选择/参数价值策略，进行`no effect`、`no next contract`、`no pairing`、`no abstain`、`same action/extra capacity`等消融。优先比较简单监督ranker/contextual bandit及等预算搜索。
- 若gate仅对原Stack64案例拟合成功却在未见条件失败，视为未验证。若普通规划/监督方法相同或更好，应改写科研主张（例如sample efficiency/执行成本），但更改主指标必须在新的下一轮独立验证前冻结。

### E3（成熟论文正方法）
- 方法固定后，至少5独立训练seed，在完全不参与选择的封存测试评价2种具有不同物理机制的操作（首选Stack与Peg/Drawer），报告native success、零样本任务参数变化、训练/推理物理成本、速度与失误、paired置信区间；不要将同64 cases的三seed结果当192独立场景。
- 若JAKA/EG2仍mimic limit FAIL，只用它作受限应用adapter和可视化，不能标为成功硬件迁移。等物理验收合格后才加入JAKA Isaac操作结果；ARES-R作为Task Runtime/SceneSnapshot/技能合同参考接口，公开site数据仍禁用。
- 要有不同任务变体、非privileged的结构化感知或传感误差鲁棒性消融，才能在合同意义上支撑“多模态具身感知”；别要求第一篇训练LLM/VLA。

### 可自行探索、但不得无界扩大
Codex可以调SAC/PPO、任务参数空间、接触/夹爪/释放控制、探索与curriculum、物理预算，在有效失败机制上推进2–3轮真实闭环；每轮至少留一个实际新PhysX/SAPIEN运行、两组强基线、实际行为图、方法决策。建议先用0.5–1.0M额外物理训练交互的有界先导预算；接近上限且确有合理机制时汇报扩展需求，不默认永久后台自主跑GPU。不产生正结果也继续改实际物理问题，但不通过挑场景/放宽强规则或终态制造论文柱状图。

## 5. 文献最接近的竞争解释（Codex需检查原论文方法与代码）

- RAPS NeurIPS2021：参数化动作接口；https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html
- STAP ICRA2023：后续技能Q值/参数搜索；https://sites.google.com/stanford.edu/stap/home
- Fabrica CoRL2025：几何task-frame+residual RL，不能将此二者相加称新；https://proceedings.mlr.press/v305/tian25a.html
- Foresight Residual RL arXiv2026-07：后续技能成功价值；https://arxiv.org/abs/2607.16506
- SCOUT CoRL2026：action-outcome dynamics belief在线适应；https://arxiv.org/abs/2609.36107
- U-GRA Machines2026：twin-Q disagreement gating残差；https://www.mdpi.com/2075-1702/14/8/924
- Action Space Design ICML2026：delta/joint/task表示实验；https://proceedings.mlr.press/v306/feng26ab.html
- RecoveryChaining IROS2025：名义技能的局部恢复；https://github.com/shivamvats/recovery-chaining
这些思想都已发表/公开，**新稿的创新应落在清晰可区分的损失/决策接口以及至少两个任务族上真实重复表现**，不是单个功能命名。

## 6. 论文与导师可见交付

在现有`research/skill_handoff_lab/deliverables/client_sci_20261027/`的真实研究材料上增量建立`10_METHOD_EVOLUTION_20261028/`：
- `SCI_CLAIM_AND_NEAREST_NEIGHBORS.md`：标题、科学问题、新颖性对照及可反证假设；
- `BENCHMARK_PROTOCOL.json`：官方任务判据/时长/seed/训练预算、scene split SHA；
- `CONTROL_AUTHORITY_MAP.csv`与物理对照视频：改变合法技能参数后工件真实末态/执行预算有何变化；
- `EXPERIMENT_RESULTS.csv`、`CASE_PAIRED_CONFUSION.csv`、`RESULTS_TABLE.tex`：主表带每seed+不确定性；
- `Fig1/2`系统与方法（标实现和拟实现），`Fig3`paired filmstrip，`Fig4`物体响应/成功轨迹，`Fig5`跨seed/任务学习曲线，`Fig6`消融及regress/rescue风险；
- 真实SAPIEN/PhysX视频须标仿真器；所有画面配正确模型、scene、实际episode和case原始SHA；视频选择同时有rescue与regression及总体概率；不可用状态动画假装新物理训练；
- 甲方中文版PDF、英文两栏LaTeX原稿、实验/创新段落同步修订；一页中文报告包括成功了什么、失败了什么、为何转向及下一步仅做什么。
- 保存模型和原始数据于非Git工作区，核对sha与loader；清理无用大型checkpoint与冗余资产先保留唯一证据、正式负结果，实际释放空间报告。不得将受控生产/现场数据push公开仓库。

**本轮优先级：首先证明多阶段技能参数确实能影响完整后续任务结果；其次让一个有真实动作权限的RL改进普通ResidualPPO的可靠性和泛化；最后才决定是否称为新算法并扩大训练。**
