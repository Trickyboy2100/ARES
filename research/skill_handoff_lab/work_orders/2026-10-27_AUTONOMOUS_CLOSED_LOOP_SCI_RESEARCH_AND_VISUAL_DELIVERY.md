# ARES 甲方 SCI｜自主研究闭环 + 实际 PhysX 对照视频 + 论文方法成果：Codex 执行工作单

**制定日期：2026-10-27。执行权限：研究代码、受控仿真、合法 GPU 训练、数据处理、论文写作与本研究独立分支提交。禁止 ARES-R 生产修改或无授权实机动作。**

**执行分支：** codex/ares-sci-autonomous-research-visual-20261027，从已交付 SCI pilot 的 ARES commit 0de3e77662919e8914bc7d112edbfc4598a79c31 创建。切换前检查远端和本地 HEAD、dirty files、其他人的作业；必要时建立单独 worktree，而不是覆盖正在进行的工作。

**这是一份带研究自主权的工作单，不是固定算法实现清单。** 允许你通过真实失败分析、资料、可执行物理对照、训练证据改变方法甚至更换研究任务；但必须围绕甲方合同和最后能发表的 RL 方法问题不断迭代。目标不是让任何命名网络“获胜”，而是找到一个物理上合理、科学上可区分、能够复现实验、可嵌入 ARES-R Skill Runtime 的真实有益方法。

## 0. 当前真实基础与上一轮负结果：开工前必须理解

甲方合同主题：《多模态具身感知与基于强化学习的仿人机械臂控制策略研究》。导师关注任务→感知→技能库/执行决策→规划和机械臂控制的闭环。本论文不需要同时训练 VLA、LLM、KG、MPC，但必须能说明 RL 具体处于哪一层、怎样复用工业技能和任务感知条件。

**权威来源：**
- research/skill_handoff_lab/deliverables/client_sci_20261027/00_README_EVIDENCE.md
- research/skill_handoff_lab/deliverables/client_sci_20261027/05_METHOD_AND_EXPERIMENT/TRAINING_DECISION.md
- research/skill_handoff_lab/deliverables/client_sci_20261027/05_METHOD_AND_EXPERIMENT/NOVELTY_AND_RISK.md
- research/skill_handoff_lab/deliverables/client_sci_20261027/02_PAPER_SEED/main.tex、sections/results.tex、sections/method.tex
- research/skill_handoff_lab/reports/CONTACT_HANDOFF_RESULTS_V6_20261016.md
- research/skill_handoff_lab/reports/P2_1_SCIENCE_DECISION.md
- research/skill_handoff_lab/tools/contact_composition_v6.py，cash 代码，v7 精确运行源码快照、NEW_RUNS_INDEX.json、FIGURE_PROVENANCE.json；对比必须按每个 run 实际 source SHA，不能把当前 HEAD 当历史运行源码。
- ARES-R（只读）：SkillRegistry / SkillRuntime / SceneSnapshot / TaskStudio / Scheme / cuRobo / JAKA 控制器，仅作架构来源，禁止向 public ARES 仓库迁入现场受限数据。

**v6b 历史证据（自建 STATE-BASED ACTUAL_PHYSX，不是实机）：** 每个冻结 Geometry split 32 例，阻尼反馈8，普通残差 seed31/41/51=13/18/23；P2 拆分每16例普通残差6/6/7，反馈5；很大一部分优势来自 P1。普通 Residual PPO 的物理学习和有限几何适应确实得到初证，不是原创 RL 已成立。

**v7 最新证据：** 新增196,608次真实在线环境交互（其中正式163,840，诊断32,768）；真实16env，12模型。新 P2 开发验证每 seed 32 例：充分修正的 B0 反馈30；B1 BODY residual30/29/30，B2 task-frame residual30/30/30，B3 task-frame+8D执行历史30/30/30。遮蔽历史与部分 next-contract 不改变成功集合，+/-4mm 接触参考改变工件状态但没有救回此前两例；旧撤回策略不能救回。**B3未优于B2，且新验证组显著饱和。不能把B3写成方法贡献；不能把新30/32与旧5/16跨分布相减。** 新封 final test 未启封，不得用于方法搜索。

**当前科学瓶颈：**
1. 对现有可控参数，正向改善下游成功的机会很少；“改变了网络输出”不等于能改变合法工件状态/救回失败。
2. P2 hard failures 在阶段1目标外约29–38mm、接触代理峰值45–59N，不是单纯稳定窗口/等待问题。
3. 任务坐标或多加历史并非独立科学贡献，与 Fabrica、Residual Skill Policies、STAP、HACMan++、RecoveryChaining 等有明显重叠。
4. 旧 CASH 少量离散候选曾被 Beam/DP/枚举压倒，不能直接改成高层 PPO 就称突破。
5. 甲方已经得到9页中文方案、可编译两栏稿、真实 PhysX 图表与研究简报；本轮应**增量更新这份交付**，不是另起一份只有概念的全新工程大报告。

## 1. 本轮真正的科学问题与论文候选定位

**稳定的问题陈述：** 当物体、目标、站位和后续技能约束发生变化，名义机器人动作技能即使规划合法，其实际物理执行结果也可能让下一阶段难以完成；如何借助物理反馈和受约束的 RL 参数决策，在有限资源内形成对新任务条件更可靠、可复用的操作技能？

论文方向不锁死：
- **方法线 A（首选）**：Task-Conditioned, Physically Grounded Skill-Parameter Adaptation；从名义技能/反馈控制上学习合法、能实际改变作用方式的操作参数，使物体轨迹、接触/姿态与后续可行性改善。
- **方法线 B（必要时）**：Outcome-Aware Skill Transition / Corrective Option；从实际技能执行结果决定是否改变准备/接触/重建动作，优化后续完整任务，而不单独预测一张不进入 Actor 的价值图。
- **方法线 C（如果接触推送仍无法区分）**：在已有 JAKA/ARES-R 控制接口下研究持物预放置、近端对齐、工具方向、障碍扰动等**新的、物理真实且独立的任务族**。不要被最近接触路线困住，但不能无依据声称已在真实现场闭环。

**标题仅为研究草案：**
CN-1《面向多阶段机械臂操作的任务条件化技能参数自适应强化学习方法》
EN-1 “Task-Conditioned Reinforcement Learning of Executable Skill Parameters for Adaptive Multi-Stage Robotic Manipulation”
CN-2《面向动态执行偏差的接触感知参数化技能强化学习方法》
EN-2 “Contact-Effect-Aware Parameterized Skill Reinforcement Learning for Robust Robotic Manipulation”
从实际正结果与对照选择最终标题。若只在 P2 完成局部接触实验，不得使用暗示整个移动双臂多模态系统验证成功的标题。

**可能的新颖性切入（均是假设）：** 把当前受 Skill Contract 约束的可执行操作参数与**已实际发生的物理效应**及下游任务完成联系起来，使策略优化的对象从仅当前位置/目标误差改变为“哪些合法接触、准备和执行参数让整段任务实际完成”；效果模型/后续值若使用必须真正进入参数决策或策略梯度，且优于普通强反馈、Residual PPO、等预算搜索、适当监督 Ranker。RAPS/HACMan++/STAP/ViSkill/RecoveryChaining 都已覆盖相近思想，科研差异须由具体机制与独立实验建立，不能仅在模块名上创新。

**可发表性的核心测试**：同一合法控制接口与同一非饱和、物理可行任务族上，实测比公平强基线更可靠完成、或相近完成但在时间/规划调用/交互预算上显著更好；跨训练种子与未见参数，且有能解释优势来自何种物理行为的消融。必要时在第二类技能任务验证可复用性。不要要求“唯有 RL 可以”，但不能把0.2微米轨迹变化写成算法正结果。

## 2. 给 Codex 的自主研究权限、禁止事项和迭代哲学

**允许独立决策：** 新动作空间或合法接触技能参数、任务设定与 curriculum、奖励塑形、网络与观测表示、PPO/SAC/TD3/混合技能 Option/短视 MPC 结合、物理交互频率、场景难度梯度、公开代码局部借鉴、算法参数调优、对照控制器增强、非 privileged 感知估计、评价脚本、摄像机/可视化改进。每个变化在实验日志中写清原因/机制/计算成本/影响哪些baseline；初始路线不用执着，不需要每次换轻微超参数都停下等待回复。

**研究闭环必须实质性完成：**
观察真实失败与基线弱点 → 提出可反证的物理/学习假设 → 有界修改环境/可执行动作/策略 → 对强基线和RL做真实物理执行 → 输出指标/接触/轨迹/视频 → 去核心模块消融 → 决定继续、替换机制或转换任务。在一个执行会话内优先完成 **至少三轮有区别的真实研究迭代**；若现有任务再次全面饱和，允许有明确物理理由地更换任务族并继续迭代，而不是只写 Gate 阻断报告。原理上不承诺某次实验一定正，但每轮须有代码/真实运行/视频/决策。

**禁止：**
- 不能降低独立完整任务成功判据、隐藏碰撞/关节约束/禁用强控制器以制造优势。
- 不能在真实 PhysX 动作过程中传送/重置自由工件，不能将机器人命令幅度当成物体真实位移。
- 不能拿伪随机图、插值动画、纯轨迹回放、脚本最终态/训练视频替代真实物理策略评价。若使用状态复播仅能标明为可视化，并与源物理轨迹哈希绑定。
- 不能借已经查看过的 v6/v7 测试集调新方法，也不能多个训练seed共用一组case后当作独立场景数。
- 不能选择最好的 checkpoint/seed/好看的个例而宣称总体优势。
- 不能公开 ARES-R 受限现场点云、生产配置、地址、真实设备读数、派生现场图；ARES-R repo read-only；不能驱动机械臂/夹爪/底盘、修改现场 SDK 保护或重启机器。
- 不装不兼容依赖污染 Isaac Sim 5.1 / Isaac Lab 2.3.0 训练环境；开源仓库先读、pin SHA / LICENSE / submodule 许可，适配核心算法不盲目大迁移。
- 不追求“必须低于100%才能发”：64/64真实性能必须报告，但若整个任务都饱和，需另找有物理意义的困难任务族。

## 3. 可先验证、但允许更好的替代设计的研究假设集

### H1｜Physical control authority：接触作用点/姿态/可重接近参数能否实际修复失败？

基于 P2 case4/31 及失败类别，首先比较局部位姿/末端 yaw、允许推板横向偏置、接触侧、后退—横移—重新建立接触等合法短段；机器人与工件全程物理连续，保留实际关节驱动、碰撞、推板/原夹爪模型。使用同冷启动原始 case，多个 action 参数配对真实执行实际后缀，不可只改中间工件 pose 并假设后续成功。

关注的不仅是 XY 推进量，而是**接触作用点能否可控地产生纠偏的角度和横向效果**，哪些情况下更适合解除接触并重新准备，是否存在不严重损害其他已成功案例的合理参数域。可视化接触力/方向、有效横向余量、目标偏差、真实工具/工件路径。如当前自由度在物理上无法修复偏差，允许扩充物理合理、JAKA可执行的局部工具姿态参数或设计另一操作任务。仅“某个参数使已有成功失败”不是足够的正机会。

### H2｜Learned corrective parameters：从真正有用的可执行动作空间学习

基线用充分调试的 task-local 反馈；另加同动作/条件的固定规则、有限参数枚举或CEM/MPC（计算物理查询计价），普通 Residual PPO。学习候选可以是残差 SAC/TD3、受约束 PPO、参数化 Option Actor；根据观测真实物体响应选择推进、纠偏、重新定位或必要时撤回。优先可复用现有训练器，不为算法名无端重写引擎。动作投影与执行限幅统一，对照不得使用更弱的可行动作范围。

奖励主目标是已通过独立审计的完整物理成功，辅以阶段价值、时间/规划调用/动作成本。几何进展或后续值通过潜在函数/明确训练目标塑形，防止持续小动作为正回报；奖励结构需要对学习基线共享，单独消融 reward 更改对性能的影响。

### H3｜Successor-conditioned adaptation：证明非短视的多阶段价值而非历史拼接

如果前段的多种合法终态能显著影响相同后段的真实结果，则可采用 terminal-state suitability、successor-conditioned critic、学习型前段参数策略或者带失败处理的 Option 机制。**后续真实物理回报必须进入 Actor 行为选择或梯度**，不是训练一个不参与决策的头。实验须通过：相同初态执行不同前段参数→实际物体/工具状态不同→相同下游实际执行→不同完成概率/代价；全程记录物理连续性与计价。与 STAP/ViSkill/RecoveryChaining 明确比较最近邻方法差异。

如果前段没有可学习的下游差异，**自主取消 H3**，改研究 H1/H2 的接触参数适应，不要为了最初摘要无限刷 B3；必要时将论文问题限定到带实际轨迹/接触执行结果的参数化局部技能。

### H4｜Task adaptation and sensing：从单场景收益转成合同相关可迁移结果

最小要求是训练域外的不同对象尺寸/长宽比、初始角度、入口朝向、质量/摩擦、不同目标、局部障碍或合法站位之一；**至少两个有机制差异的物理任务组**而不只是随机重贴同一个P2名字。可补充另一合法操作如持物近槽预放置、支承面重定向、抓后稳定放置，或已存 ARES-R 只读任务合同映射到合成 Isaac 任务。避免复制受限现场资产。若存在可用视觉/点云状态提取，增加少量非privileged输入或带观测误差的结构化传感测试，不强制第一篇训练大视觉编码器。零样本指冻结policy在未见条件评估，不冒称few-shot梯度更新或真实部署。

## 4. 有自主权也要有可审计的循环节奏

### 循环 R0｜把“真实差异”先展示出来（优先交付）

- 锁定并保护已有 v6/v7 checkpoint、scene SHA、脚本与状态；重现两个清晰事实：旧未见几何 residual>反馈；新 v7 B0≈B3 不能取得方法优势。
- 先拍**真实 Isaac PhysX 基线 vs residual**的4类诚实示例：基线失败/学习成功、共同成功、共同失败、以及 P2 单独代表案例（实际存在才可称救回）。
- 不能直接比较跨不同任务的旧 v6 和新 v7 视频并说是算法胜利；同一视频要两条 policy 在同一场景参数和独立一致冷启动物理执行。
- 真实图、轨迹/统计和一页视觉索引先提交，让导师立即能看到“已做出什么”。
- 快速报告哪种真实失误最有潜力产生新科学增量，并着手下一轮，不停在复盘。

### 循环 R1｜物理控制权/失败修复

- 用最新实际失败及合理新开发任务组，跑配对可执行参数敏感性试验；工具 yaw / 接触位置 / 放弃再接近等由你选。画参数→物体位置/旋转→后缀结果图。
- 如果物理动作没有差异，修复的是**任务和动作自由度**，不是继续增大 PPO 网络。
- 形成一个强规则反馈/规划基线，和对应可学习参数机制；在新开发案例跑真实物理闭环。
- 至少生成第一版同初态并排视频与关键帧，失败案例也包含。

### 循环 R2｜学习方法首次实战

- 基于 R1 确认的可控自由度，训练至少一个真正产生合法新动作的 RL 策略，对比最新强非学习控制、原 residual 与同动作范围的残差基线；适当训练 seed/case 配对，每次迭代给出 learning curve、接触、完成案例映射、失败聚类和预算。
- 若 PPO 不学习，可自主切换 SAC 或学习更低维 Option 参数、调整训练 task/curriculum、利用公开实现辅助；不能反复换随机超参数却不改物理机制。
- 分析模型是否真的选择不同接触/准备行为。不是仅看其最后 NN 参数变化。
- 完成一次消融，把真正有效的模块与简单任务坐标/奖励/额外参数分离。

### 循环 R3｜第二次独立机制改进 + 发表可信度

- 根据 R2 的失败模式自主继续：比如结合下游值、修正 actor action model、残差 SAC、任务几何特征；或如果物理任务不具备训练空间，建立另一种合成工业操作任务/工具动作，保持相同合法约束。
- 争取至少三个有物理与学习实质差异的方案实际训练/评价，**不用局限于固定 B0/B1/B2/B3 编号**，但每次有前因假设和可证伪指标。
- 证据成立时做跨种子至少5 seed、最终独立冻结组、真实预算与配对不确定性；尚未稳定时不得启封最终组反复选算法。
- 同时持续更新“拟科学贡献是什么、证据足不足以投稿、如果没有增量应如何收缩论文”的判断。不能为了达到轮数而做无价值训练。

### 算力与自治边界

允许以现有服务器空闲GPU跑同预算真实训练；先小规模 1/4/16 env 物理校验。初期每机制先短pilot（例如16k–32k新交互/seed），有行为改善后合理扩至64k–128k/seed。**自主科研总预算建议首批额外 0.4–0.8M 环境交互为管理上限**（可按GPU实际空闲与实验效益节约）；每轮必须有真实训练/测试/影片结果，若预算接近上限而问题仍有明确价值，写一个明确的扩充理由与下一步计算需求，不能静默无限使用服务器。不要为了“持续工作”保证单次执行跨平台后台永久运作；在当前可执行会话内尽可能完成多轮并提交真实结果，环境受限就如实报告。对任务动作/控制器可持续改进，不设某个成功率硬门槛来直接阻断全部后续探索。

## 5. 视觉交付是主目标之一：四种能让甲方看懂的对照材料

每次至少保留一组 **actual physics、same-case、two-policy** 的 paired comparison；最终在一个总览中能从定量结果直接打开对应视频/帧。

**V1｜同初态左右对比 MP4（强制）**
- 单个画面左右两栏：Left = tuned baseline（真实充分调试控制器/原 ResidualPPO，标真实名称），Right = Ours（当轮新模型，若未证实则命名 Candidate）。同任务配置、case ID、物体/机器人初态、几何/摩擦，独立冷启动运行；共享统一视觉时间轴，可同步结束后 hold last frame，但不得隐藏超时。
- 物理时钟同时显示；叠加 task goal、当前stage、yaw error、lateral error、距离、独立判定 PASS/FAIL；在画面右下标识版本/配置 hash。两侧物理分别独立执行，不要求复制接触求解器隐藏cache。
- 至少挑选 1 例 baseline fail & method success（**仅若确实存在**），1 例双方成功，1 例双方失败；另外完整列出所有独立评价 case 的配对结果，选择代表案例的规则提前写明（如典型失败类别/初态难度和随机抽样），不能只给 best-of-one 制造总体优势。
- 分辨率尽量≥1920×1080，24/30 fps，约15–60s/个；兼顾文件大小。采用 MP4 H264，附可读 GIF preview。若当前相机依赖有差异，先用现有 Isaac 5.1 Camera/Omniverse渲染接口；不能直接照抄 Isaac Lab 最新版本 VideoRecorderCfg 假设2.3.0拥有完全相同API。

**V2｜接触过程四至六帧对比 filmstrip（强制）**
- t0、第一次有效接触、阶段切换、首次偏转/对齐、结束/超时；各方法并排或逐行。对齐物理事件与实际 tick，不以像素差冒充动力学。
- 最好另有俯视/入口近景、目标误差矢量和真实工件姿态；图题和子图统一论文级字号。
- 源码与PNG+PDF，图注写清独立物理、case ID、仿真任务、成功判据与碰撞/传感边界。

**V3｜轨迹、接触和完成曲线可视化（强制）**
- 同一case真实工件 XY/yaw、速度/角速度、工具实际位置、有效动作、接触代理随时间变化；叠加基线与新策略、阶段界点和真实目标容差。
- 汇总对照曲线：同一方法多种子学习曲线（合理不确定性）、每case成败热图、不同任务条件的成功率、完成时间（包括失败超时并单独呈现）、真实规划/训练成本。
- 论文用 vector PDF/SVG；数值由CSV/JSON自动生成 LaTeX booktabs，不能描绘“目标数字”或伪造漂亮的增长曲线。

**V4｜真实 Isaac 仿真世界高质量展示板（强制）**
- 真实 JAKA 机器人/研究推板/工件/导轨/目标，4×4训练 env 概览与单环境近景；统一相机、光照、真实对象几何；可优化外观、采样、阴影但不得改变动力学、collision USD、策略checkpoint或动作。
- 要清楚标注 synthetic state-based PhysX testbed，不是 ARES-R现场真实料盘、实机或某官方公开 benchmark。
- 不重复把 v7 已有视频仅改标题充作新 B3 结果。

**文件交付结构（在原 v7 成果包之上增量）**：
- research/skill_handoff_lab/deliverables/client_sci_20261027/08_ITERATIVE_RESEARCH/00_START_HERE.md
- 08_ITERATIVE_RESEARCH/METHOD_SELECTION_AND_LOOP_LOG.md
- 08_ITERATIVE_RESEARCH/figures/ + captions/ + tables/ + references/ （只公开合成、无敏感、小尺寸结果）
- 08_ITERATIVE_RESEARCH/comparison_index.csv + VIDEO_PROVENANCE.json（对所有视频建立case→baseline/model→原始轨迹→真实判定→分支/哈希的映射）
- 视频原档/完整120Hz轨迹/模型：非Git研究盘 runs/client_sci_closed_loop_20261027/paired_videos/ 和 runs/.../models/，使用相对路径或已验证绝对路径索引；不要push大原始NPZ、受限资料和不可再许可资产。
- 08_ITERATIVE_RESEARCH/MENTOR_VISUAL_BRIEF_CN.pdf（建议2–4页图文）、THEORY_AND_NOVELTY.md、PER_ITERATION_DECISIONS.md、LATEX_UPDATED_CLAIMS.md。
- 在既有 01_CLIENT_PROPOSAL_CN.md/PDF 与 02_PAPER_SEED/ 的题目/摘要/Method/Experiments/Results/Limitations中增量更新本轮已实测成果；不得删掉旧v7负结果，把未完成的图/表保持TBD。相关正式表/图 provenance 完整、论文 PDF可编译。若最优算法机制转向别的技能族，应更新题目和摘要，不能为保持原标题隐瞒实际问题变化。

**视频与图像真实性检查**：每个源标注 env/seed/case/policy SHA/scenario SHA/action trace SHA/physics ticks/fps/源相机；动作阶段全程只发送真实控制命令，State-write-during-actions=0；录像长度匹配原 episode 原始tick和时间，视频结局和独立验收一致；报告选片规则和总体 paired confusion table，不能用图像渲染改变任务结果。

## 6. 严格但不僵化的实验设计

- Baselines至少：强名义反馈、与新策略**相同动作接口/观测/可行参数范围**的 Residual PPO；按研究问题加入规则/受限穷举或CEM/SAC/监督Ranker。需要说明结构不等价的比较，避免为新RL开放自由度却不给控制基线使用同样合法动作。
- 在同一新开发环境让 baseline 与 candidate 都能真实执行并独立验收，物理参数、初始状态、训练预算一致；对复杂对象的初始几何可行性做单独分类，不能将一切planner求解器失败算几何不可能。
- 将 P1/P2 和后续任务单列，主指标为**完整多阶段 terminal-at-end + 稳定保持 + 姿态/速度**的独立判定；另报目标误差、峰值模拟接触、违规率、规划/执行时间、额外尝试和在线物理交互。
- 对 v6/v7 历史场景，只用于描述性报告与旧checkpoint再演示；新机制的开发验证从独立 scene_group 开始，**不要读取或复用已封最终测试调阈值**。一旦方法稳定，另封正式final组并冻结哈希，再五seed评估一次。
- 主结果按训练 seed 为不确定性单位，scene group 做配对分析；same-case多个policy复评不视为新样本。报数值及paired bootstrap/置信区间（适用时），小样本只能说探索性，不虚报显著性。
- 时间指标不要只展示共同成功条件子集而忽略失败与超时；如果无成功率正结果但效率有稳定收益，重新评估新主张**需在下一新实验前注册**，不能在已读测试结果上事后转指标。
- 需要检查reward action惩罚实际执行增量、gamma/GAE和跨rollout边界、done/truncation，避免之前已修问题复发；下游结果确实进入决策时审计Actor行动/物理结果的因果链。
- 有模型提示的场景/动作条件不得包含训练时可见的未来终态标签，避免拿Privileged仿真质量/摩擦当作已验证真实感知；至少小规模感知误差实验。

## 7. 公开资源与如何使用（优先读/借鉴，不能机械复现或污染旧环境）

以下均为公开研究**启动资源**，具体许可、依赖、权重和可复现性由你再次核验并固定commit SHA。

**方法与代码：**
1. RAPS / Parameterized Action Primitives (NeurIPS 2021)：https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html ; https://github.com/mihdalal/raps 。参考合法动作基元参数接口，不把它重称创新。
2. HACMan++ / Spatially Grounded Primitives (RSS 2024)：https://sgmp-rss2024.github.io/ ; https://github.com/JiangBowen0008/HACManPP 。参考 contact location、primitive how/where/what、Q-map 可解释图；其旧PyTorch/MuJoCo/Gym依赖不应装到Isaac51主环境。
3. STAP / Sequencing Task-Agnostic Policies (ICRA 2023)：https://sites.google.com/stanford.edu/stap/home ; https://github.com/agiachris/STAP 。参考下游技能Q、CEM/参数搜索强基线，不机械复制到没后续差异的任务。
4. Fabrica / Dual-Arm Assembly by Planning+Learning (CoRL 2025)：https://fabrica.csail.mit.edu/ ; https://github.com/yunshengtian/Fabrica 。紧邻研究，task-frame+残差已存在；其learning使用旧IsaacGym，不在本机Isaac51环境整体移植。
5. RecoveryChaining / Local Recovery Policies (IROS 2025)：https://github.com/shivamvats/recovery-chaining 。参考真实失败库、可执行恢复、后缀评估和名义恢复的对照。
6. Reskill/Residual Skill Policies (CoRL 2022, PMLR 2023)：https://github.com/krishanrana/reskill ; https://proceedings.mlr.press/v205/rana23a.html 。参考真正残差/skill priors的实现与公平基线。

**仿真/基准/成像：**
7. NVIDIA Isaac Lab（框架及现有工程支持）：https://github.com/isaac-sim/IsaacLab ; https://research.nvidia.com/publication/2025-09_isaac-lab-gpu-accelerated-simulation-framework-multi-modal-robot-learning 。**版本锁定**：本地已有 Isaac Sim5.1/Isaac Lab2.3.0，不照搬最新版 2026 视频API；必要时先看该版本Camera/Recorder接口。
8. Isaac Lab 官方录制范式：https://isaac-sim.github.io/IsaacLab/main/source/how-to/record_video.html 。其目前页面面向新版API，只作架构参考，不保证老环境同名类可用。
9. ManiSkill3（作为可选第二个公开任务库，不强制装）：https://github.com/mani-skill/ManiSkill ; https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html 。有真实不同操作与现成SAC/PPO基线，可选择隔离新环境做补充；注意SAPIEN仿真结果不能叫Isaac PhysX数据，独立报告性能差异；资产许可证按模型查看。
10. ARES-R技能架构（内部只读）：https://github.com/Trickyboy2100/ARES-R/blob/codex/golden-offline-workbench-20261020/docs/SKILL_LIBRARY_ARCHITECTURE.md 。目标接口 SkillInvocation/SkillPlan/SkillResult/SceneSnapshot，先以 mock/sim adapter证明兼容，不碰真实运行路线。

从以上资料选2–4个**最接近的新方法机制**深入看核心代码，给出“已有人做过什么—我们的实际区别—强基线是什么—需要新训练哪一项”。若有更好公开算法/模型，可加入，但需要出处、许可、版本、复现实验目标与融合成本；不为堆参考数量去下载100GB资产。

## 8. 论文/甲方成果更新与“正结果”判定

**必须同时交付科研进展与甲方可直观理解的产物**，不允许跑了几十万步却只给文字“Gate fail”。

最低产物：
1. 研究方法路线选择（有2候选技术切口和真实物理差异证据）。
2. 至少3轮真实研究闭环记录，每轮必须写下一轮如何根据上一轮证据调整。
3. 至少一个实际训练且物理改变可执行动作的RL新候选，和充分调试非学习/普通Residual基线同预算公平比较；若候选非原创则明确写是强基线/过渡方法。
4. 可观看的实际对照 MP4 至少3组，代表性 filmstrip PNG/PDF ≥2组，场景高质量截图两视角，以及逐case/逐时间真实轨迹图。
5. 真实数据驱动的 LaTeX Booktabs表，绘制种子间曲线、机制消融，英文论文Method/Results/Limitation 更新，中文导师/甲方图文版同步更新。
6. 跨任务或至少跨对象/几何域外验证的诚实状态报告，若未完成标 not tested。
7. 非Git本地模型包/源码快照/scene/物理轨迹/视频 + SHA 索引（不得push受限数据/大文件）。
8. 结束时清理无用checkpoint/重复文件：保留正式种子、关键中间、负结果、评估与哈希唯一源；按完整性检查归档冗余副本、实际释放空间并记录；不得删除唯一正式证据。

**鼓励的有说服力正方法**：在物理合理且非饱和任务中，新方法有实际更好的整链完成集合或可靠更低成本，并且机制消融证明效果来自学习的接触/准备参数调整而不是修复旧控制器、任务设得更容易或使用更多计算。论文完成态最好达到5独立seed、独立final scenes、至少两类有差异的操作、符合合同的技能/感知系统图及一项非privileged状态评估。

**若无正增量**：也交可视化比较、真实失败轨迹、已经排除的原因、下一候选、额外物理实验，而不是制造正柱状图或将既有Residual PPO结果改写成新算法发明。要保持积极探索、连续迭代，但明确哪些科学假设被数据否定。继续投入有据可依的改动，不因某个机械化阈值卡住整个研究，也不无限刷同一种失败机制。

## 9. 最终向用户/导师的一屏式简报

必须先给直接可打开的四个入口：
- 甲方图文研究方案 PDF / 英文论文 PDF 与tex源；
- **同初态真实 PhysX 基线vs当前最佳候选 MP4/filmstrip**（附case ID与批量成功表）；
- 真实机制实验/成功率/轨迹/消融LaTeX结果表与矢量图；
- 新研究候选checkpoint、恢复脚本、源码/scene SHA索引。

然后用精炼中文回答：
① 与合同、ARES-R任务链和论文问题的关系；
② 我们已经证实了什么新RL方法，什么尚未证实；
③ 最佳候选 vs 强反馈 vs 普通ResidualRL 的匹配案例结果、每seed、成功/时间/成本和失败类型；
④ 对比视频中能**实际看见**什么物体/机械臂行为差异，其物理测量是否一致；
⑤ 选题相对RAPS/STAP/HACMan++/Fabrica等的独立研究贡献以及下一轮仍缺什么；
⑥ 新交互数、计算成本、本地真实模型/视频位置、是否已公开推送；
⑦ 本轮释放了多少磁盘空间。

**最终立场：尽最大努力通过连续真实闭环形成能够投稿的正方法与可视证据，而不是声称一定胜利。凡是要展示“我们的方法让机器人做得更好”，必须同时展示同任务的真实物理对照、可审计的全体案例统计与算法因果消融。**
