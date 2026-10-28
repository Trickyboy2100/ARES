# 2026-10-28｜一篇 SCI 论文导向的 ARES 下一阶段研究决定与 Codex 执行单

> **唯一主要目标：**以现有 ARES-R 的多模态感知→Skill Contract→Skill Runtime→cuRobo/机器人底层控制架构为工业动机，研究 **Task-Conditioned Executable Skill-Parameter Adaptation for Reliable Multi-Stage Robotic Manipulation**（面向多阶段机械臂操作的任务条件化可执行技能参数自适应强化学习）。不能再把某个接触推板/PPO技巧/漂亮视频自发升级成论文主题。本文要解释并解决**当前技能局部执行似乎合格、却因实际物理终态不利而导致后续技能失败**的具体问题。
>
> **工作分支：** `codex/ares-sci-single-paper-skill-handoff-20261028`，从已验证 `7f12d69b4866e8f7282bbc0a189271c756b32d00` 建立。研究前检查本地工作区、并发 GPU、源码引用和未提交修改；独立 worktree，不改 ARES-R 生产分支、真实机械臂或受限现场数据。
>
> **禁止宣称：**已发明后续价值/技能参数化/任务条件化/Residual RL，已完成新方法增益、具身视觉策略、JAKA 实机转移或整个长程智能系统。本次面向甲方的图文资料、LaTeX主稿必须与真实结果同步、正负结果完整。

## 1. 对 7f12d69 的实证判断：哪些证据真正可用

本轮 **903,584 次物理控制交互**，其中 **614,400 次PPO在线训练（3 seeds ×2动作接口，每模型102400）**。有6个正式学习模型、后缀真实配对采集、独立CEM、真实轨迹/视频/LaTeX与SHA，都是可信工程研究资产；非全部都属于“强化学习训练”。

**StackCube-v1 最新64开发例**：名义60；Residual PPO（31/37/43）63/62/64；参数PPO62/63/60；pairwise-abstention 60；固定高度与等数据成本 CEM 62。参数模型对Residual平均约−2.08pp；差分选择器没有救回名义失败。不能挑seed43的64/64。Residual逐步修改XYZ+夹爪，parameter PPO主要放置XYZ参数；当前性能比较**不是同动作权限和同频率的算法消融**。单次放置三选一的paired模型是contextual bandit，不是SMDP/Option RL。64例之中名义60已饱和，平均增量小，继续堆历史/gate/训练不利于论文。

**StackPyramid-v1 实际物理机会**：原生成功谓词、250步、不改物理；单独16个设计案例，官方规划与读回版本名义12/16，第一方块位姿−12mm同12/16（1救回1退化），+12mm7/16（无救回、5退化）。执行前提前松爪之后名义11/16，−/+版本均9/16，不单独优于原12/16。**从`PYRAMID_PAIRED_CASES.csv`+`PYRAMID_RELEASE_CASES.csv`逐案例联合计算，六条真实执行策略的已见成功集合并集是15/16**，有三个原12/16不成功的设计案例在某一变体成功，唯一全部失败种子122002。**此15/16为事后已知结果的可见机会集，不是部署策略成功率、统计置信保证或同动作接口公平消融。** 提前松爪改变执行时序和后续物理状态，分组时必须明确控制合同。16个设计案例已被研究者看过，后续仅可做物理诊断，不得当未见测试。

**PegInsertionSide-v1**：原生100步，已检查8个设计案例，强控制约1/8；规划步骤108–161超出原生期限的不能当官方成功。无可信强baseline前，暂不把Peg作为下一轮主方法验证任务。**StackCube继续仅做已成熟基线/退化对照，Pyramid作为唯一主方法研究载体**；一旦主方法有稳健实际增益，再以 Peg、OpenCabinetDrawer、TwoRobotStackCube 中最可行的一个承担第二物理机制或双臂展示，不同时开多个大环境。

**重要语义审查**：当前Pyramid官方脚本在靠近下一次抓取过程中才松开第一个方块。尽管提前松爪消融真实执行过，但**不能把脚本 waypoint 当成“已完成并验收的前段技能终态”**。必须从释放夹爪、支撑方块稳定、真实机器人/工件状态和新目标可达性定义 SkillResult 及唯一真实 SkillBoundary，再研究下一技能。不能把这项工程语义修正作为学习算法的增益。

## 2. 一篇论文的稳定主张与待证明的假设

**选题一句话：**机器人有可复用的名义操作技能，但即使当前步骤满足局部几何目标，执行得到的实际支撑关系、物体姿态、夹持/释放状态、机器人状态也会影响下一个技能是否能够完成；研究RL怎样在任务/后续技能约束下选择**合法、实际可执行的前段参数**，并利用真实执行后的物理状态决定下一段参数，从而提高完整长程任务成功及零样本任务条件适应。

**选用单一论文方法工作名（可随实验证据修订）：** `Contract-Conditioned Skill-Boundary RL`（合同条件化的技能边界RL）。此为方法架构名，不是新颖性结论。论文理论对象是物理执行的SMDP/option，不是单步分类、纯trajectory回放或离线CEM贴RL标签。

物理接口：
`o_k = (proprioception, object/goal relative geometry, current/next skill preconditions, stage, remaining budget, observed physical postcondition)`。
`a_k = (mode_k, xi_k)`，其中 `xi_k` 是小维、可执行、受当前/下个技能合同约束的参数，`mode`只允许已验证的真实释放/重定位/保持动作；不选择无法执行的虚假值。
`s_{k+1} = ExecutePhysically(skill_k, a_k, s_k)` 并经独立`SkillResult`与物理状态验收。
至少**两个独立有用的决策边界**：第一方块放置/实际松爪之前选参数；观察第一方块真正稳定并释放后，决定第二方块抓/放/对位参数。如果只有单次参数选择，必须称 Bandit，不宣称两阶段价值传播。

有限时域技能级 actor-critic 可用标准 PPO/SAC/离散Option价值学习；计算实际技能持续时间 `tau_k` 的回报与 `gamma^{tau_k}` 或显式说明`gamma=1`有时间成本。以独立**全部任务成功**为主目标，辅以时间、失败、额外规划/执行调用；对所有同接口基线共享合法条件与reward。可试从真正运行的后缀样本估计 `Q(s,xi; C_next)`；若使用相对名义的优势 `Q(s,xi)-Q(s,xi_nom)` 与保守修正，就要和普通绝对Q、单次监督ranker/固定规则/等预算CEM比较，并证明差异真正进入Actor或参数选择。后续价值思想已被STAP/ViSkill/Foresight Residual RL研究，不能把相减公式或预测头起名包装创新。

**必须明确验证的三个科学事实：**
A. 前段技能存在不止一种合法、当前技能本身成功、但后续表现不同的实际稳定终态；不能让只在前段失败的样本冒充“handoff问题”。
B. 在新工件/目标条件下，利用当前可获得的场景几何和真实执行结果，学习型参数策略能相对名义规则、普通ResidualRL、同接口PPO/SAC与强规划/监督ranker改善实际任务结果或等成功率下减少可量化成本。
C. 去掉next-contract/真实前段效果/自适应参数/后续价值后，至少一个关键机制的真实性能独立下降；否则保留有用工程方法但不宣称新算法。

特别警惕：当前Actor直接看到完整模拟状态时，历史执行效果可能已被下一时刻状态完全包含，不一定增加可预测信息。若研究不可观察摩擦/夹持等动态属性，应提供合理的部分可观测输入/噪声状态，让动作—效果历史真正发挥辨识作用；不能给Actor未来成功或测试case标签。对多模态合同的支持最终至少需要视觉/点云提取的结构化位姿或受控的感知误差实验，不能把纯privileged state称视觉闭环。

## 3. Codex首先做的真实物理机会研究：下一轮不能只出文献和Gate

### E0｜统一可解释的完整 Skill 合同（先完成真实物理试验）

从已pin ManiSkill StackPyramid 官方强planner/真实Panda机器人及原生250步实验构造**所有方法共用**的组合skill interface：`PickA→PlaceA/Release/Verify→PickC→PlaceC/Release/Verify`。子任务是否结束由工件位置/速度、真实夹爪松开、接触/支撑和机器人状态判定；合法移交后不重置/瞬移任何物体、不把下一步已有抓取错归前段完成。保留上游官方planner及原生成功规则；不得用最初0/16的弱手写Pyramid反馈当唯一强基线。

先对训练域随机且物理可行的多批任务条件，比较 **原名义、横向前段目标参数、提前/延迟实际释放、第一阶段结束后下阶段合理的对位修正**。可根据物理可行性自由调整参数范围，不拘泥旧−12/0/+12mm，所有控制动作经真实SAPIEN执行。必须把某参数请求→实际关节/抓爪动作→物体稳定终态→下一技能实际成功这一链打通。 
至少64–128个全新训练设计场景的有界cold-reset配对机会诊断（根据GPU耗时裁量），记录每种方案的：当前阶段合法完成 / 后续真正接续成功 / 完整任务成功 / 末态有效支撑几何margin / 物理时间与额外尝试。按照不同reset group分割训练/开发和未见测试；不能因为只有原名义失败样本才选择评估场景，不过滤对候选不利的案例。
计算 **实际候选成功并集（hindsight oracle）与任何一种固定基线的差**，但仅作为“决策空间是否有机会”的上界探针，不用oracle标签在测试时选择动作。将原16已观察样本作为诊断而非正式泛化。
如果参数改变只有坏结果，修复合法控制自由度/再试其他确实会影响支撑状态的动作（例如释放timing或重对位），不要直接增加神经网络。若某几何公式/固定规则同样有效，保留它作为强基线。

### E1｜真正的两个Skill边界RL方法原型

在共同技能executor上构造实际发生物理状态转移的有限时域SMDP；第一段学真实放置/松爪参数，第二段利用**真实已观测第一段结果**学习放置C/对位参数。允许先用低维离散/连续混合actor来缩短试错成本，具体实现允许选择PPO、SAC或已验证Option策略，优先保留可重复的强planner reference而非从0重新学习抓取关节。

最小公平对照（都在**同样允许的技能参数/释放动作集合、相同strong nominal executor、相同原生250step合同**）：
- Planner/强固定规则 + 手工几何间隙与正确release；
- 参数固定调优 / 有限搜索或同预算CEM；
- 相同参数边界、同观测与预算的普通Parameter PPO/SAC（**必须补，上一轮缺失**）；
- 单次绝对结果Q/supervised ranker 和上轮paired difference（检验“其实不需要RL”）；
- 本轮两边界的task+next-contract/outcome-conditioned RL；
- ablation：去next contract、去真实前段后验状态/作用（须控制observable信息）、去第二段重决策、去后续Q或相对优势/预算风险模块。

学术比较要按 `interaction train + critic数据预采集 + planner solve + time`实际总价计账。若选择SAC，公开DAWN(ICML2026)中base-policy transitions和critic LayerNorm的设计可作为强现有baseline/样本效率启发，**不算原创**，且其官方ManiSkill为PegInsertionSide-v2等，与本机v1不同，不直接搬分数。新候选若只是RAPS/STAP/Foresight的一种常规拼接，须从方法定义和所解决场景上再收缩主张。

### E2｜连续研究闭环与第二实验任务

不直接对旧Stack64饱和开发集调参数，也不因为Peg1/8就立刻扩大PegPPO。建议：
- **主研究任务：StackPyramid-v1** 原生250step，可在合理条件下形成约12/16基线和跨技能参数机会；新增主验证应有≥3训练seed、独立开发场景与任务参数外推，不从16已看设计案例选择成功视频当最终主表。
- **辅助：StackCube-v1**，用于确认方法不会显著破坏已经可成功的名义技能，不把64/64当成算法优化主要任务。
- **第二有差异任务**：先调研/小规模smoke `TwoRobotStackCube-v1`（双臂支撑体在下一臂堆叠前放好，可联系ARES-R双臂SkillContract，官方100步且有原生dense reward）、或 `OpenCabinetDrawer-v1`（关节物体与把手接触）、Peg（仅在有正常可解强基线后）。只选择一个最符合科学主张/工程成本的，不必安装所有bench。
- 允许Codex独立做2–3轮真实闭环，从E0的物理可学习机会逐步打磨两个边界RL，先3 seeds训练pilot，机制有力再5seed和封存测试。初期建议约0.4–0.8M新物理交互上限（可根据单env planner费用/并行能力合理细化）；临近预算前完成已开始的科学问题，而不只写gate报告；超额需要实证理由。每轮都需有真实训练/真实物理配对轨迹及可观看对照，而不只是审计与模型名。

Pyramid官方planner在单env CPU优化路径可能是速度瓶颈：不能静默认可进行64env高速在线RL。先报告实际每episode规划/physics墙钟，把强参考技能实现为可运行的受控执行器（如planner生成轨迹后真实servo跟踪或缓存可复用**候选计划**但真实对象每次都必须运行物理且起始状态严格匹配）；缓存离线plan成本与训练采集全部计价，不得把虚拟技能结果伪装成PhysX交互。若无法经济地获得真实两边界训练，则先缩为真实参数SMDP小任务组，不改变原任务成功判据或用旧script0/16弱baseline制造优势。

## 4. 紧贴 SCI 投稿的统计与图文证据

**优先实验表**：阶段0验证合格率、阶段1完成后的支撑margin/姿态/对象速度、阶段2条件完成率、全任务成功率、救回数/退化数、真实执行/规划代价、训练物理交互数，每个方法独立seed和场景组。务必报告 `P(downstream success | predecessor physically verified)` 的变化，避免不同前段失败数误导后续成功率。

**主图**：Fig1任务+ARES-R感知/Skill Contract/策略层到机器人执行（把未接入与已实现做显著标识）；Fig2唯一新方法结构含2边界决策与真实SkillResult后验；Fig3前段相同初态不同参数→物理终态→同后缀效果（真正物理视频filmstrip）；Fig4新的2边界RL学习曲线；Fig5 独立数据成功率/失败归因/配对提升与不确定性；Fig6 no-next/no-effect/no-boundary/no-advantage消融；Appendix 与常规CEM、强rule控制与实验成本。

**统计**：任务抽样场景与训练seed分别评价，配对数据只对相同case做比较，标出相同case在多策略/seed中复用，不按全部episode当完全独立；不进行test挑模型；所有已看16 Pyramid设计案例与Stack新64都不称密封测试。固定最终方法后只开一次独立sealed test；成功率之外包括配对净收益及不确定区间。对外只写独立验证通过的数字，不能把case oracle上界/位姿请求或某张救回视频写成新策略得分。

**一篇论文的目标投稿工作标题**：
中文：《面向多阶段机器人操作的任务条件化技能边界参数强化学习》
英文：*Task-Conditioned Skill-Boundary Parameter Reinforcement Learning for Reliable Multi-Stage Manipulation*
题目可据完整结果微调，但不能再次变成“发明全部层级VLA”或“单个手工Pyramid调参”。如果新方法增益无法证实，允许论文收缩为真实物理参数与技能接口的系统性研究，但须诚实评估是否够SCI方法论文，别提前把不成立的Algorithm当贡献。

**推进与最后交付：**
在研究成果目录 `research/skill_handoff_lab/deliverables/client_sci_20261027/11_PAPER_FIRST_20261028/` 输出 `00_PAPER_THESIS.md`、`PHYSICAL_SKILL_BOUNDARY_SPEC.md`、`OPPORTUNITY_STUDY.csv`、`METHOD_AND_BASELINES.md`、`ONLINE_RL_RESULTS.csv`、`CASEWISE_EFFECTS.csv`、`VIDEOS/index.html`（视频大文件留非Git）、`FIGURES/*.pdf/*.png`、`LATEX/*.tex`、`MENTOR_DECISION_CN.pdf`和源码/模型/场景SHA索引。同步维护现有01客户中文方案和02英文论文（不能把新设想写成已测算法）。少量技术图是论文证据，不因为想展示多视频就生成不必要模型/训练。`STORAGE_CLEANUP_REPORT.md`实际清理无引用/重复大ckpt，正式模型/物理负结果/原始轨迹/受限现场资料不删；不改Git历史或生产ARES-R。

**最终应向用户解释**：这篇论文只解决什么具体问题；两个RL决策与单次参数分类有什么区别；第一阶段真的完整结束了吗；新的可执行参数在哪些真实物理条件下改善了后续可行性；强基线为什么没有同样解决；训练了多少、每种子结果如何、是否跨场景/任务成立；能直接打开哪些同案例视频/论文表；离可投稿还缺什么。

**执行原则：**允许Codex自主尝试有实际依据的算法与物理参数，但始终服务上述一篇论文。强planner、原生终态、真实物理与任务分布不可为制造正结果而削弱。不要无界审计/文献搜索、不要持续盲训Stack64、不要把16设计案例的15/16 oracle并集写成RL胜利。优先跑出新训练集上的实际可控handoff与两个边界RL原型，然后再决定方法方向。
