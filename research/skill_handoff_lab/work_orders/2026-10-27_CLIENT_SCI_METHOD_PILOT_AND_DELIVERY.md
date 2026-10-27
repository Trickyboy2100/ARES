# 2026-10-27｜ARES 甲方 SCI：正方法探索、真实初步实验与图文/LaTeX 交付工作单

> **任务性质：直接执行的研究工作单，不是审计计划。**
> 来源基线：ARES 研究分支 codex/ares-skill-handoff-p2-1-20261013，已核对 SHA 79448a47ffda5c8c1887f53e33a5dbb90d3eb490。ARES-R 仅作只读系统参考；其 2026-10 现场材料位于受控非 Git 目录。**本工作单所在分支是从上述 ARES 研究 SHA 创建的独立工作分支；先核对本机 HEAD/未提交变更与可用算力，不覆盖其他人的 worktree。**
> 合同主题：《多模态具身感知与基于强化学习的仿人机械臂控制策略研究》。目标为一篇可支撑甲方汇报、后续可扩展为真实 SCI 稿件的**强化学习方法类**论文，以及尽快可交付的初步图文成果。合同未要求特定分区，也未要求以某次现场 IK 故障作为科研核心。

## 0. 不可混淆的证据边界

**目前已验证，不得丢弃也不得夸大：**
- ARES/Isaac Sim 5.1 + Isaac Lab 2.3.0，真实 PhysX 16 env 在线 RL 管线；v6 新增 901,120 次交互、16 个训练模型（**含诊断/视频**，非 16 个正式种子）。已有 3 训练种子正式物理评价。
- 现有自建 P1 两阶段推送、P2 深槽接触操作；56 维仿真状态 Actor、PPO/ResidualPPO、阻尼控制、准确 post-step 终止、过程独立验收和受限动作接口。
- v6b Test-Geometry 每方法固定 32 例：阻尼反馈 8/32；普通残差 PPO seeds 31/41/51 为 13/18/23；新增学习推进/保持为 15/15/18，**没有证明它优于普通残差**。P2 任务单独统计普通残差为 6/6/7（每 seed 16），尚是主要不足。
- ID、Dynamics 测试残差与反馈基本接近；旧 WideCabinet 64/64 饱和；接续等待、混合动作有效似然均未证明有实质性独立提升。P1 增益不能冒充 P2 窄槽优势。
- ARES-R 已有 SkillRegistry、SkillRuntime、Task/Scheme、WorldModel/SceneSnapshot、观察记录、cuRobo/TransferChain、真实 JAKA/底盘/夹爪服务等架构和工程记录；部分链已离线 replay，不等于端到端智能实机自主成功。
- 真实 10 月点云/运行配置/机器人读数属于受控现场材料。**不能加入 public ARES/ARES-R Git、不能制作公开截图或在公开文稿泄漏场地几何。**

**目前未证明：**合同条件化提高表现、执行结果价值改善策略、跨任务快速适应、零样本实机迁移、视觉/点云 Actor 完整闭环、全链技能 RL 优于同预算规划搜索。论文摘要/图表必须区分 VERIFIED、PROTOTYPE、PROPOSED、NOT_TESTED。

权威入口（按具体运行捕获的 task_source、helper source SHA 复现，不以 HEAD 替代旧执行源码）：
- research/skill_handoff_lab/reports/CONTACT_HANDOFF_RESULTS_V6_20261016.md
- research/skill_handoff_lab/reports/CONTACT_HANDOFF_METHOD_V1.md
- research/skill_handoff_lab/reports/P2_1_SCIENCE_DECISION.md
- research/skill_handoff_lab/results/contact_handoff_20261016_r1/README.md
- research/skill_handoff_lab/tools/contact_composition_v6.py、run_contact_v6.py
- research/skill_handoff_lab/cash/{env.py,gym_env.py,model.py,option_backend.py}
- ARES-R docs/SKILL_LIBRARY_ARCHITECTURE.md、src/ares_r/skills/{contracts.py,registry.py,runtime.py}、task_runtime.py；真实技能系统只读。
- 不重新进行广泛无界 repo/设备/现场审计，也不直接扩大旧饱和任务 PPO。

## 1. 论文定位与原创性约束

**首选问题：**机器人面对变化的工件/入口/任务条件，如何在成熟名义技能之上，用 RL 自主调整合法参数，结合实际执行结果与后续技能要求，改善**完整多阶段操作**的可靠性与未见条件下的适应，同时尽量减少额外规划/执行预算。

投稿中文**工作**题目：
《面向多阶段机械臂操作的任务条件化执行结果感知参数化技能强化学习方法》
英文**工作**题目：
“Task-Conditioned Outcome-Aware Parameterized Skill Reinforcement Learning for Multi-Stage Robotic Manipulation”
可以通过证据缩短题目；**不能为了契合题目宣称已有跨技能学习和多模态 Actor**。

建议区分两层：
- **已有系统背景**：多模态观测→结构化目标/SceneSnapshot→Skill Contract/Scheme→已有 cuRobo/反馈/学习型局部技能→执行结果与校验；LLM/知识图谱/VLA 可展示预留接口，但不假称整链训练。
- **本论文算法对象**：任务/几何条件化的合法技能参数动作，以及实际物理执行结果对后续参数或动作选择的影响。重点是在**同样名义控制、相同观测/动作许可和训练预算**下取得独立可消融增益。

**不能冒充创新**：RAPS 已做 RL 参数化动作；STAP/ViSkill 已做技能链后续价值；Residual Skill Policies 已做技能空间残差；HACMan++ 已做作用点+动作基元；GRN 已做预测规划/IK 可行性；RecoveryChaining 已做恢复与名义策略组合；Fabrica 已做规划+几何等变残差技能；TASP(ICRA2026)已做黑盒异构技能层次拼接。**仅“任务条件拼接”“局部坐标系旋转”“额外预测头”“标准 PPO/SAC + Skill Library”“执行资格掩码”均非单独的新颖算法。**

精读这些最贴近机制的论文方法/实验/可公开关键代码（不是再罗列几十篇），输出每篇 4 行重叠与不同点：
1. RAPS, NeurIPS 2021: https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html
2. STAP, ICRA 2023: https://sites.google.com/stanford.edu/stap/home
3. Residual Skill Policies, CoRL 2023: https://proceedings.mlr.press/v205/rana23a.html
4. HACMan++, RSS 2024: https://sgmp-rss2024.github.io/
5. GRN, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/561dd71d04695c85e1c0f6a0fbe0bd85-Abstract-Conference.html
6. Fabrica, CoRL 2025: https://proceedings.mlr.press/v305/tian25a.html
7. RecoveryChaining, IROS 2025: https://merl.com/publications/TR2025-152
8. TASP, ICRA 2026: https://ieeexplore.ieee.org/abstract/document/11697062
文献比对完成后写 NOVELTY_AND_RISK.md：明确可否发表为应用方法及真实新机制缺口；不许称“首次提出”已有概念。

## 2. 有界的“高概率形成正结果”方法探索

**主攻基础 A（最先落地）：任务/几何条件化 Residual Skill RL。**
现有 P2 真实旋转/导向偏移/卡滞比“等待交接”更有学习空间。先把目标、速度、工件姿态、允许动作与入口方向归一到任务局部坐标系，修正名义控制在 BODY-X/Y 上的硬编码对齐；保证强反馈基线一起修正。连续策略可选 Residual PPO（首选复用）或有界 Residual SAC（当样本效率确有问题再比），动作先为合法的局部前向/横向参数，经过项目硬限制映射到 JAKA 关节驱动并在 PhysX 真实步进。可提出物理合理的接触位置/再接近参数，但必须先用同起点真实 PhysX 可控性对照证明该自由度确实能改变后续结果。仅换局部坐标，虽可能提升几何泛化，却必须记作**工程/归纳偏置基线**，不单独声称新方法创新。

**科研增量 B（优先检验、但不硬凑）：执行结果感知的跨阶段参数更新。**
在真实两阶段任务中，让当前技能的可调整参数影响后续技能可执行的物体姿态、侧向位置、角速度/接触模式等；使用真实**后缀执行**与独立验收获得训练回报/条件价值。设计一个真正进入 Actor 决策/价值更新的后续结果机制（如 successor-conditioned action value / qualified outcome-conditioned residual adaptation），并与普通 Residual RL、统一 task-frame、同容量预测头但不接入策略、监督成功率排名、等预算固定搜索进行公平对照。**禁止未使用的 auxiliary loss 冒充贡献。** 当前 R0 的等待/保持负结果说明“降速等待一定好”不成立；探索空间应侧重物体姿态、接触位置及合法修正动作，不重复无用等待。
若 B 未显示清晰的后续因果作用与独立增益，不无限堆头；第一篇可降为“任务条件化、几何/合同约束的共享残差操作技能”应用方法论文，但必须多任务、非饱和、强基线与清晰的真实独立增量，否则只交研究方案，不谎称已具备投稿结果。

**最小数学接口（可按结果修订）**：
- 当前时刻观测 o 包含机器人状态、工件相对目标的几何/速度/姿态、阶段、任务及后续合同 C_next、历史执行结果 h；训练若使用 simulator privileged 属性，必须分清 Actor/Critic/诊断，不宣称已部署视觉。
- 合法参数 xi = Project_C(xi_nom + f_theta(o,C,C_next,h))，物理投影同时作用于名义及学习分量，不删除硬约束。
- 奖励以独立验证整链任务完成为主，次为执行成本/真实规划调用/时间；辅助项优先用 potential-difference shaping：r= r_task + gamma Phi(s_next)-Phi(s)，而非无限的接近目标正奖励。单独消融奖励重设收益；统一新旧学习基线的奖励与动作可执行性。不同 Option 时长正确按 SMDP 或明确简化有限时域。
- 如果只是一次候选评分，诚实标记 contextual bandit；必须和监督 ranker / 等预算搜索比较，不能冒充长程 RL。

**算法选择规则**：第一版标准 Residual PPO + 强反馈控制，固定同等训练预算。仅当机制/探索瓶颈实测支持，加入 SAC 或混合 Option actor，不允许同时开发世界模型、MPC、复杂 KG 与 VLA 作为本次论文前置条件。

## 3. 真实任务、非饱和性和公平实验

先使用**已有自建物理任务**：
- T1：P1 两阶段物体推送；作为基础研究/稳定性对照，不让容易的 P1 掩盖主要科学问题。
- T2：P2 深槽导向、入口方向/槽宽/初始 yaw/摩擦与质量变化；**P2 单独成功率为主指标**，阶段真实动态连续，不瞬移工件。
- T3 仅当 T2 机制成立，再低成本建立第二种明确不同的局部操作（如不同导向几何、入口方向/物体长宽比造成不同可用接触与姿态修正），或使用真实来源、明确限制的 JAKA 预放置局部仿真；不是给 P2 换名称或贴图。Real-aligned 现场方案是独立回归，非新训练标签。

固定 Train/Validation/Test-ID/Test-Geometry/Test-Dynamics 的**scene group**；旧 v6b 的 Test-* 已被阅读，**不得重新用于选网络/阈值/方法**。在新开发集比较后，预先封新的最终测试集合与哈希。真实物理上不可行几何单列 UNKNOWN / INFEASIBLE，不把规划器某次失败判为绝对不可行。

难度设计按真实几何余量、目标扰动、控制误差/摩擦等物理参数预声明。**不人为制造“RL胜出”的测试分布，不通过禁用基线合法动作制造差距**。若固定反馈或完整方法出现 64/64，就保留结果，同时增加独立、物理有依据的变化任务族，使主评价有区分度；不为卡某个目标成功率挑测试案例。不允许强行目标“低于基线无用所以删结果”。预设论文目标是**相对于充分调试的普通残差RL和规则/搜索基线的可重复正增量**，而不是预先保证数值。

先以 1env 物理校验 → 4env对比 → 16env训练，训练真实在线物理交互而非缓存计划播放；有界 pilot 例如 16,384–32,768 环境交互/方法/seed，先三配对训练seed；出现有技术意义的P2增益后延至 65,536 或适当预算、5种子、64–128 frozen case 每组。报告总训练交互、网络容量、优化更新、规划/观测资源与墙钟。旧 901,120 交互计入历史，不能再写作本轮新交互。

**最低对照：**
B0 合法阻尼/任务局部几何反馈（充分调试）；
B1 现有 BODY-XY Residual PPO（同一新物理基线、观测与训练预算重训）；
B2 统一 task-frame Residual PPO（结构化工程基线）；
B3 B2 + 执行结果/后续合同机制（拟正方法）；
B4 在同样实际合法参数范围内的有限固定搜索或监督 scoring 基线（如果决策接口允许）。
P2 部分与 P1 部分单独汇总；不得仅报宏平均。对于 B3，增加去 outcome、去 next-contract、同容量假条件、无效预测头（不入 logits）等最小能分离机制的消融。未执行的动作不能记罚也不能记作正动作影响。

**科研里程碑门槛（用于减少无效计算，非事后筛选结果）**：
G1 物理可行性：合法动作经机器人关节真正改变工件状态，无重写/瞬移，独立后缀验收完整；
G2 Baseline：修复后任务局部反馈和 PPO 公平对照可复现；若无可学多决策空间，直接报告。
G3 创新 pilot：B3 相比 B2 在 P2 新验证组的**配对**成功集合/完成时间/几何泛化有一致方向且消融确认关键输入与动作变化，优先扩正式多种子。若优势只在单个 seed 或 P1 出现，不能推广为方法优势。
G4 最终确认：独立未见场景、多训练种子，统计以 seed 和 scene 为分析层级，报告 per-seed 成功率、配对差/区间、失败类型和训练代价；不把同一64例多次评价当成独立 N 倍扩充。
失败时写原因、最优保留成果和备选机制；不要假设一定获得新正结果。

## 4. 甲方初步交付物：真实、学术、可编辑

本轮**必须交付导师/甲方可阅读的完整初稿，不许等全部方法实验完成才写**。在研究根目录下新建：
research/skill_handoff_lab/deliverables/client_sci_20261027/

固定结构（可增加，不改变含义）：

1. **00_README_EVIDENCE.md**：一页中文摘要、实验状态、所有图表来源、无现场数据声明、运行方法与文件导航；标注 EXACT / REPRODUCED / PROPOSED。
2. **01_CLIENT_PROPOSAL_CN.pdf + editable source**：建议 6–10 页高质量图文方案（优先 Markdown+LaTeX 可维护排版，必要时 DOCX），至少：合同关联、真实研究问题、拟定中英论文标题、CN/EN 研究摘要（明确尚未验证的增益）、相关方法比较、方法图、已有/拟建仿真、现有正负结果、创新假设与机制、实验设计、拟完成目标、阶段交付与局限。**不能把 v6 的 P2 小增益说成最终算法胜利。**
3. **02_PAPER_SEED/**：可编译的 IEEEtran 或普通通用两栏 LaTeX SCI 工作稿（不要伪装已被某刊录用）：
   main.tex、references.bib、sections/{abstract,introduction,related,problem,method,experiments,results,limitations}.tex、figures/*.pdf、tables/*.tex，另有 latexmk/tectonic 脚本。英文标题、正文与图注使用学术稿件表达，明确 proposed/verified，不抄写他人摘要段落。无数据处“Not yet tested / TBD”，不填虚假结果。
4. **03_FIGURES/**：
   Fig1 全系统分层矢量图（任务/多模态感知 → ARES-R Skill Contract/Task Runtime → 论文 RL 参数优化 → cuRobo/接触 skill/物理执行 → outcome/evidence），标明已实现与拟实现；PDF+SVG+源代码；
   Fig2 核心方法计算图（o、C、名义动作、合法参数、物理反馈、真正入 Actor 的后缀结果）；PDF+SVG+源；
   Fig3 **真正 Isaac Sim 5.1/Isaac Lab** 的一张精美场景展示板：高分辨率实际 PhysX 画面（建议 single task success/fail 和 4×4 16env 并行至少两张），统一相机透视/光照/标签/比例、原始未调场景下的物理身份，渲染美化只能修改外观，不能修改碰撞/质量/初态/结果；标明 SYNTHETIC 自建研究任务、无真实料盘抓取、非现场或官方 benchmark；
   Fig4 真实在线 RL 学习曲线（mean+seed分布，给出时间/交互横轴与 train/eval 区分）；
   Fig5 **真实**旧 v6b 固定冻结测试图（至少 Geometry, ID, Dynamics, P1/P2 split，三种子，非学习基线），历史结果引用原脚本/哈希；
   Fig6 新 pilot 真实 B1/B2/B3 消融 / 机制图（**有新结果才绘制，不画占位高柱**），另给中间可视轨迹与角速度/物体接触偏差图。
   所有图优先 PDF/SVG 矢量字体正常，截图另提供 PNG/JPG；配 source/*.py、data/*.csv、caption_en.md 与 provenance/FIGURE_PROVENANCE.json。禁止使用网上不相关机器人图代替本实验真实截图。
5. **04_TABLES/**：LaTeX Booktabs 表：Tab1 环境/任务/已验证资产；Tab2 真实 v6b 结果 + P1/P2，不混原始与独立；Tab3 新 pilot 基线和方法（没有的格填“TBD”）；Tab4 机制消融/多种子/预算（未做的填“Not tested”）；表数据由 CSV/JSON 生成，不允许手填想要的均值或正数。
6. **05_METHOD_AND_EXPERIMENT/**：预注册小型 pilot 协议（假设、物理变量、随机种子、场景 split、同预算方法、成功判据、统计单位、失败处理）、NOVELTY_AND_RISK.md、TRAINING_DECISION.md（选择或放弃具体正方法有实验依据）。
7. **06_RUNS_MANIFEST/**：新模型 ckpt/源码快照/run_config/输入 scene SHA/曲线/视频/真实状态与测试单例的索引；正式大文件留本地非Git，Git仅保存无敏感小型派生图及哈希/路径映射；不得把只给甲方内部看的真实现场衍生图公开。
8. **07_MENTOR_BRIEF_CN.md + PDF**：中文两页以内：解决问题、RL 在哪层、已完成真实数据、拟创新、风险与正在做的实验、预期交付，与研究方案一致。

**标题与摘要初稿**可用 SCI 论文常规结构“研究背景—已知缺口—拟方法—实验与结果状态—意义”，不是机械套用他人句子。至少给 2 个备选标题，并明确哪个是首选。客户版主要写可信现有成果与拟验证目标；正方法结果尚无时不能写“显著超越”。

## 5. 表格与图必须能审计

任何来自旧 v6b 的结果仅用已存真实评估：Geom feedback 8/32，residual seeds 13/18/23，recover 15/15/18；ID feedback 29/32，residual 28/28/29；Dynamics feedback 28/32，residual 28/29/29。说明 case 数、seed 数、历史已使用测试，明确普通残差不是新原创。P2 residual 6/6/7 每16与 feedback 5/16；不能伪造 B3 图表或填“expected +20%”。

若新 pilot 结果为负，也提交对应原始表/曲线/失败图，导师交付仍须完成；给出下一物理机制而非仅“重新训练”。**可设置目标（如非饱和新几何下 B3 对 B2 有具有实际意义的 P2 正增益），但目标不是测得值或录用保证。**

表格对统计口径做每种子/每场景分组；训练 seed 样本方差不是成功率的置信区间；成功率同时报告配对差与不确定性。绘图提供完整原始数据 CSV 的哈希、脚本和 LaTeX 引用；同一图中颜色、线型/marker与图例一致。

推荐真实渲染：同一个 Isaac 摄像机位、宽长比统一，场景光影干净、工件/导向槽/机械臂/目标路径可清晰识别，必要时小图放大关键接触区域。图注不写“真实料盘抓取”“场景全链”等未验证范围。截图 provenance 至少包括 source task SHA、asset SHA、Physics steps、checkpoint SHA、case ID、渲染外观 diff。

## 6. 历史 checkpoint/大文件实际清理（本轮必须执行）

先 du/find 统计 runs 与研究资产占用；提交 STORAGE_CLEANUP_PLAN.md（文件大小、SHA、被何论文/图/测试引用、唯一备份、可重建性）；**需在不损失历史正负结果可复现性的前提下实际释放无用空间**。保留正式模型的最终/必要中间 checkpoint、每种子权威配置/代码快照、真实失败证据、原始 eval、现场受控归档。对重复的中间调试 ckpt、已校验重复 USD/图/临时解压文件和确认为无引用的 cache，先哈希核对并压缩归档/删除冗余；日志索引重建后抽样加载正式 ckpt。不要删除唯一原始 trace 或公开发布已删底稿的不可核实图。
禁止全盘 rm -rf runs、Git 历史重写/force push、擅自 git gc、误清共享服务缓存或删除受控现场数据。给出清理前/后 GiB、保留/删除清单和不可清项。清理与新训练可并行，但不得影响正在运行的 worker。

## 7. 执行顺序、资源与保守门槛

- E0（立刻）：核实最新基线、写 client 提案初稿、用已有真实数据生成历史图表与首版 LaTeX PDF；**有真实图文成果先提交，不要等新算法正结果**。
- E1：新开发任务物理可执行性、Task-frame 强名义控制、参数可控性与局部P2失败标签细化。
- E2：B1/B2 同预算小 pilot；若 task-frame 变化为对照价值，再研究 B3 的实际后续状态/执行反馈机制。
- E3：B3 真实训练与可解释配对干预/消融；仅在验证集上依据预先固定条件判断是否扩 5 seed /最终测试，不读旧 Test-Geometry 调参。
- E4：完善 Fig6、Tab3/4、方法与结果章节、导师简报、存储清理报告，并提交明确的下轮训练指令。
- 可自主选 PPO、Residual SAC、合同动作投影、实际反馈接口等实现，但研究自由度必须**以完成 E2–E3 真正可运行训练为中心**。一条假设失败保留证据，改进任务/动作/物理观测后再用一轮有界实验检验，不允许无界换 5 种网络。
- 190 及其他服务器只按当前权限/空闲资源调度、单 worker 有超时与 RUN_COMPLETE、明确停止；不修改生产 ARES-R、真实机器人、驱动或重启；受控数据永不 push。
- 通用代码与无敏感研究方案可提交到该研究分支；需交甲方的细节若可能含现场数据，仅存受控本地位置，汇报文件用安全脱敏摘要。

**本轮至少应结束于：**一份可读的 6–10 页甲方方案 PDF + 可编译 LaTeX 工作稿 + 两张方法矢量图 + 两张真实仿真截图 + 从旧真实结果重建的图表/LaTeX 表格 + 一个新任务条件化 RL 的真实 PhysX pilot（含 checkpoint/动作/对照/负结果也可）+ 存储清理实际结果。不能只交大量审计和 TODO。

## 8. Codex 最终简报（面向甲方与导师）

用一页中文概述：
(1) 论文问题与合同关系；(2) 已有真实正结果是什么；(3) 新方法相对标准参数化技能 RL/ResidualPPO/STAP/Fabrica 的可能不同是什么，哪个差异经实验确认；(4) 新旧任务和场景是否吻合；(5) 真实仿真、新方法训练、实验数据和 LaTeX/图/表的可打开路径；(6) 本轮继续研究与仍待验证内容；(7) 清理空间大小；(8) 下轮只要做的具体 2–3 个实验。

核心原则：**争取正方法，而不是制造正结果；甲方图文与真实初步实验同步交付；保留长程技能/VLA 扩展接口，但不牺牲一篇可完成的 RL 方法论文。**
