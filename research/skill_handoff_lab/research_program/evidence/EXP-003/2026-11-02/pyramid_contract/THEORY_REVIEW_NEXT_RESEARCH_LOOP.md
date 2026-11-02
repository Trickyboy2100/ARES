# 2026-11-02｜基于累计物理证据与最近邻论文的下一研究循环

状态：**理论审查 + 已保存数据的重新分析；本文件不代表新PhysX或新RL训练。独立算法增益 UNVERIFIED。**

## 1. 不重启课题：科学合同与已证实的能力

不变的科学问题为“任务/后继技能条件下，使用真实 SkillResult 和合法技能参数的强化学习，改善完整多阶段操作”；现有 STAP 核、普通 SAC、名义几何/反馈、实际 SkillResult 边界、对数Q排序、物理轨迹/同步视频全部继承。Panda/ManiSkill物理和JAKA/ARES-R工业运行完全区分，ARES-R只读。保证原生成功标准、全episode时限、实际物体/抓持连续性、相同信息及动作权限。

**最新 EXP-003 Pyramid**：90条真实PhysX、22500控制步、0优化；nearest symmetry29/36、四等价姿态可行路径36/36，后两者36个逐步完全相同的前段物理前缀、7救回/0退化。四姿态查询144次，全部25个较弱方法失败都是 C_PREPLACE 规划错误。四个(2物理scene×2要求yaw)条件在强几何下均九参数成功；没有学习完整成败的额外空间。**RETAINED：对称等价姿态可行性检查；LOCAL_NEGATIVE：该yaw面板不是RL算法机会。** 修复是规划器贡献，非RL。

**历史 576 条 Pyramid 原生物理**：64 scene×9参数；名义52/64、最佳固定59/64、有限实测动作事后oracle62/64。存在9个“合法前段参数影响后继终局”的scene、13个“同后段初态更换后段合法参数会改变结果”的scene，两类交集6个scene（420010/022/026/036/038/052）。相对名义A及名义C，420002和420023只有同时改变两个阶段参数才能在已测网格中找到成功组合。这不是超过最佳固定的学习得分，更不是独立方法增益；提示早期真实双阶段依赖没有因为新yaw面板被全部否定。需避免在历史已看case上挑模型测试。

**工具 train96**：8物理scene、每scene 2目标、每目标6个合法动作，16条件中7组全6成功、7组全6失败、2组混合；仅49/96完整成功，固定8/16、已测oracle9/16。注意即使二元成败相同，实际目标距离仍可随参数变化，例如 all-fail 场景743043/+80mm的距离范围约229–348mm，不能用“成败无区分”冒充“物理作用不变”。连续量是探索解释指标，不能事后修改论文终局主指标。

**未解释的可研究残差**：STAP同物理前段开环6/8、重规划3/8；后验Q接近零却被迫argmax，真实物理/预算与模型偏离；工具35mm抓持在首碰撞前约10.4°漂移，减速减少冲击而未治愈抓持漂移；TwoRobot完整后继几何8/8，同初态改抓取轴2/4，其时限预测把7个实际成功误判为不可执行。上述现象涉及不同失败模式，尚不能拼成一种统一新算法。

可从仓库运行标准库审计脚本 `python research/skill_handoff_lab/research_program/scripts/audit_cumulative_skill_opportunities.py`，读取本目录SUMMARY、HISTORICAL_OPPORTUNITY以及skillresult/EPISODES.csv，不运行GPU。新程序在连接的repo中已提交，须在Codex Ubuntu工作树运行并验证。

## 2. 近期文献、源码与不可重复发明的东西（核对到2026-11-02）

| 方向 | 原始来源、源码 | 已有机制；本项目应借用/不能主张独创 |
|---|---|---|
| STAP (ICRA2023) | https://github.com/agiachris/STAP ; https://arxiv.org/abs/2210.12250 | skill Q×dynamics×CEM及风险规划；pin旧核并保留强规划基线，不把重规划/置信拒绝包装为新方法 |
| SeqDex (CoRL2023) | https://proceedings.mlr.press/v229/chen23e.html ; https://github.com/sequential-dexterity/SeqDex | 用实际终态/短历史训练 transition feasibility，再后向训练前阶段和策略切换；实现依赖旧IsaacGym，优先借理论/结构，不直接换整套环境 |
| ViSkill (IROS2023) | https://arxiv.org/abs/2307.16503 ; https://github.com/med-air/ViSkill | 从实际后继价值训练技能连接，代码 train_sl/train_sc；“增加后继价值头”不是独创 |
| Foresight Residual RL (2026) | https://arxiv.org/abs/2607.16506 | 真实后段统计标签加前段残差训练；当前ARES是state后继价值类比而非官方VLA实现 |
| TASP (ICRA2026) | https://benned-h.github.io/tasp/ ; https://cs.brown.edu/people/gdk/pubs/tasp_icra.pdf | initiation/termination与可组合Interaction Primitives，用运动规划连接黑盒/力控/学习技能；合同+几何连接是强规划组成而非创新 |
| SPIN (CoRL2025) | https://proceedings.mlr.press/v305/jung25a.html | Skill-RRT规划、技能接续connectors、规划蒸馏；只把4次对称搜索蒸馏为单网络并不自动是原创 |
| SPIRE (CoRL2024/PMLR2025) | https://proceedings.mlr.press/v270/zhou25a.html | TAMP分解后先用模仿奠基、在真正有必要的局部动作上RL，适合作为复用成熟底座的研究组织方式 |
| Practice Makes Perfect / EES (RSS2024) | https://ees.csail.mit.edu/ ; https://github.com/rai-opensource/predicators | 对参数化技能根据长期任务收益主动选择练习样本；“主动选择困难case训练”也有强先例 |
| DAWN (ICML2026) | https://proceedings.mlr.press/v306/ma26s.html ; https://github.com/Guozheng-Ma/DAWN | base-policy数据预热+Critic LayerNorm；可增强普通ResidualRL，但不算原创；源码ManiSkill2需明示版本适配 |
| Demystifying Action Space Design (ICML2026) | https://proceedings.mlr.press/v306/feng26ab.html | 真实机器人动作表示/空间影响学习与稳定性；ARES 3D-SAC之后屏蔽Z并非真正2D Actor/Critic/熵，需要公平2D对照 |
| FAWAM (CoRL2026) | https://fawam.github.io/ ; https://github.com/HaotianHehaha/FAWAM | 力/力矩预测及执行时残差校正已有人研究；官网源码目前只发布部分Force-Envisioned模块，残差纠正实现尚未释放，不能声称全量开源可复现 |
| Contextual uncertain-duration planning (RAS2025) | https://doi.org/10.1016/j.robot.2025.104926 | 时间限制下的状态条件化随机执行时长已有理论，不可将“把剩余步数喂进Q”称首次发明 |

## 3. 更值得验证的科学切口：几何可行之后的真实执行条件

把最新三轮的证据组织为 *三种竞争解释*，不能一开始合并成大网络：

**G：确定性几何/合同不足**。如yaw的C_PREPLACE与Pyramid夹指扫掠碰撞，增加有效姿态搜索/有效运动规划可直接消除。应保留为强基线，而不是算法机会。

**E：几何名义条件相似，但SkillResult中的实际接触/抓持效果改变后段物理转移**。工具首碰撞前抓持漂移和接触事件说明此假设值得检验。要区分原阶段根本没稳定完成、后段碰撞、实际附着变化，验证真实后续动作是否能在不破坏强控制的前提下救回，不能把有问题的名义完成标签当学习信号。

**B：物理序列时长与后验价值偏差**。STAP后验错误argmax和TwoRobot时长假阴性提示：同几何动作计划不能等于真实100/250步下成功；物理阶段切换/反馈消耗时间。可先复用可观测控制状态、校准时长与成功概率、同权限在线RLS/几何MPC和STAP UQ；不得仅凭更小位置MSE就定义更好决策。

若E/B有实际共同案例且对合法动作的排序有影响，可以研究**执行结果及剩余预算条件化的技能级SMDP价值**（描述性候选，不是已原创）：
Q(s,a,C,b)=E[r + gamma^tau * 1(tau<=b) * V(s',C,b-tau)]。
s'与tau来自真实SkillResult和实际步数。可进一步显式区分几何可达、后段真实接触完成与剩余时间概率，或使用事件/生存式监督处理在horizon停止的未完成运行。必须与标准SMDP Q、SeqDex/ViSkill后继feasibility、Foresight类比、STAP风险规划、单独时长模型及直接几何规划对比。现有方法思想高度邻近，只有新训练机制经独立消融和正结果才能声称独立增量。

**不要仅因为没有“最优参数严格反转”就停止研究**。同权限参数可能影响接触裕量、阶段时间或真实成功概率；但主论文全任务真实终局成功优先，成本/数据效率主张需在独立评估前明确。

## 4. 恢复较早df268c0那种持续自主研究闭环（下一Codex会话执行，不是本文件已运行）

第一段在旧记录中开展实际探查，不重复大扫参：读取Pyramid的576真实原始轨迹与两例需双参数改动的seed420002/420023，再核对9个前段效应scene与13个后段动作敏感scene。检查支持面释放、完整后段几何、预算和物理接触错误分别占多少；旧例只能研究因果，不是新方法测试。并行复查TwoRobot适应条件下到位/时限实际误差及Tool g035先漂移后碰撞过程的100Hz证据。

接下来在**现有任务实现**上选具有明确物理依据的最小新条件组合：优先尝试一类真实后继接近/净空要求的Pyramid或TwoRobot交叉物理配对，务必保留刚验证的四姿态几何可行搜索和完整后段几何；若依然all-pass不继续重复该几何网格，而主动在同会话转至工具接触/稳定抓持与后继施力/剩余预算问题。只有在物理机制需要时才引入任务变体，公开按新名字标记，且对所有方法共同有效。不能以人为降低名义性能、延长horizon或调紧成功阈值制造胜利。

候选任务数据至少有两个实际参数决策、前段真实状态未reset、后段根据测得SkillResult重新选择。先测同一物理初态下2种合法任务要求×2–4种前段参数×2–3种真正执行的后段动作，从真实物理计算的区域中有针对性地扩展；不能只看离线oracle/静态IK。比较强固定、强几何、实际可执行预算控制、相同接口普通RL或有能力的训练器。几何强解决则冻结规则资产并转向下一机制；真实可利用物理差异出现则**同一会话继续实际训练**而不是只写 Gate。训练总账分开记录episodes、有效RL决策、optimizer updates、物理步、规划查询及墙时，含失败和失败时长右删失。物理结果反常可自主追加1–2个竞争假设的单因素干预，像df268c0主动检验先滑移还是先撞击，不给每个小变更新工作单。

若发现高质量未见分布仍有物理机会，先训练同动作/信息/预算普通两边界RL、一个严格的后继feasibility实现，再研究一个最小执行效果或有界时长模块；不要直接复活失败的relative+contract五头。与A线的真正2D Actor/Critic（非3D遮罩）短训练作为能力基线并行，不抢主线；改善的基础算法属于已有方法。

最终每轮只需：新PhysX验证数量与原始记录、一个最有辨识力的真实物理对照图（RAW+ANNOTATED+同步曲线）、强基线配对成功/退化与实际成本、清楚的RETAINED/LOCAL_NEGATIVE/OPEN_MECHANISM/PROPOSED_METHOD及下一项实验，并更新STATE/EXP-003/DECISIONS/EVIDENCE。优先完成新研究闭环，不重编论文/不刷测试种子，ARES-R永远只读。

## 5. 是否可以立即宣称创新？

不可以。原生强控制器和最近90条的确定性增益已成立；任务条件价值、执行效果残差、预算学习都已有强理论先例且目前没有在新物理条件中显示独立收益。下一步的科学成果不是包装上述符号为新算法，而是找到**在最强几何和成熟两边界RL之后仍有可辨识的真实后继执行风险**，然后通过一个可消融的新学习或决策机制持续削减它。

## 6. Tool-contact额外高价值对标：Grasp to Act开源代码

Grasp to Act (IEEE RA-L 2026)：https://doi.org/10.1109/LRA.2026.3677744 ；官方完整实现 https://github.com/hgupt3/g2a ，提供 LEAP hand 抓持优化、真实力矩扰动下的稳定性检测、残差PPO训练和预训练策略。README的实际命令包括 `python -m g2a optimize`、`train`、`evaluate`，环境支持Isaac Sim5.1 / Isaac Lab2.3.2；与ARES当前Panda平行夹爪或工业JAKA硬件**不可以直接等价移植**。方法已经研究外力作用下的功能抓持、关节残差与抑制滑移；因此工具实验若仅把抓持稳定性RL做出来，不构成首次创新。下一步优先阅读其力矩/真实抓持评估protocol并在当前工具物理原生成功、时间、稳定合同下实现一个公平可复用的**强抓持基线**；再看真实后继技能条件与完成预算是否有进一步独立学习价值。
