# 研究最近邻论文与开源基座（限制搜索范围、明晰创新边界）

**用途：**让 Codex 面对一个新的方法想法先问“最近邻是否已有相同内容”，
而不是一开始无边界检索论文或将通用思想拼在一起自称原创。下表是启动
文献与源码索引；**尚不等于本项目已经完整复现各论文**。正式引用前需
逐项检查原文、官方版本、代码LICENSE、发布日期和确切算法归属。

| 领域与研究 | 直接资源 | 已有内容 / 我们必须区别 |
|---|---|---|
| 参数化原语：RAPS (NeurIPS2021) | [论文](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html) · [代码](https://github.com/mihdalal/raps) | 学习原语参数已经存在 |
| 后续技能规划：STAP (ICRA2023) | [源码](https://github.com/agiachris/STAP) | Q/后续技能价值和参数搜索已有 |
| 空间化动作：HACMan++ (RSS2024) | [项目](https://sgmp-rss2024.github.io/) · [代码](https://github.com/JiangBowen0008/HACManPP) | 哪里接触/选什么基元已有 |
| 规划+残差：Fabrica (CoRL2025) | [论文](https://proceedings.mlr.press/v305/tian25a.html) · [源码](https://github.com/yunshengtian/Fabrica) | 几何task-frame+残差和装配架构不是首次 |
| 名义策略残差：Policy Decorator (ICLR2025) | [代码](https://github.com/tongzhoumu/policy_decorator) | 冻结名义policy并通过残差微调已有 |
| 长程后续价值：Foresight Residual RL (2026) | [预印本](https://arxiv.org/abs/2607.16506) | 优化前段终态以利后续技能是最邻近的主张 |
| 残差价值基础：DAWN (ICML2026) | [论文](https://proceedings.mlr.press/v306/ma26s.html) · [代码](https://github.com/Guozheng-Ma/DAWN) | base-data critic 预热/归一化是现有训练技巧 |
| 动作效果适应：SCOUT (CoRL2026) | [项目](https://liy1shu.github.io/SCOUT/) | 预测/测量action-outcome的适应不是全新 |
| 接触阶段分解：Focus-Then-Contact (ICML2026) | [论文](https://proceedings.mlr.press/v306/qiao26f.html) | 接近与接触策略分工已有 |
| 工具抓持：Grasp to Act (RA-L2026) | [项目](https://grasp2act.github.io/) | 真实工具抓取与动态接触的RL适应已有 |
| 恢复技能：RecoveryChaining (IROS2025) | [代码](https://github.com/shivamvats/recovery-chaining) | 基于失败学习恢复与名义拼接已有 |
| 仿真与基线：ManiSkill | [代码](https://github.com/mani-skill/ManiSkill) · [RL 文档](https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/baselines.html) | 原生Panda任务、SAPIEN/PhysX、版本必须pin |
| 自有JAKA系统：Isaac Lab 2.3.0 | [对应版本任务](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html) | Isaac/JAKA是单独适配证据，不等于ManiSkill成绩 |

## 每个核心候选机制必须有一张最邻近对照

- 既有工作**具体算法/损失/动作空间**是什么？
- 借用哪些组件、如何pin版本/许可？无需全文复现可重用训练技巧；
- 我们新增的**不可被简单组合或换名替代的机制**究竟是什么？
- 哪个对照能排除只是强规则、动作权限变化、更多历史输入、
  额外离线数据或额外物理/规划查询造成优势？
- 同一套机制为何在不同任务条件下仍然有用？

只要没有独立实验证据，就在 `paper/CLAIM_MATRIX.md` 标记PROPOSED或
UNVERIFIED。不要因为论文发布日期较新而省略强先例。

## 2026-10-31 最近邻外审焦点（实际工作前需核对论文全文/代码）

- Sidhik等，**An adaptive framework for trajectory following in changing-contact
  robot manipulation tasks**，Robotics and Autonomous Systems 181 (2024)，
  https://doi.org/10.1016/j.robot.2024.104785
  已讨论预测模型误差、在线接触模式辨识和控制器调整。
- Liu等，**Hybrid Residual Reinforcement Learning for Contact-Rich
  Robotic Book Insertion**，2026预印本，
  https://arxiv.org/abs/2609.19962
  已将名义接近/插入控制与有界Residual PPO纠偏结合。
- Li等，**SCOUT**，CoRL2026，https://liy1shu.github.io/SCOUT/
  已利用动作—结果误差更新对未观测动力学的在线信念。
- Ma等，**DAWN**，ICML2026，https://proceedings.mlr.press/v306/ma26s.html
  及 https://github.com/Guozheng-Ma/DAWN；名义数据预热和Critic LN
  已是先例，其示例是ManiSkill2/v2，不能不经适配直接运行本机v1。
- **Constraint-Grounded RL for Variable Impedance Control**，2026，
  https://arxiv.org/abs/2609.13516；根据任务/力约束调整残差、
  推进率与控制增益已有先例。

因此“用动作效果预测+RL”“学习插入推进率”“切换接触阶段”
均不自动具备独立新颖性。方法核心应落在**任务/技能合同中的
有效物理作用条件、可测而不冗余的执行信念与跨任务参数决策**，
并通过几何/RLS/MPC/相同历史ResidualRL等强对照证实。
## 2026-11-01 论文主线所需理论与最近邻（非额外提纲）

- Sutton, Precup & Singh (1999) Option/SMDP：时间抽象技能、开始
  集/内部策略/终止，https://doi.org/10.1016/S0004-3702(99)00052-1
- Bacon, Harb & Precup (AAAI2017) Option-Critic：学习option终止策略
  已有成熟理论，https://doi.org/10.1609/aaai.v31i1.10916
- RAPS (NeurIPS2021)：已把参数化基元当作RL动作接口，
  https://proceedings.neurips.cc/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html
- STAP (ICRA2023)：相邻技能参数的几何依赖、Q价值、搜索，
  https://sites.google.com/stanford.edu/stap/home
- Foresight Residual RL (2026)：局部成功终态的下游价值与残差训练，
  https://arxiv.org/abs/2607.16506
- Fabrica (CoRL2025)：双臂规划/残差接触多物体装配，
  https://proceedings.mlr.press/v305/tian25a.html
- DAWN (ICML2026)：已研究名义策略数据锚定与Critic LayerNorm，
  https://proceedings.mlr.press/v306/ma26s.html
- SCOUT (CoRL2026)：动作-结果信念在线更新，
  https://liy1shu.github.io/SCOUT/

因此论文不能以“参数化”“加一个后续Q头”“加入动作历史”
“学终止门限”或“CriticLN”中的任何一项单独声称首次提出。
**原创机会必须从任务条件/真实SkillResult/后继可执行合同的
具体耦合机制与多任务独立增益中建立**。

## 2026-11-02｜针对真实多技能执行的新近邻（已读论文摘要/方法与官方代码说明）

更深入的逐论文方法、现有实验和下一物理闭环见 [THEORY_REVIEW_NEXT_RESEARCH_LOOP](evidence/EXP-003/2026-11-02/pyramid_contract/THEORY_REVIEW_NEXT_RESEARCH_LOOP.md)。
以下先例使旧的“后继合同/价值辅助”不能自动被称为原创。

- **Sequential Dexterity** (CoRL2023) https://proceedings.mlr.press/v229/chen23e.html / https://github.com/sequential-dexterity/SeqDex：真实前段终态分布、后继transition feasibility及后向微调。代码提供bi-optimization/tvalue_trainer，旧IsaacGym不可直接搬到ManiSkill。
- **ViSkill** (IROS2023) https://arxiv.org/abs/2307.16503 / https://github.com/med-air/ViSkill：value-informed技能链，提供train_sl.py和train_sc.py，说明后继价值学习本身是先例。
- **TASP** (ICRA2026) https://benned-h.github.io/tasp/ / https://cs.brown.edu/people/gdk/pubs/tasp_icra.pdf：black-box/力控/学习技能的initiation/termination与CIP head/tail可行性规划；不能把几何衔接和SkillContract接口独立声称新算法。
- **SPIN** (CoRL2025) https://proceedings.mlr.press/v305/jung25a.html：Skill-RRT+低扰动技能connector+planner distillation；减少强规划在线查询属于可研究的系统性能，但蒸馏本身已有先例。
- **SPIRE** (CoRL2024, PMLR2025) https://proceedings.mlr.press/v270/zhou25a.html：TAMP对长程任务做学习子问题分解，再结合模仿与RL；优先学物理接触剩余误差而不是替代全部规划。
- **Practice Makes Perfect / EES** (RSS2024) https://ees.csail.mit.edu/ / https://github.com/rai-opensource/predicators：基于技能能力与任务分布主动选择练习数据；“主动采样动作/任务”不是首次，predicators项目研究使用请注意README提醒。
- **Demystifying Action Space Design** (ICML2026) https://proceedings.mlr.press/v306/feng26ab.html：动作表示影响学习与稳定性；ARES三维SAC后投影Y-only不是严格二维网络，适合补强A线公平对照。
- **FAWAM** (CoRL2026) https://fawam.github.io/ / https://github.com/HaotianHehaha/FAWAM：接触力轨迹预测与执行时纠偏；官方代码截至本次核查仅发布Force-Envisioned模型训练/推理，在线residual correction代码尚未放出，不可宣称完整复现。
- **Uncertain task duration planning** (Robotics and Autonomous Systems2025) https://doi.org/10.1016/j.robot.2025.104926：情境条件化随机执行时间及限时规划早有研究，deadline/survival必须作为基线或与之区分。

核心思考：当前STAP后验误选、工具碰撞前附着漂移、双臂执行时间误判，可能属于“已验证几何可达之后，真实后段接触/执行时间仍不确定”的不同残差机制。先分开真实物理检验，防止将所有失败归因一个所谓新网络。强几何Pyramid36/36是控制资产，非RL正增益；研究状态继续UNVERIFIED。
