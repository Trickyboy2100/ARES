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
