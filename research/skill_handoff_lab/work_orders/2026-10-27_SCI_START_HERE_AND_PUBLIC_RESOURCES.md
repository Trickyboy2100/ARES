# ARES SCI 自主研究闭环｜启动提要、数据入口与开源资源

**主任务书：** research/skill_handoff_lab/work_orders/2026-10-27_AUTONOMOUS_CLOSED_LOOP_SCI_RESEARCH_AND_VISUAL_DELIVERY.md

## 一、先读结论（不要从零开始审计）

- v6b 有真实的普通ResidualPPO未见几何适应初证（每32例：反馈8，RL三seed13/18/23）；**但P2只有每16例6/6/7对反馈5**，P1贡献很大，不能报“深槽显著提升”。
- v7已实际运行196,608次新环境交互，12训练模型，最新P2开发split是反馈30/32，任务坐标残差30/32、结果历史策略30/32，三个seed均无新机制成功增量；*这些数据来自不同split，不要和旧P2相减*。
- 科学上最缺：一个确实提供物理可控制权、能改进完整物理任务的**方法机制**，并能在同初态比较视频、真实轨迹、任务结果、消融和未见任务中得到重复支持。
- 合同需要“多模态具身感知与RL机械臂控制”；ARES-R上层系统只读参考，论文RL主攻中层可执行技能参数，近端接触是一个可用实验平台，不是唯一可选择任务。

## 二、开始后的真实产出优先级

**Priority 1｜立刻可见的诚实对照：** 读取既有独立eval、正确checkpoint、Task source hash；从同一case的独立PhysX环境获取 baseline/residual 真实对照 MP4 +关键帧+数值轨迹，并输出本研究“已有正结果/目前失败”的2页图文。不能只复用 v7 接近饱和的训练录像，不能只看漂亮截图判断胜负。

**Priority 2｜改变有效动作能力：** 真实 P2 失败中，工具接触/物体 yaw/横向余量实际可受哪些参数改变？分析 P2 的物理可控性，允许自主扩合法末端工具方向/接触位置/接近—重接触或换更具任务适应需求的物理技能，做有真实后缀的配对对照，不能瞬移物体。名义反馈与基线同步加强。

**Priority 3｜真正可训练的有界机制：** 用查到的有效动作接口训练强化学习，优先重用ResidualPPO，再按数据决定SAC/TD3/参数化Option/基于后续值的Actor选择等；每次与强反馈/原Residual/同许可参数的规则或枚举对照。真实seed/case都留证。累计做3轮有学习意义的研究闭环而不是3轮独立文献综述。

**Priority 4｜将结果变为论文：** 用真实图、真实视频、LaTeX表格更新既有 9页客户方案和两栏英文稿；保留现有v6/v7负结果与已封测试，明确论文主张是经验证还是 proposed。单独出版配对视频/filmstrip 图/具体物体运动指标和二维码或HTML索引（不得将内网链接伪装成公网可访问）。

## 三、推荐优先精读的公开论文及实现（所列项目已找到实际公开页面）

| 优先级 | 论文/源码入口 | 为什么读 | 注意 |
|---|---|---|---|
| 1 | **HACMan++** https://sgmp-rss2024.github.io/ ; https://github.com/JiangBowen0008/HACManPP | 真正联系“在哪儿接触、怎么执行、选何基元”的RL动作设计；借鉴critic maps视觉表达 | 旧Gym/MuJoCo/Torch版本，严禁污染既有Isaac环境；不能称首次提出接触点动作 |
| 1 | **Fabrica, CoRL2025** https://fabrica.csail.mit.edu/ ; https://github.com/yunshengtian/Fabrica | 规划+task-frame residual物理装配，最接近本项目拟方法；用来找我们尚缺的真正不同机制 | 其learning是IsaacGym环境，不直接替换Isaac Sim5.1 |
| 1 | **RecoveryChaining, IROS2025** https://github.com/shivamvats/recovery-chaining | 真实失败、局部恢复、名义后缀/额外执行预算的做法 | R11式“hold/recover”已在本项目失败，不可原样重刷 |
| 2 | **STAP, ICRA2023** https://sites.google.com/stanford.edu/stap/home ; https://github.com/agiachris/STAP | 真实后续技能价值、CEM参数搜索公平强baseline | 不是只接预测头，而是用价值实际选择动作 |
| 2 | **RAPS, NeurIPS2021** https://github.com/mihdalal/raps | 学习低维合法动作基元参数，而不重学全部关节 | 参数化原语不是独立新颖性 |
| 2 | **Residual Skill Policies/ReSkill** https://github.com/krishanrana/reskill | 普通残差与技能先验基线、训练工程经验 | 不搬所有旧依赖/演示数据 |
| 工程 | **Isaac Lab** https://github.com/isaac-sim/IsaacLab ; **官方录像参考** https://isaac-sim.github.io/IsaacLab/main/source/how-to/record_video.html | Isaac相机/视频采集与单环境/16环境可视化 | 公开页面为最新版，先核本机2.3.0 API |
| 可选基准 | **ManiSkill3** https://github.com/mani-skill/ManiSkill | 多种任务、GPU仿真和现成PPO/SAC资源，必要时作为外部验证 | 在独立env中做，SAPIEN不等于Isaac PhysX；资产许可检查 |

**依赖与科研原则：** 优先看核心机制、LICENSE和具体可移植代码片段，pin commit。外部开源 benchmark 只在确实有助于论文独立性时才引入，不必跑全部官方实验。不得复制官方结果当成我们的、不得使用未获授权的外部论文图/视频/模型。

## 四、马上做的3项

1. 检查本机 HEAD/未提交改动、已有runs、GPU和任务脚本；必要时新 worktree，严格分离 ARES-R。先恢复一个实际v6对照baseline/residual在同一case的物理运行并真实录制。
2. 建立一个专门的paired-video recorder（同初态、两策略独立运行，视频、时间戳、真实轨迹和独立验收挂钩），这是以后每轮研究都复用的成果基础设施。
3. 用实际P2失败复现和物理动作敏感性试验确定下一轮具体参数策略。若H1无法救回，主动试另一个有可执行物理调整空间的任务/方法，而不是再次训练无意义的B3历史输入。

**交付路径：** 研究报告/小型图表到 research/skill_handoff_lab/deliverables/client_sci_20261027/08_ITERATIVE_RESEARCH；视频和大模型到非Git runs/client_sci_closed_loop_20261027；既有中文方案/英文LaTeX同步修订。所有发布产物写明 EXACT/REPRODUCED/PROPOSED，及物理任务/模型/场景真实哈希。
