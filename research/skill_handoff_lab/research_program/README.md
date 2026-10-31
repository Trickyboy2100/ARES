# ARES SCI 研究总控索引 — 唯一长期入口

**这是 ARES 甲方 SCI 方法论文研究的当前指挥目录，不是又一份一次性工作单。**
旧 `work_orders/`、实验结果 `deliverables/`、全部源码、模型与负结果均继续保留。
此处不复制大文件、不改 ARES-R，也不把研究路线绑到日期。

## 永久研究问题

> 面对不同工件、目标与接触偏差，机器人能否在复用既有规划/控制技能的前提下，
> 通过强化学习基于任务条件与**真实执行效果**调整合法技能参数，改善完整任务
> 成功、可靠性、学习效率及对未见操作条件的适应？

**工业目标**：能与 JAKA MiniCobo 双臂移动平台的感知、Skill Runtime、
cuRobo和控制接口兼容。**当前研究证据**：公开 ManiSkill Panda 的 SAPIEN/PhysX
与自建 Isaac Lab/JAKA 研究仿真；不是实机或已训练多模态 Actor。
**当前新方法独立增益**：尚未证实，不能把普通残差 PPO 正结果写成新算法。

## 先看哪里（唯一动态状态以 STATE.json 为准）

| 阅读顺序 | 文件 | 用途 |
|---|---|---|
| 1 | [STATE.json](STATE.json) | 当前阶段、唯一活动实验、下一具体动作与已经了解的风险 |
| 2 | [PAPER_CONTRACT.md](PAPER_CONTRACT.md) | 一篇论文到底解决什么、贡献与禁区 |
| 3 | [STAGE_ROADMAP.md](STAGE_ROADMAP.md) | 不设日程的阶段依赖、完成证据和回退路线 |
| 4 | [EXPERIMENT_BACKLOG.md](EXPERIMENT_BACKLOG.md) | 研究实验队列及优先级、依赖 |
| 5 | [METHOD_STRATEGY.md](METHOD_STRATEGY.md) | 科学方法/训练梯度、强基线、算法演进与对比设计 |
| 6 | [EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md) | 历轮真实结果、负结果、精确出处，不假装已验证方法 |
| 7 | [PRIOR_ART.md](PRIOR_ART.md) | 近邻论文、源码、必须避免重复的已有贡献 |
| 8 | [OPERATING_PROTOCOL.md](OPERATING_PROTOCOL.md) | Codex 自主续接、实验质量、算力/视频/数据规范 |
| 9 | [DECISIONS.md](DECISIONS.md) | 为什么保留/放弃一项方法及下一科学决策 |
| 8a | [VISUAL_EVIDENCE_PROTOCOL.md](VISUAL_EVIDENCE_PROTOCOL.md) | 原版、英文圈注、机制曲线与真实来源规范 |
| 10 | [paper/CLAIM_MATRIX.md](paper/CLAIM_MATRIX.md) | 投稿前的论文主张—证据映射，暂不反复生成PDF |

**新增真实视频联动工具**：
[program_charger_video_sync.py](../tools/program_charger_video_sync.py)
支持从同案例的真实物理状态记录生成 RAW.mp4、ANNOTATED.mp4、
DATA.json、MANIFEST.json 和 SYNC_VIEWER.html；视频与曲线共享物理step。
已在本地RTX3080实际生成并验收：[完整视频与同步曲线](evidence/EXP-002/2026-10-31/video/README.md)。[新偏差场景救回对照](evidence/EXP-002/2026-10-31/new_video/SYNC_VIEWER.html)也已生成，支持接触段放大和真实/观测误差同步。原版/高亮视频均201帧；属于MEASURED_POSE_REPLAY，新增物理步0。

可执行实验卡片在
 [`experiments/`](experiments/README.md)。**动态任务只有一个来源：`STATE.json.active_experiment`。**
如果文件里的“当前”摘要与 STATE 不同，以 STATE 与最新实际证据为准，修正陈旧摘要。

## 一次启动，持续续接

从 Git 根目录：

```bash
python research/skill_handoff_lab/research_program/scripts/program_status.py status
python research/skill_handoff_lab/research_program/scripts/program_status.py check
```

启动 Codex 时只需要一次入口指令：

> 请按 `research/skill_handoff_lab/AGENTS.md` 和
> `research/skill_handoff_lab/research_program/README.md` 自主继续一篇 SCI 方法
> 论文研究。先检查 `STATE.json` 中活动实验、原始证据、运行环境与工作区。
> 按阶段依赖连续进行有研究价值的物理诊断、算法训练和强基线比较，迭代
> 到当前可用会话资源允许的程度；更新活动实验、决策记录与 STATE，提交
> 真实结果。不要停在写计划/机械Gate，不制造正结果、不改实机/ARES-R。
> 每轮只产出必要数据、代码与核心图表，方法验证后再集中更新论文。

**能力边界**：一次 Codex 会话可以连续研究和提交多轮，但它不能靠一份
Prompt 在会话结束后永久自动运行。下一次会话从同一入口、同一 STATE 恢复，
不用用户重新拼装前轮上下文。

## 历史可追溯入口

- [最新已执行：PullCubeTool/PlugCharger 研究](../deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029/README.md)
- [Pyramid 信息与机会分析](../deliverables/client_sci_20261027/12_SCI_OPPORTUNITY_20261029/README.md)
- [两真实技能边界与RL比较](../deliverables/client_sci_20261027/11_PAPER_FIRST_20261028/README.md)
- [参数学习与风险消融](../deliverables/client_sci_20261027/10_METHOD_EVOLUTION_20261028/README.md)
- [公开基准转向](../deliverables/client_sci_20261027/09_BENCHMARK_PIVOT/BENCHMARK_SCOUT_DECISION.md)
- [实际 PhysX 闭环/可视化](../deliverables/client_sci_20261027/08_ITERATIVE_RESEARCH/index.html)

以上是**历史证据**而非仍需逐条执行的旧工作单。
