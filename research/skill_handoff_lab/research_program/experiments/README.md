# 实验工作台：每张卡片解决一个科学不确定性

**动态任务唯一权威**：上级 `STATE.json` 中 `active_experiment`。
本目录主要记录**假设—真实运行—科学判断**，不复制大型原始数据和影片。

| 实验卡 | 研究目的 | 初始状态 |
|---|---|---|
| [EXP-001](EXP-001_BASELINE_AND_CAUSALITY.md) | PlugCharger强基线、重复随机性、失败归因 | READY |
| [EXP-002](EXP-002_ACTION_AUTHORITY.md) | 真正可控的接触阶段动作、实际物理纠偏能力 | QUEUED |
| [EXP-003](EXP-003_STRONG_RL_BASELINES.md) | 成熟在线ResidualPPO/SAC及DAWN基线 | QUEUED |
| [EXP-004](EXP-004_EFFECT_CONDITIONED_METHOD.md) | 执行效果条件化机制和同信息消融 | QUEUED |
| [EXP-005](EXP-005_GENERALIZATION.md) | 未见场景、第二任务、感知与JAKA适配 | QUEUED |
| [模板](_TEMPLATE.md) | 新任务的标准短卡，按需要复制 | TEMPLATE |

工作完成时在对应卡片末尾追加真实摘要、决策、源/场景/模型哈希、
运行路径及下一实验；同时把STATE中的状态从running改成done或pivoted。
新任务建立EXP-006等唯一ID并在STATE与本目录索引同步注册。

一个实验可以跨Codex会话，没做完不必为了“有新文档”关闭。
核心真实性：真实训练、真实物理完整任务、视频回放三者必须分别标识。
