# 唯一论文主张—证据矩阵（不因实验数量自动更新摘要）

当前工作题目：
**Task-Conditioned Executable Skill-Parameter Reinforcement Learning
for Adaptive Contact-Rich Robotic Manipulation**。
研究领域/标题可以根据真正的正机制收缩，不能据未实现功能扩张。

| 欲写成论文的主张 | 现有支持 | 必要的新增验证 | 状态 |
|---|---|---|---|
| 名义规划与物理接触结果存在可测差距 | 历史P2、Stack、Charger真实物理案例 | 新任务下强名义控制与可重复失败分类 | PARTIAL |
| 任务条件化合法动作能改变完整物理结果 | Pyramid 9候选和Charger同后验参数试验 | 新独立场景、同动作复跑、因果机会与强规划比较 | PARTIAL |
| **我们提出的RL新机制** 比普通RL/规划更有效 | 尚无 | 同动作/同obs/同数据预算、独立seed与关键消融 | **UNVERIFIED** |
| 执行效果表示的作用是独立的 | 初步物理效果记录，历史输入消融无收益 | 匹配history，真实效果残差参与动作决策；新条件 | **UNVERIFIED** |
| 方法在不同任务机制和未见条件泛化 | 原有两个独立环境各有普通基线结果 | 新方法跨第二接触任务、OOD/传感误差、复现实验 | NOT_TESTED |
| JAKA/ARES-R适配可用 | Isaac研究初步基础、EG2仍有mimic问题 | 正确抓取物理验收，模拟接口与全任务实验 | NOT_TESTED |

禁止将局部参数例图、oracle上界或已看开发集包装成新方法因果证据。
论文用表/图应由真实实验生成，引用与来源见EVIDENCE_LEDGER/PRIOR_ART。

**正式撰稿启动**：一项独立方法机制达到可重复正向实验与关键消融后；
此前保留现有seed manuscript，不反复编译新PDF。投稿前追加严谨统计、
鲁棒性/泛化和完整工程限制；不凭“模型/视频/PDF交付完成”宣布可投稿。
