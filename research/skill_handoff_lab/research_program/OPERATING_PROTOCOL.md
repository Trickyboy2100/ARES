# Codex 持续研究操作协议（不是一次性的固定提示词）

## 0. 每次执行都从同一处恢复

1. 检查Git当前branch/HEAD、dirty文件、已有本地模型、其他GPU作业与隔离环境；
   任何非自己进程/工作树不动。
2. 读取根/Scoped AGENTS、`README.md`、`STATE.json`、`PAPER_CONTRACT.md`；
   只读取活动 `experiments/EXP-...md`、相应源代码和EVIDENCE的必要部分。
3. 运行 `python .../scripts/program_status.py check`；若失败先修索引，
   不以这个检查代替科学实验。
4. 基于活动实验从实际物理失败里形成有可反证的**本轮假设**，可以自主搜索
   最近邻论文、改变合法接口/控制算法、跑多轮有界实验直到当前会话可用
   资源用尽。无需每次小改动都等待用户新Prompt。
5. 审查强基线、动作权限、观测、成本和原生成功；写入实验卡、证据链接，
   追加DECISIONS、更新STATE。若活动问题结束，选依赖已具备的下一个
   实验；物理机会不成立则返回前一阶段并说明原因。
6. 提交到独立研究分支，真实数据留非Git/许可范围；向用户给出精简中文报告。

**会话与后台事实**：该协议保证下一次Codex能读到持久研究状态，
并不意味着在单次会话退出后AI还在后台执行。若用户希望持续的定时研究
须另有明确调度与算力/权限配置。不要将等待/时间表写进论文规划。

## 1. 每张实验卡必填

`Question`、`Prior best explanation`、`Manipulable action`、`Matched comparator`、
`Train/Dev/Sealed split`、`Measured physical outcomes`、`Budget/data cost`、
`Outcome+limitation`、`Next decision`、`Exact paths+SHA`。
训练实验还需模型、初始/终权重与种子、checkpoint、梯度/训练曲线来源。

## 2. 统计/公平硬边界

- **所有策略都从合法起点真实物理闭环**；状态重放只作为注明的影片。
- 官方benchmark的物理、成功阈值、horizon不为了提高ours得分而改；
  任何定制任务另命名，不冒充原任务排行榜。
- 同case的多个seed/参数、多个物理步骤并非独立场景；场景组和训练种子
  分开分析，带配对混淆表与适用的置信区间。
- 强名义反馈、修正规划器、普通同接口ResidualRL、同候选监督/CEM均能
  使用合法动作。新算法不能偷偷多用未来success/仿真隐藏真值。
- 同动作/同初态重复用于估计随机性，尤其MPLib RRT fallback；规划源/缓存
  /求解次数、episode steps、成功时钟分开记录。
- 若高质量baseline已完全解决当前任务，不造RL虚假优势；切换有依据的
  合法任务条件。动作有效而RL不学习则修训练，不能只加epoch。
- 原先训练/开发集结果被查阅后不再是最终密封测试；新机制确定后冻结
  新独立sealed场景，一次正式评估，不在test上选模型。

## 3. 每次运行的最小记录

在非Git `runs/<program>/<experiment-id>/<unique-run-id>/`（示意，不要求
硬改现有目录）保存 `manifest.json`：研究commit、确切执行源码/配置
哈希、依赖/仿真器版本、官方任务/是否改动、策略/控制接口、名义控制、
scene seed/group/split、episode数量、真实物理step、有效RL决策数、
模型训练交互、离线数据与规划搜索成本、success独立验收、失败分类、
输出checkpoint/trajectory/video精确hash，及是否包含现场数据。

Git中只放实验卡、精简数值、可复现脚本、必要论文图和模型/大文件索引。
不覆盖唯一旧数据，不删除失败，清理必须有冗余验证。视频只在解释真实
行为或准备对外阶段成果时制作有代表性的正/负配对，不必每轮73段。

## 学术图文原版和机制 Highlight

代表性真实物理样例遵循
[VISUAL_EVIDENCE_PROTOCOL.md](VISUAL_EVIDENCE_PROTOCOL.md)：同源
RAW、ANNOTATED、TRACE和英文图注，保持SHA/scene/帧号；
区分姿态回放与独立物理运行，记录恰当因果解释等级。
一轮只需少量科研最有信息价值的图，不反复生成整篇论文PDF。
## 4. 每轮研究总结推荐格式

**实验ID / 研究命题** → **最新结果对最强基线** →
**证据支持/否定何种机制** → **成本与限制** →
**接下来最有价值的新实验/回退动作**。

只更新实验卡 + STATE + DECISIONS（若有新论点同时更新EVIDENCE/CLAIM）。
不必更新客户PDF或LaTeX，除非主张发生实质变化或进入S7投稿。
不为了文档数量而浪费实验预算。

## 5. 长期方案维护

`STATE.json` 的 active_experiment 必须指向现存的实验卡；
所有后续实验可以自主新建编号EXP-006等，同时在实验目录索引中注册。
一个活动研究问题可以跨多次会话；没有结果时诚实保留running并写当前
已运行的检查、瓶颈、完整证据。已完成的实验卡状态“done”只表示
实验与科学判断已记录，不表示方法一定得到正性能。

## 研究自主性与当前证据

EXP-001的独立控制器科学验证已完成。Codex应根据最新STATE
进入EXP-002，不再重复原8场景的规则调试，也不因一次“新场景也
被强控制解决”就返回空的Gate汇报。可以连续做官方新初态、
合法物理变化操作族、实际动作权限及在线普通RL能力实验；
方法/场景发生改变须留真实控制器对照、训练数据及任务命名。
如果在同一会话产生足够可学习证据，主动更新STATE并推进EXP-003，
不需要用户重新发一条长Prompt。只有在真实数据支持后才把
新机制写进论文贡献；每轮视频挑少量有机理价值的案例。
