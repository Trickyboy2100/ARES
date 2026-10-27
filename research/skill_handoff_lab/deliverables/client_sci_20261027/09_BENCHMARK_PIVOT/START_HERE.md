# 2026-10-27 公开基准转向：真实成果入口

## 结论先看

已完成 Isaac Lab 原生 Lift / Stack / Drawer 的 1、4 环境物理运行，以及
ManiSkill3 Pick / Stack / Peg 的规划、反馈及真实 PPO。7 个最终 checkpoint，
合计 **2,560,000 次在线训练交互**，另保留全部 scout 与评估交互。

| 同一 64 案例开发验证 | 最终成功 |
|---|---:|
| 充分结构化反馈 | 58/64 |
| 普通残差 PPO，seed31 / 37 / 43 | 61 / 62 / 60（各 64 例） |
| seed31 残差强制置零 | 58/64，逐案例与反馈相同 |
| 同接口 Cartesian PPO，256k | 0/64 |
| 官方关节 PPO，1.024M | 0/64 |

残差平均 95.31%，样本标准差 1.56 个百分点；反馈 90.625%。这是普通残差的
初步重复性证据，不是新算法贡献。每颗种子修复相同 5 个失败，但各引入 2、1、3
个新失败。seed31 的配对精确检验 p=0.453125，不能宣称统计显著优势。
同一 64 案例反复测试不是 192 个独立场景。Cartesian 与关节 PPO 的接口不同，
后者不能作为机制公平消融；本轮训练也不是充分调优的最终 PPO 上限。

## 直接打开

- `BENCHMARK_BRIEF_CN.pdf`：3 页中文图文简报。
- `PAPER_PIVOT_DRAFT.pdf` / `.tex`：可编译论文补充稿，明确未达到投稿状态。
- `paired_improvement.mp4`：反馈失败、普通残差成功的同初始状态录像。
- `paired_regression.mp4`：反馈成功、普通残差退化的同初始状态录像。
- `RESULTS.json`、`PAIRED_CAUSAL_AUDIT.json`：全部数值与动作因果审计。
- `JAKA_INDEPENDENT_AUDIT.json`：JAKA 实际物理运行后的限位/mimic 失败。

完整便携包另含 `index.html`、7 个最终模型、真实 Isaac/JAKA 视频、源代码快照与
`RAW_DATA_INDEX.json`。模型及全量视频保存在非 Git 目录，不推公开仓库。

本机完整运行根目录：
`/home/andyee/robot-research/ARES-Skilllab/runs/benchmark_scout_20261027/`

便携包：该目录内 `client_delivery/`。NAS 目标为用户指定 Projects 下独立的
`ARES_SCI_20261027_BenchmarkPivot/`，不覆盖旧 ClosedLoop 包。

## 研究决策与未完成项

公共主平台选择 ManiSkill3；Stack 用于参数/释放/恢复的快速闭环，Peg 是第二种
接触条件候选。JAKA 双臂移动平台和 ARES-R 技能架构仍保留在 Isaac Lab 应用轨道。
本轮 JAKA + EG2 实际运行 720 物理步，但夹爪关节越限和 mimic 偏差验收失败，
不得描述为抓取、Stack 或跨机器人迁移成功。

下一轮最值得投入的是：同观测同合法动作范围下，**执行结果感知的技能参数/恢复决策**
与普通残差 PPO 的真实消融，并在 Peg 接触条件及独立封存测试场景上验证。
先改进物理失败对应的参数和恢复接口，不盲目增加网络。普通残差已是需要超越的基线。
没有独立方法增益、未见场景结果和第二任务训练前，尚不足以形成 SCI 核心贡献。

## 保全、许可与复现

旧 P1/P2、现场轨迹、所有失败及 checkpoint 未覆盖。ARES-R 未修改，未操作实机。
本包不含现场点云、生产配置或现场派生图。ManiSkill 代码 Apache-2.0，资产存在
CC BY-NC 提示，商业使用及渲染成果使用范围需复核；不重新分发仿真资产。

实际执行源文件在每个 run 内，不能用当前 HEAD 代替历史执行版本。
用 `benchmark_build_evidence.py` 从 manifest/原始 episode/TensorBoard 文本日志重建图表；
用 `benchmark_package_delivery.py` 重建便携包与 SHA256。图文 PDF 已渲染检查，
LaTeX 经本机 pdflatex 编译；内置编译器因沙箱启动器缺失不可用，未伪称其编译成功。

新模型/日志体积有限且均属于本轮证据，保留；不为“清理”删除正式失败或 checkpoint。
LaTeX 可再生 aux/out 文件不属于正式证据，不推送。
