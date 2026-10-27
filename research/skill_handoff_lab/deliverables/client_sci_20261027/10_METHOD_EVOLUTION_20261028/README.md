# 方法演进：真实物理与学习证据（2026-10-28）

**先打开 `index.html`。** 全部图片、视频、PDF可离线打开；无需服务器。所有新机器人画面均为实际SAPIEN/PhysX运行，非实机、非理想状态动画。`HISTORICAL_`单独标记旧公开实验，不混入本轮成本或新训练结果。

## 当前结论

- **真实执行与训练：PASS。** 6个普通PPO最终模型，614,400次在线训练；所有新探针、训练、监测、评价、视频复跑共903,584次实际控制交互。没有永久后台GPU作业。
- **独立方法增量：NOT_ESTABLISHED。** 新64开发条件：名义60；Residual PPO（31/37/43）63/62/64；参数PPO 62/63/60；配对弃权60；固定参数与等采集成本CEM均62。参数减残差平均−2.08pp，描述性分组/种子配对95%区间[−6.25,+4.17]pp。三模型共用64例，不能当192独立场景。权限也不完全等价：Residual为每步XYZ+夹爪，参数PPO为每步连续XYZ，M1仅一次高度三选一；只有paired与absolute/对应消融在相同决策频率和候选集合下直接匹配。没有把接口差异报成算法收益。
- **后续物理可控性：PILOT_PASS。** Stack参数可改变实际工件与完整终态。Pyramid原生250步、同16初态：12/16、12/16、7/16；第一层−12mm包含1救回、1退化，+12mm有5退化。小样本设计证据，尚无Pyramid RL增益。进一步同16例及时松爪控制11/16（1救回、2退化），偏置变体9/16、9/16，亦未获得整体改善。
- **Peg强基线：NOT_READY。** 原生100步三轮跟踪/接触调整仍仅1/8；没有以0基线包装RL胜利，未开Peg RL训练。
- **封存测试：UNOPENED。** 当前是开发与设计探针；state-based输入含仿真真值。非privileged感知、多模态泛化、JAKA或实机迁移均未验证。ARES-R只读，未用受限现场资料。

## 审查顺序

1. 中文 `CLIENT_METHOD_UPDATE_CN.pdf` 第1页即本轮结论，后4页为证据和范围；英文两栏 `PAPER_METHOD_EVOLUTION.pdf/.tex`已同步。上一级01中文版和02论文入口亦更新，旧实验章节仍保留于Git历史。
2. `figures/Fig6a_native_results.png` 查看全策略/每seed终态；`Fig6b_paired_tradeoff.png`同时比较救回与退化。`CASE_PAIRED_CONFUSION.csv`保留每case，不只报告均值。
3. `media/PAIRED_stack_rescue_200003.mp4`、`PAIRED_stack_regression_200006.mp4`、`PAIRED_selector_regression_200020.mp4`查看同初态实录。参数43的总体为60/64：所选影片既有救回也有退化，不能据影片推成功率。
4. `media/PAIRED_pyramid_rescue_122007.mp4`与`PAIRED_pyramid_regression_122005.mp4`展示第一层后果如何改变下一层。左名义，右−12mm；均20Hz原时序，较短程序结束后仅为对比保持末帧，**没有补跑物理**。`PYRAMID_PAIRED_CASES.csv`、`Fig8_*`给出实际末态与几何。`Fig9_release_control`、`PYRAMID_RELEASE_CASES.csv`和`PYRAMID_HANDOFF_DRIFT.csv`检验及时松爪这一简单强控制解释，假设在运行前冻结。
5. `Fig5_residual/parameter`来自真实TensorBoard训练，`Fig4_*`来自工件/夹爪状态，不是期望TCP路径。`Fig7_peg_failure`保留失败接触误差。未在无Peg学习时伪造跨任务学习曲线。
6. `Fig0_historical_stack64`与`HISTORICAL_MANIFEST.json`只读复核旧58 vs61/62/60、5个共同rescues和seed43的3个regressions。**不用于拟合本轮gate**。旧两策略并集63/64只是那组策略的可见并集，不是所有方法上限。

## 证据位置与诚实范围

| 文件 | 来源/含义 |
|---|---|
| `BENCHMARK_PROTOCOL.json` | 训练前冻结Stack/Peg协议、split seed与SHA；不事后改写 |
| `NEXT_TASK_DECISION_PREPROBE.json` | 看到Stack负结果后、Pyramid探针前冻结的实际转向 |
| `ENVIRONMENT_AND_ISOLATION.json` | 单GPU0、无网络路由、无串口、ARES-R只读的实际运行回执 |
| `EXPERIMENT_RESULTS.csv` / `RESULTS_TABLE.tex` | 同64新开发条件，所有策略和每seed |
| `CONTROL_AUTHORITY_MAP.csv` | 合法参数、原生预算与实际完整结果；非RL贡献声明 |
| `ACTUAL_INTERACTION_COST.csv` | 配对分支、CEM、所有视频复跑均单独收费 |
| `MODEL_INDEX.json` / `ACTOR_UPDATE_AUDIT.json` | 权重SHA、实际Actor更新、真实物理执行loader证据 |
| `RAW_SOURCE_INDEX.json` / `RAW_TRAJECTORY_INDEX.json` | 非Git原始结果与逐轨迹文件的精确来源、SHA |
| `FIGURE_MANIFEST.json` / `HISTORICAL_MANIFEST.json` | 每张图、视频的来源，历史与新物理明确区分 |
| `repro/frozen_r1/` | 实际训练时源码；后续改动只有更准确的日志与采集选项 |
| `SCI_CLAIM_AND_NEAREST_NEIGHBORS.md` | 原论文/代码核对与新颖性边界；未复现方法不装成已复现 |
| `QA.json` / `REVIEW.md` | 完整性、数值、原生预算、字体/逐页视觉审查 |
| `RESEARCH_BUNDLE.json` | 非Git便携包路径、SHA与解包方式，含实际模型/轨迹 |

全量原始轨迹与模型在本机 `runs/method_evolution_20261028`，公开Git仅含允许的模拟素材、源码、表格和索引。公开Panda任务不能替代JAKA几何/夹爪commissioning；没有把ARES-R离线成果算成本方法的硬件证据。

## 为什么没有扩五种子或打开test

本轮已经完成普通PPO与参数PPO、差分/绝对结果选择及消融、等成本CEM和三任务实际探针。新机制未稳定超过普通残差与固定调参。继续在饱和Stack刷样本或用封存测试调选择器会削弱可信度。下一轮应在**新冻结的Pyramid条件**比较同权限规则/PPO/绝对价值/配对差分，并补齐同候选、同边界决策频率的RL比较；先确认先前效果在当前几何之外有决策价值；没有独立收益就继续改具体物理控制，不能提前命名算法或扩大论文主张。
