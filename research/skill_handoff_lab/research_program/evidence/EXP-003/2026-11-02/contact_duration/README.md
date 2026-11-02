# EXP-003｜接触、技能时长与真实执行结果：累积研究闭环

2026-11-02；基于4dcb50d。**UNVERIFIED_INDEPENDENT_ALGORITHM_GAIN**。

先打开 [REVIEW.html](REVIEW.html)，或 [完整同初态视频与逐帧曲线](video/SYNC_VIEWER.html)。三张图分别保留PNG/PDF；[SUMMARY](analysis/SUMMARY.json)、[模型预测与实际结果](analysis/PREDICTIONS.json)、[模型及真实训练记录](models/TRAINING.json)、[全部原始轨迹与源码归档SHA](MANIFEST.json)。没有实机、ARES-R写入、现场资料或密封测试。

## Question / Prior explanation

合法的抓位和后继执行调度，是否通过真实SkillResult影响接触与任务结果？分段重规划最初可能由“修正附着漂移”获益；竞争解释为：额外时间、末端到位、规划终点误差、不同接触路径。依次做了同前段物理的干预、冻结信息、减速、等时等待、只改变screw积分分辨率、只读接触/FK检查，没有预先发明新网络。

## 实际结果

| 实验 | 结果 | 结论范围 |
|---|---:|---|
| 历史Pyramid576复算 | 前段敏感9/64，后段13/64，重叠6；联合案例420002/420023 | 原始轨迹诊断，非新实验 |
| 还原两案例原始初态：legacy / 四姿态后继规划 | 7/18 / 7/18，1救回1退化 | 仅这两旧初态；保留此前新设计36/36能力，不相互替代 |
| 工具12条件：baseline / measured refresh / frozen refresh | 3/12 / 5/12 / 5/12 | 2救回0退化；更新附着信息不是必要因素 |
| 新设计108条：baseline / frozen / slow | 7/36 / 7/36 / 11/36 | 6初态×2质量条件×3抓位×3调度；不是108独立样本 |
| 同108条有限网格：最佳固定 / 逐条件已测oracle | 6/12 / 7/12 | 仅1/12额外选择空间；不是连续动作理论上限 |
| 新开发实际执行：固定 / 普通两边界Fitted Q / 监督 / STAP-open / STAP-replan / 后继价值类比 | 均2/4 | 2初态×2质量条件；所有学习方法对固定均0救回0退化 |
| 几何最短路径选抓位，三种调度 | 均0/4 | 不能把它当唯一强对照；继承100mm参考也为2/4 |
| 同前段细查：frozen / settle9 | 2/2 / 0/2 | 额外等时等待不充分；这是1场景两个抓位 |
| TRANSFER screw步长0.02 / 0.005 | 各1/2、成功抓位相反 | 更小规划误差不保证更好接触，不宣布通用控制增益 |

## 为什么这些信息值得保留

1. **Pyramid的失败并非全部是放置yaw问题。** 历史420002大量C_NOT_GRASPED；420023也有A_BOUNDARY_NOT_VERIFIED、抬起丢失与两局部合同成功但native失败。抓C时目标跟踪误差约35.7–38.7mm、C曾移动3.4–45.8mm；只能据此提出接触/跟踪解释，不能直接证明原因。对新的n2初态做C中心单因素刷新，仍7/18；没有把此确定性修改包装成算法。
2. **相同seed不是同一物理初态。** 原数据n16、新探针n2，ManiSkill随机初始化随batch改变。先产生的54条保留为初始化诊断；另36条只在第一物理动作之前恢复原始记录状态并验证读回误差<1e-5，之后完全真实物理执行，无技能边界状态重置。标记RECORDED_INITIAL_STATE_COUNTERFACTUAL_PHYSICS_NOT_RNG_REPLAY，不能称逐轨迹复现旧n16动力学。
3. **排除了一个诱人的反馈故事。** 35mm案例在首次桌面/方块接触前的相对工具旋转<0.5°；大约12°变化发生在接触以后。冻结初始附着估计也救回。6个只读仪表对照的全部100步动作和原有测量逐项相同；36个原始干预的PICK前缀精确相同。
4. **具体软件/几何混淆已找到。** MPLib plan_screw返回Success时，实际规划终点FK误差仍13.48–14.22mm；分段规划降到0.96–1.42mm。原路径等待9步仍失败，说明不能仅解释为物理未停稳。细积分使误差进一步变小，但成功抓位交叉，说明规划精度、实际跟踪和接触路径仍是不同变量。细积分保留原有失败后规划fallback；不是禁用fallback的纯解析求解器对照，后续需明确记录每次solver分支。
5. **真实训练没有增益。** 72训练条（4场景）、36校准条（2场景），11000次实际优化更新，单初始化seed31132；模型保留。Q1校准Brier=0.424865，差于常数0.5的0.25。继续在两个新开发场景实际执行36条，各学习方案均与固定2/4持平；没有以训练拟合或几何弱基线作独立算法证据。

## 动作、观测、合同与公平范围

ARES-ToolLoad-v0为公开PullCubeTool-v1的工具密度0.75/1.25变体，保留官方cube/base XY<0.6m与100step判据。PICK_TOOL必须抓住、抬起、满足实测稳定边界，然后USE_TOOL；实际SkillResult含相对位姿与剩余步数。三抓位35/70/105mm，第二边界选择原TRANSFER、固定附着分段重规划或降低TRANSFER加速度；不在已饱和yaw/旧hook网格上训练。所有模型同45维当前观测、相同动作接口、原生限制及100步物理预算。质量系数为共同已知任务条件，不声称隐藏状态辨识。几何后继搜索、native步长1、不压缩轨迹、速度1.8、加速度6（slow单因素3）、稳定最多10步全部继承。

STAP式比较为继承技能Q×预测后继Q的稳定log域枚举9组合，非重新声称完整官方benchmark复现；普通Fitted Q使用实际后继状态上的max-Q自举；监督使用终态成功；Foresight类比只学固定baseline后继，不等同原论文代码。均单种子小数据诊断，非充分收敛SAC/PPO结论；继承已有普通SAC，不覆盖旧模型。共有物理预算，但离线计算/标签利用不同：Q0 1000、Q1 2000、动力学2000、ordinary2000、supervised2000、foresight2000更新；总11000，不能把共享模型总成本重复算到各自或假称等训练成本。实际时长、规划次数在原轨迹/MECHANISM。训练时钟约7.16秒，不含物理数据收集与环境初始化。108条数据收集计时372.96秒；36条新开发执行125.38秒（逐episode计时之和，均不含环境构造）。

原始失败分类另见[FAILURE_CLASSIFICATION](analysis/FAILURE_CLASSIFICATION.json)：36条记录初态Pyramid中14成功、14抓C失败、6前段合同失败、2无控制异常但native失败；108条工具中25成功、67后继执行完但native失败、12预算删失、4后继规划失败。只读FK检查也以6对全部100步测量完全一致验证未扰动物理。

## 最近邻与原创性边界

- [SeqDex](https://proceedings.mlr.press/v229/chen23e.html)：技能转换可行性/反向微调；前段影响后段本身不是创新。
- [ViSkill](https://github.com/med-air/ViSkill)：价值引导技能衔接；后继Q/合同监督不能直接声称新方法。
- [TASP](https://benned-h.github.io/tasp/)：运动连接的head/tail构造，要求认真保留完整几何路径对照。
- [SPIRE](https://proceedings.mlr.press/v270/zhou25a.html)：TAMP分解与学习协作；确定性接口修复不计RL增益。
- [Grasp to Act代码](https://github.com/hgupt3/g2a)，pin4b4a4298f6eec3ba6220427d5c863c24cd2529d0：实际调用relative-palm误差与不可逆StabilityAccumulator，只作被动Panda轨迹分析，15mm/5°，没有复现LEAP12方向外加wrench实验。未发现顶层代码许可证，不转存其源码/权重/资产；本地只读分析依赖明确固定源码。我们的判据仍原生任务成功。

## 预算、软件错误与可复现性

共320个执行尝试、45500物理控制步。其中312个有效研究episode（90Pyramid+222Tool）、44700步；另8个FK记录器API错误尝试仅做terminal hold800步，明确INVALID，不作任务失败或阴性科学结果。修正后重跑并保存旧日志。生成视频为0新物理步，101帧/20fps，读回误差1.19e-7。全部模型、轨迹、100Hz接触、源码快照、错误及开销保留。没有多随机种子或封测结果。

`MANIFEST.json`每个tar成员SHA均已解包复算。ROOT原始路径见manifest；解压到统一ROOT后，可用其中SOURCE路径定位。分析脚本当前读取RESULTS内绝对路径，迁移机器需只重定位path字段、保持sha不变（不可更改物理数据）。

复现顺序：`program_tool_effect_probe.py` → `program_effect_value.py` → `program_effect_evaluate.py`；因果探针`program_tool_all_contacts.py`和`program_tool_endpoint_probe.py`；报告`program_effect_mechanism.py`、`program_effect_report.py`、`program_effect_predictions.py`；视频`program_effect_video.py`。均在tools目录，物理/渲染使用handoff_offline.sh隔离。具体参数见REPRODUCE.sh。未保存当时初始模型文件，另按相同构造函数+种子精确重建INITIAL_RECONSTRUCTED.pt并明确标记；终权重和完整loss曲线为真实训练产物。

## 永久保留 / 下一项有辨识力的实验

保留旧36/36几何能力、全部SAC/STAP/log-Q、SkillResult合同；新增可复用只读接触/FK证据、实际两边界调度接口、完整物理来源审查。**不将frozen12默认推广为更强控制器**（有预算失败）；不将dense005推广（有退化）；不叠加新辅助网络。

下一轮先在少量新训练设计初态把“计划终点FK、实际到位、接触开始、剩余时长”作为显式可执行合同的四个分量，共享给所有方法。记录screw/fallback来源，比较同预算的反馈到位+contact入口和冻结调度。观察在统一高质量完整控制后抓位×后继调度是否仍有稳定交叉排序；若有，再训练普通两边界Fitted Q/SAC及单一持续时间/可执行性增量，按独立场景配对与STAP/监督/强控制比较。当前没有独立方法收益；不把本轮数据当新封测、不在这两个救回案例上宣称泛化。
