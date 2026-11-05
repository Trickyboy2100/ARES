# 已有证据总账（只记来源支持的结果，附局限）

**研究快照截至基座提交 `c013f7e33fe65c3a76c3ef2dfc1dedc6d4816a67`。**
这里不是训练任务队列。新实验后可追加完整、准确的链接并更新状态。
所有数值使用自己对应的seed/case/success/horizon；不同任务或控制器不相减。

| 研究集 / 来源 | 数值与范围 | 支持什么 / 不支持什么 |
|---|---|---|
| [旧 JAKA PhysX v6 Geometry](../deliverables/client_sci_20261027/05_METHOD_AND_EXPERIMENT/TRAINING_DECISION.md) | 反馈8/32，普通Residual三种子13/18/23；P2分层反馈5/16、RL6/6/7 | 说明普通RL在自建未见几何有初步帮助；主要来自P1，不是新机制 |
| [R3执行资格似然](../deliverables/client_sci_20261027/08_ITERATIVE_RESEARCH/R3_RESEARCH_DECISION.md) | 强反馈26/32，普通PPO22/25/20，候选24/24/18 | Actor改变≠稳定增益 |
| [公开Stack新64](../deliverables/client_sci_20261027/10_METHOD_EVOLUTION_20261028/RESULTS.json) | 名义60；Residual63/62/64；参数PPO62/63/60；paired60；固定/CEM62 | PPO基础有正结果；动作权限不同、develop已看，不能宣称新算法 |
| [Pyramid机会与闭环](../deliverables/client_sci_20261027/11_PAPER_FIRST_20261028/OPPORTUNITY_SUMMARY.json) | 64设计名义52、固定59、九组合oracle62；64开发普通PPO51/53/56、风险候选51/54/55 | 前后技能参数有真实物理作用；风险候选均值+0，oracle不是policy |
| [Pyramid价值/接触诊断](../deliverables/client_sci_20261027/12_SCI_OPPORTUNITY_20261029/README.md) | 576物理试验，前段失败33、后段未完成36；风险13/735；开发名义/监督/FQ/合同30/32 | 旧任务学习空间不足、风险标签稀疏；数值非独立场景数 |
| [PullCubeTool](../deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029/README.md) | 官方0/4→增强3/4；另一新8例几何前瞻7/8 | 强几何/执行器改进大；目前不值得刷RL，不是新方法增益 |
| [PlugCharger](../deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029/README.md) | 12设计场景×9参数，固定5/12、有限成功并集11/12（分支有波动）；新8开发例固定2、几何2、监督3、FQ1、margin-FQ1 | 局部动作可改变终态，离线价值学习未胜；独立RL新机制UNVERIFIED |
| [Charger真实配对案例](../deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029/metrics/VIDEO_CASES.json) | 同抬起后态，50mm与75mm接近：误差22.05→3.66mm，失败→成功 | 证明某个参数在某一案例有效；非总体因果/泛化收益 |

## 原则

- 既有 `53/57` 等条件成功与 `53/64` 全任务成功，不能混成同一指标。
- 12训练×9分支是**108条执行**，不是108独立任务初态；同样如此，
  物理tick、RL有效决策数、PPO更新次数各自分账。
- c013中“388回放/56,400完整记录物理步”不是388独立训练场景，也不是
  新的RL在线训练预算；三模型是监督/Fitted Q训练。
- 所谓“未见开发集”在结果被查看后已经用于研究决策，**之后不再是密封测试**。
  独立 sealed split 必须在方法冻结后才运行。
- Pose replay/渲染不能称新PhysX训练；实际成功来自源运行/独立审核。
- 新方法已证实独立性能增益：**否**。不能将学术工作标题写成已完成创新。

## 新证据应如何添加

追加一行：`experiment id | source snapshot SHA | scene group/split | native task &
horizon | fair baselines & action/obs | seeds/cases & metric | outcome |
uncertainty & limitation | exact raw path/hash | science decision`。
更正旧条目要保留更正依据，不抹去真实负结果。

## EXP-001｜2026-10-31 原生强控制与重复性

[完整精简证据与源码/原始SHA索引](evidence/EXP-001/2026-10-31/README.md)：原生PlugCharger200，14初态、74完整执行+105步已记录异常，合计14905控制步、0优化步。18跨进程配对成败一致、17记录状态一致；设计4例原积分+保持1、有界反馈+保持4；冻结新8例1→5，5救回/1退化。事后局部轴向保持修正4例2→4，不是新确认集。保存真实非Git轨迹与一组标注的姿态回放视频；无现场数据。支持控制器修复，不支持独立RL创新；科学决定D-006，活动实验仍EXP-001。

## EXP-001 independent confirmation and visual mechanism, 2026-10-31
[Source-backed evidence](evidence/EXP-001/confirmation-v2-2026-10-31/README.md): 24 cold-process episodes on 8 new scenes, 2/8 integral vs 7/8 original servo vs 7/8 local-depth servo. Four additional inspected grasp-angle diagnostics all fail. 5600 actual steps, zero optimizer updates. Protocol RAW/ANNOTATED/TRACE/CAPTION generated for two historical diagnostic cases; replay adds zero physics. No new RL claim. EXP-002 opportunity not yet established beyond the strong suffix.

| [EXP-001独立确认55861de](evidence/EXP-001/confirmation-v2-2026-10-31/README.md) | 新8初态积分2/8、servo7/8、depth-hold7/8；7/7进入后段成功；同成功7例首次到位平均快18.71步 | 当前插入后段确定性控制饱和；方法RL更新0。另25/5等历史样本不可拼为训练新证据。 |

## 2026-10-31｜EXP-002真实双轨探索、在线SAC与反馈消融

[主证据](evidence/EXP-002/2026-10-31/README.md) · [逐例SHA](evidence/EXP-002/2026-10-31/analysis/EPISODES.json) · [成本](evidence/EXP-002/2026-10-31/analysis/SUMMARY.json) · [成对比较](evidence/EXP-002/2026-10-31/analysis/EVAL_COMPARISON.json) · [模型](evidence/EXP-002/2026-10-31/models/INDEX.json)。

原生新4scene3/4；自定义质量/摩擦12/12由强控制成功。观测偏差新4scene×3条件：geometry4、RLS6、search10、SAC1、effect SAC3、prior-only3、同SAC零Z7，分母均12。零Z六救回零退化；effect/prior-only逐例完全同成败。所有当前比较的记录前段边界相同，无前段失败；不是不可见PhysX内部状态等同证明。

真实训练包含普通SAC、效果特征SAC、既有critic-LN对照；每轮1024技能决策、896次actor/critic更新，训练物理步/尾段/原生全任务成本分别记录。LN alpha=.1协议偏差与补齐的alpha=.02分开报告，实际结果从模型训练manifest解析。崩溃旧run的四episode800步也计入，重复恢复文件不重复计数。完整累积成本以SUMMARY为准，不把重渲染或CPU分析计为物理实验。

方法收益：UNVERIFIED。没有复现SCOUT/DAWN/STAP论文成绩；最近邻源码/许可已核对，LayerNorm不是新方法。完整视频与英文圈注为实测姿态重播，保留源轨迹、投影、帧PTS和浏览器同步QA。下一项EXP-003，重点参数权限和强RL学习能力，不以扩大弱SAC或小样本反馈特征网络代替机制研究。

最终补齐对照：critic-LN alpha0.02普通SAC10/12，alpha0.1偏差组4/12；同设置effect+LN2/12，prior-only2/12。共五组真实训练，每组1024技能决策、896次actor/critic更新。结果仍不支持候选特征的独立算法收益。

## EXP-003｜2026-11-01 实际执行追加

[结果与精确来源索引](evidence/EXP-003/2026-11-01/README.md)：A三种普通LN-SAC各4096决策、3840优化更新；完成合同/全权限与投影权限/强搜索/真实checkpoint对照，6开发初态、单训练seed。B三任务77条原生物理执行、9200步、14初态种子；强完整后段TwoRobot8/8与同前段正交2/4，无RL更新。全部源快照、模型、回放缓冲、原始轨迹、SHA清单及三组完整实测状态回放视频保存。动态认证未证明，无实机/密封测试/独立算法增益。成本详见A SUMMARY与B_RESEARCH_SYNTHESIS，不能将多个参数条件或视频帧视为独立样本。

- 2026-11-01 STAP follow-up：[224条真实物理、5600优化、算法核12组、严格同前段后验退化与强预算控制](evidence/EXP-003/2026-11-01/stap_followup/README.md)。原始包/模型/源码/SHA/全程双视频齐备；无新RL增益。


## 2026-11-02｜真实SkillResult配对、数值修复与最小机制否证

[结果/关键图/同步视频](evidence/EXP-003/2026-11-02/skillresult/README.md)。保留历史，新增320真实物理回合/32000控制步、13600优化更新。96干预面板固定8/16、事后上限9/16，无严格目标排序反转。复算三例退化，区分低Q强制改参、预算与双成功合同。发现概率饱和选参并列，保留112诊断并以logits/log-Q另跑112确认。修正后ordinary1/8，generic/contract/relative/relative+contract均2/8；固定、强几何、STAP-open和3critic-UQ均3/8。组合机制没有独立增益；2个新测试初态已作开发诊断，未使用743052/53。公开模型/真实数据/四个完整视频及同帧曲线已归档。下一步优先原生多技能后继合同冲突的物理机会，不扩大窄hook网格或网络模块；保持UNVERIFIED。

## 2026-11-02｜96条原始干预的独立上下文分层（既有数据复算）
来源：evidence/EXP-003/2026-11-02/skillresult/EPISODES.csv中group=train96。
按(seed,mass,target)的16组，每组6个(grasp,hook)执行结果；成功数量分布0/6×7、3/6×1、4/6×1、6/6×7，总成功49/96。两个混合条件：743041,mass1.25,target+0.08m时grasp90mm 3/3、100mm 0/3；743046,mass0.75,target+0.08m时hook−55mm 0/2、−65mm 2/2、−75mm 2/2。其余14组所测参数的完整成败全相同。强固定8/16、已测有限动作事后oracle9/16；未显示按正负目标反转的最优参数排序。8个物理初态各2目标，质量在初态之间平衡，不能当作16个独立场景或同场景质量实验。该分析只描述所测参数对终局成功标签的辨识度，不证明物理轨迹不变或合同学习普遍无效；新增物理0、优化0。见DECISIONS D-013。

这一小网格的混合成败分别由抓点主效应和hook主效应解释；未观察到成功标签上的抓点×hook交互项或随目标变化的明确最佳参数反转。二值成功与有限2×3动作网格的分辨率不足以排除真实连续物理交互。

可重复计算：python research/skill_handoff_lab/research_program/scripts/audit_skillresult_opportunity.py （只读取已提交CSV，包含96条、7/7/2分组、固定/已测oracle、阶段参数二值交互和跨目标反转的断言；新增仿真0）。


## 2026-11-02｜累计物理探索：后继朝向差异被强几何解释

[关键图、同初态完整视频与原始证据](evidence/EXP-003/2026-11-02/pyramid_contract/README.md)。先执行新增机会脚本，并核对96原始工具轨迹、576旧Pyramid终态及26历史轨迹SHA。新增90真实回合/22500控制步、0优化：单代表朝向pilot0/18；最近对称姿态29/36；四等价姿态预放置/放置可行性检查36/36。后两者36组前段记录完全相同，7救回0退化，属于确定性规划改进。强控制下2初态×2真实顶块朝向合同均9/9参数成功，固定与已测上限均4/4，无排序反转。保留SkillResult、对称可行性组件、原有RL/STAP与负结果；不在饱和条件训练。后续仅在真实布局/后继净空出现强规划后的不同物理参数偏好时训练。原生成功/250步不变，追加朝向合同另列adaptation variant；stride3及未建world-cloud碰撞约束明确披露。独立方法增益仍UNVERIFIED；密封测试未动。

## 2026-11-02｜历史576真实双阶段敏感案例与最新强几何的交叉复核

**本项为读取已提交原始统计JSON和CSV的离线再分析，不新增物理或优化。**
[完整理论复核及下一轮](evidence/EXP-003/2026-11-02/pyramid_contract/THEORY_REVIEW_NEXT_RESEARCH_LOOP.md)；
[标准库可复算代码](scripts/audit_cumulative_skill_opportunities.py)（本次仅提交代码，尚需Codex在真实仓库路径运行核对）。

最新Pyramid `SUMMARY.json`：pilot0/18（只1 scene，与后两组不能当严格配对）；nearest symmetry29/36；4个可行对称姿态36/36，后两者36同前段、7救回0退化；feasible版本实际记录144 candidate queries。各自四个(scene,yaw)上下文，feasible全部9/9；2独立物理scene，没有RL训练。

历史Pyramid `HISTORICAL_OPPORTUNITY.json`：576条/64scene，名义52/64、强固定59/64、9动作后验成功并集62/64；前段有效参数改变后继成败的scene数9，后段同起点更改动作敏感的scene数13，二者重合6（420010、420022、420026、420036、420038、420052）。仅相对名义A/C单侧改动均无法救回而联合参数改变可成功的已测scene是420002和420023；既有开发样本不能用于新方法泛化成绩。工具train96有14/16二值终局不随六合法动作变化，但同上下文连续target距离可变化，如743043/+80mm约229–348mm，不能等同物理效果恒定。后验STAP Q退化及TwoRobot预算假阴性仍为具体OPEN_MECHANISM线索，尚未形成原创增益。

## 2026-11-02｜接触/时长累积闭环（contact_duration）

完成312有效物理episode/44700步，另8个FK日志API错误尝试800步单独INVALID保留；11000真实优化更新。旧Pyramid576审计与原初态counterfactual厘清C抓持/前段合同失败，原36/36几何资产保留。工具干预baseline3/12→measured/frozen refresh均5/12；严格前段一致36/36，反馈信息非必要。等时等待不充分，plan_screw Success仍有13–14mm终点FK误差；细积分下降但成功抓位交叉，不能宣称通用修复。108新设计有限网格最佳固定6/12、oracle7/12；新开发普通FittedQ、监督、STAP-open/replan、后续价值类比及固定均2/4，继承100mm参考也2/4。没有独立算法增益，UNVERIFIED。

永久保留：旧SAC/STAP/log-Q/36of36、两边界真实结果、只读100Hz接触与FK检查。下一轮先共享计划终点/实测到位/接触入口/预算合同，并用新设计物理配对验证抓位×后继调度残差，不扩训练量或叠模块。初始n2与原n16同seed不相同，54诊断条不可当旧case复现；另36条仅reset恢复记录初态，之后真实物理。详见[研究结论](evidence/EXP-003/2026-11-02/contact_duration/README.md)、[审查页](evidence/EXP-003/2026-11-02/contact_duration/REVIEW.html)。


## 2026-11-02｜共享执行修复与真实双边界 SAC（contract_sac）

[研究结论](evidence/EXP-003/2026-11-02/contract_sac/README.md) · [审查页/完整同初态视频](evidence/EXP-003/2026-11-02/contract_sac/REVIEW.html)。596完整真实物理回合/59200控制步，另76已保存部分步及首次崩溃未知0–100步。FK计划误差降低不等于实际到位；stroke2×2证明3/8→6/8由合理行程权限解释，无需后验反馈；实际起点微重规划6/8→6/8。接触前重载抓持漂移与短期稳定不矛盾，统一100Hz力/20Hz姿态定义。

恢复真实两边界SMDP SAC：保留575已完成决策、1022实测更新、SAC256/512及中断模型/缓存；645条转移物理审计误差0。第一次67回合TOPP崩溃保留原数据，不能假称恢复丢失actor。相同323回合数据训练模型/监督/后继价值10000更新。新8开发初态，固定70/105、SAC256/512、监督、STAP-open/replan、后继价值全部7/8。共同失败746204为TRANSFER规划失败，STAP概率约0.00019；不是高Q误导。1训练seed与有限检查点，不宣称收敛/已充分RL。原有A线、Pyramid36/36、STAP/log-Q保持。

36高度网格无额外后继二值机会；36加速度配对无救回且一次预算退化；拓展yaw一次退化；翻转夹爪未救回两个共同失败。原生PokeCube50步标准/直接接近各0/4，已实际追查预算与抓取，不能当弱基线制造RL收益。下一具体问题是强规划下剩余可达路径/中间工具姿态与实际跟踪/接触预算；第二任务先建立可信共享控制器。所有正负模型/物理/成本/SHA公开仿真归档，密封数据不动。独立算法增益 **UNVERIFIED**。


## 2026-11-02｜post_contact：真实动作作用与效果方向SAC负结果

[研究结论](evidence/EXP-003/2026-11-02/post_contact/README.md) · [完整审查页](evidence/EXP-003/2026-11-02/post_contact/REVIEW.html)。本轮858完整原生物理回合/85800控制步，另173部分步；不是858独立场景。SAC3186梯度迭代（actor/critic各一次），比较器1000联合优化。原高度审计修正为11个已选择上下文、8个实际HOOK/PULL上下文，原9的表述被原始记录推翻；高度仍无完整成败增量。746204仅3次CPU规划/IK诊断，0新物理。

继承强控制后执行四类后接触参数干预、普通SAC诊断失败同前缀追查、120条新设计物理面板与两轮时机干预。新面板14/24抓持上下文达到PULL修正边界，3个同前缀救回，但均已由另一名义抓持覆盖；最佳名义抓持与联合oracle均8/12。22个被审计状态有14个更早接触，PULL决策延迟中位19.5步。首接触与PULL使用同一个“修正后重接原航点”原语时可有方向排序变化，零动作8/8完整轨迹保持；不能将不同延续语义或时钟本身误当机制。

实际训练冻结效果方向映射+SMDP SAC，保留普通同权限、全局方向、通用效果辅助、监督评分与STAP功能核。物理尺度特征使训练场景CV效果误差0.02530→0.001301（固定效果0.003740），但最佳方向轴仅18/22→19/22。共同500更新：普通SAC/STAP6/8，方向候选/全局/通用辅助5/8。各再256实际决策后普通与候选均6/8，固定87.5mm也6/8；固定同一物理前缀三者均6/8，候选0救回0退化。普通/候选实际在线成本14170/14275步，单训练seed，不能声称完全等物理成本或充分收敛。两个576转移缓存与早期128缓存均由真实物理复算通过。效果映射Jacobian成立于共同no-op投影之前，不声称已证明执行动作原子分布的微分熵。

**决策：** 保留普通训练器、实际事件/前缀验证和所有正负权重；不将预测准确率、方向重参数化或共同控制进展认作独立RL方法。不扩当前yaw训练。下一项有价值实验必须建立最强抓持/控制后仍存在的可执行恢复收益，重点区分工具倾斜下的无效接触与真实钩入；若共享控制已覆盖，则以已有PlugCharger200成熟线研究实际抓持结果/接触条件，不降低基线、不动封存。方法继续 **UNVERIFIED**。


## 2026-11-02 — Recovery and native charger two-boundary learning

Actual continuation from 525d67e: 354 complete physical episodes / 64,400 steps plus 207 saved partial steps. Tool recovery: all six controls 6/8; early interventions all 0/4. Native charger strict-prefix factorial separates angular collision from advance/budget effects. Corrected whole-task-success SAC: 1,024 transitions / 1,792 updates, checkpoints and reconstructed buffer retained. New development: nominal and force guard 8/8, SAC256 3/8, SAC1024 7/8, same nominal grasp + SAC successor 8/8. Named mass/friction variant also covered by 15-degree nominal (8/8 conditions, two scenes). No independent algorithm gain; UNVERIFIED. Retain instrumentation and training objective correction, freeze covered recovery/pace panels. Next: genuine successor-contract constraints with residual failure after strongest shared control, not manufactured weak grasp or more old-coordinate training. [Full evidence, negative runs, videos, costs and SHA](evidence/EXP-003/2026-11-02/recovery/README.md).


## 2026-11-03 — Parallel native Peg physics and trainable skill-value mechanism

[Complete evidence and code](evidence/EXP-004/2026-11-03/peg_value/README.md). Active EXP-004 now runs alongside EXP-003 common execution. Official GPU Peg OBB helper used stale CPU world pose; preserve literal baseline and adapt local collision geometry to actual GPU pose. Corrected official timing1/6, explicit shared retiming3/6. Frozen-control two-boundary panel:60 trials /10 natural scenes, strongest fixed6/10, measured union8/10; same-prefix rescues and regressions are real. Prototype: paired return-difference loss, same-data FQI/absolute/suffix-value/STAP-style enumeration, discordant sampling ablation, then ordinary masked LN-SAC with paired/absolute/shuffled critic gradients. Historical early closed-selection40/69 vs ordinary32/69 largely matched generic labels; later43/69 vs44/69. Fresh actual Tool all3/6; Peg all2/4. Independent algorithm gain UNVERIFIED. Retain actual models, all failed trials, source/hash/buffer evidence and exact-frame video. Costs:160 complete100-step episodes including6 invalid instrumentation terminal holds;75600 joint value updates and51600 actor/critic updates EACH. Native horizons and success unchanged; no sealed data or ARES-R edits. Next causal question is executable contact tracking/budget under the shared Peg controller; reconstructed target is correctly +50mm along goal x, so do not re-debug a presumed reversed frame or merely train another grasp score.


## 2026-11-03 — finite physical Options, acquisition revisions and extraction mechanism

Source: [evidence/EXP-004/2026-11-03/discrete_handoff/README.md](evidence/EXP-004/2026-11-03/discrete_handoff/README.md); code `2ac433e`, base `1ea0965`. Independent algorithm gain remains **UNVERIFIED**.

- V0 continuous projected SAC has off-support Actor/Critic targets. V1 implements exact legal finite options, real SkillResult, duration, termination and remaining budget; DSAC/FQI/IQL/CQL/supervised/functional-STAP fixtures and physical trace checks pass. This is mature algorithm infrastructure, not originality.
- Actual offline→online study uses compatible60-episode Peg prior, fresh natural training scenes, two optimizer seeds and native100.38 model histories exactly reconstruct from their actual physical buffers.2458 full episodes/245800 steps include all training/evaluation/posthoc confirmation;250400 primary critic optimizer iterations exclude audit reconstruction and toy fixtures.192 pairs reexecute and charge both full prefixes.
- Expected-update acquisition was implemented, physically trained, then revised for wasted duplicate coverage, correlated suffix outcomes and current-task anchor distribution. Joint outcome prediction improves old-scene leave-one-scene-out counterpart Brier0.313→0.182; covariance changes7/16 actual acquisition rounds, but does not yield a reliable policy gain.
- Eight predeclared fresh development scenes: original joint and EES-style4/8 for both seeds; current-anchor revision2/8 and4/8;128-episode ordinary SAC5/8 and3/8. Same-history mature supervised/FQI5/8 for both. Best fixed6/8 is a hindsight comparator, not a training-selected policy. Across14 development scenes, fixed maximum10/14 versus finite measured union12/14 preserves a conditional-adaptation opportunity.
- A further actual intervention changes only extraction of frozen Q values: current-joint seed31301 Actor2→minQ4→meanQ5; mature seed31302 Actor3→minQ2→meanQ5.48 complete native rollouts confirm it. Mean-Q rescues3 and2 respectively with0 regressions; min-Q causes1 regression in mature seed31302. Actor approximation and clipped-Q aggregation are distinct. This is explicitly posthoc development diagnosis, not held-out performance or an independent RL contribution.
- The acquisition surrogate scores greedy decisions of one critic while deployment uses a learned Actor. Positive scores do not establish actual return gain; current-anchor seed31301 has observed update returns4→4→3→4→4→3→3→3→2 on the retrospective real branch panel. Retain this falsification and align any future mechanism with the actually deployed policy before expanding it.
- Permanent assets: finite Option runtime/learners; compatibility filter; exact optimizer/prefix audits; complete real branch evaluation; actor/min/mean diagnostic; trained checkpoints and RAW/English videos with frame-synchronized curves. EES and STAP comparisons are functional adaptations with disclosed limits, not claims of full original benchmark reproduction.
- No sealed data, hardware, ARES-R modification or restricted site data. New acquisition variants were frozen before reading expansion outcomes; later extraction confirmation is labeled posthoc. Next work is shared-baseline policy/value consistency and genuinely new development validation, not relabeling a standard extraction fix as novelty.


## 2026-11-04 — Natural coverage, value improvement and measured handoff targets

[Complete evidence](evidence/EXP-004/2026-11-04/value_improvement/README.md). From60760b8, actual96 new natural initial states and822 native100 episodes/82200 steps;374400 primary critic iterations,268 checkpoints. Shared mature finite VIAC/USAC/BEE/IQL/FQI/CQL/STAP/conditional-scoring controls; no renaming mean-Q or empirical DP as novelty. Original3x2 training fixed24/32 vs union25/32; physical split/refreshed option adds one reproducible joint rescue, giving26/32 but no development union increase: both21/32 vs hindsight fixed19/32. All positives, regressions and exact prefixes retained.

Candidate substitutes physically measured suffix values at exact SkillResult into predecessor backups; complete-coverage revision changes only activation on full legal successor support. Offline3-option original14/32,13/32; revision15/32,17/32. Six true32-episode online runs: revision19/32,19/32; ordinary mean-critic SAC19/32,19/32; exact value improvement18/32,19/32. Candidate vs SAC1 rescue/1 regression, then0/0.48 additional actual final-policy rollouts exactly match their complete100-step physical branches; other32-scene model scores are explicitly logged physical-branch evaluation. This is catch-up, not independent gain.

New diagnosis: legal newly introduced actions lack historical execution support. Ordinary SAC terminal new-action Q exceeds1 on7/29 distinct old handoffs per seed; CQL removes this audited overflow. Same-data FQI15/32,18/32 versus CQL18/32,18/32, still not an independent method. IQL18/32,19/32. Neither calibration nor an isolated control rescue proves a paper contribution. Development now seen; complete-coverage/CQL are disclosed iterative revisions, not sealed validation.

RETAIN strong shared controller, explicit action-schema compatibility, finite RL controls, all real data/weights and shared split primitive. LOCAL_NEGATIVE measured-backup/coverage revision. Next investigate a physically legal broader predecessor parameter domain with explicit grip margins and the nominal option retained; require opportunity beyond strongest fixed/search control, keep conservative/support-aware baselines, and do not expand this failed backup unchanged. Method status remains UNVERIFIED. ARES-R/read-only and hardware/network isolation preserved.


## 2026-11-04 — Closed-loop feedback and budget-conditioned joint training

[Full evidence](evidence/EXP-004/2026-11-04/closed_loop/README.md) · [Review](evidence/EXP-004/2026-11-04/closed_loop/REVIEW.html) · [Metrics](evidence/EXP-004/2026-11-04/closed_loop/RESULTS_SUMMARY.json). Native100 physical curves, actual8-action successor execution, matched SAC/scalar/budget training, initialized-training costs, strict native-prefix falsification, deployment-label ablation, all model/source/raw archives and failure records. New operation benefit and independent algorithm gain bothUNVERIFIED.


## 2026-11-04 — Successful-behavior retention and controlled exploration

[Complete real evidence and reproducible artifacts](evidence/EXP-004/2026-11-04/success_retention/README.md). Fixed actual predecessor, same Peg100/feedback actions. Four32-episode SAC branches followed by4000 replay updates:8/16,9/16,10/16,9/16; nominal9/16. Controlled SAC2 rescues/1 regression. Same-replay whole-episode nominal-success anchoring restores9/16 with0/0 on both behavior datasets; weight0.05 returns1/1. Posthoc strong phase/budget feedback12/16,3/0 is control progress, not RL originality. All trajectories/models/source SHA retained.505 full episodes/50500steps,41120 primary update iterations,1336 training successor decisions,3841 planner queries;9 exact native repeat trajectories.

Retain both nominal-retention and learned-recovery endpoints, stronger feedback reference and all negative variants. No task/runtime/horizon change or sealed-test expansion. Next test stronger-reference real training data as a shared learning anchor; do not expand probability heads or repeat the same coefficient search. See DECISIONS and STATE for the continuing experiment. **UNVERIFIED independent method gain.**


## 2026-11-04 — Intervention credit and changed continuation

[Full real evidence](evidence/EXP-004/2026-11-04/intervention_credit/README.md). Fromcc6774c;616 complete native episodes/61600steps,9480 primary Actor/Critic iterations,4543 planner queries. New24dev nominal17, frozen rule18; fixed-pair SAC10/AWAC-style17/measured-gap16. Current-policy paired acquisition with SAC/AWAC/gap/direct preference all17; same-data memory consolidation both17. Budgets equal within groups only; one inherited neural initialization, no independent gain.125 measured-prefix/queue-verified contrasts; no PhysX restores.

RETAINED: real online intervention-credit trainer, same-data mature AWAC-style control, exact queue evidence and all weights. Concrete positive training contrast757300 action1 at59; prior action probability1.8e-6 attenuates expected-SAC gradient. Direct signed log-odds learns that first intervention but complete native reexecution still fails through later replans; first69 actual steps equal the successful witness. Thus failure is not merely missing action selection. Frozen continuation credit does not automatically justify modifying the shared continuation policy. Memory repetition adds no development gain. This is a local negative method result with a specific policy-improvement defect, not another prediction-head proposal.

Next continue this mechanism through continuation-preserving policy improvement versus mature trust-region and successful-witness imitation controls, sharing physical data/authority. Do not relabel pairwise preference, AWAC, Q-filter or KL as originality. Do not return to fixed branch scoring or tune the old strong rule. Preserve native100, actual prefix costs and all failed trajectories; no sealed/hardware/ARES-R edits.


## 2026-11-04 — Continuation-explicit Options and three learning revisions

[Real implementation, models, raw outcomes and review](evidence/EXP-004/2026-11-04/continuation_option/README.md). Basec4219f2; code758ab2c. New event-terminated reference-continuation Option reproduces all100 physical steps of757300 success witness; phase-only commitment fails. This is mature temporal-interface capability, not originality. Action0 retains the active queue. Ordinary learners and frozen-rule control share the new interface; original frozen rule8/16 versusOption-wrapped rule7/16 shows the contract is not universally better.

Actual three training loops: same-data standard Option-SAC / successBC / measured preference+suffixKL; then joint continuation likelihood; then real SMDP termination-target credit.24 new natural training starts,16 initially new development starts. Final ordinary7/16,7/16,6/16; local4/16,4/16,6/16; joint3/16; SMDP-credit4/16. All learned final variants zero development rescues. Selected training native retakes candidate5/5,BC4/5,ordinary2/5; bothcandidateandBC truly learn758002/004, so local selection is no longer the only failure. Candidate overgeneralizes action5 to12/16 development states; measured terminal deltas and bootstrapped current-tail advantages must remain distinct. No independent gain, convergence, multiseed or sealed-test claim.

Costs484 fullnative episodes/48400steps,48 training episodes/24 exact native-prefix pairs,6400Critic and9600Actor iterations (including3200extra Actor-only),90savedmodels,3847plannerqueries. All old assets and all new failed variants retained. New runtime and learned training repairs are reusable; simple Option commitment,KL,joint preference or SMDP correction are not claimed original. Next isolate timing/state generalization with real reachable boundary interventions and checked current-policy tails; do not repeat unchanged losses or return to fixed-branch scoring. Predecessor remained real grasp1 for attribution; whole multi-skill adaptation and second-task research remain active. ARES-Rread-only; nohardware/sealed/restricteddata. **UNVERIFIED independent method gain.**

Post-study shared-runtime fix: four intermediate evaluations exposed successful empty planner paths; none were final checkpoints. Measured-q one-step hold eliminates the indexing failure and separately rescues758204 in100steps. Add1validationepisode; preserve all historical scores. This is sharedcontrol, not algorithmgain. See continuation_option/EMPTY_PLAN_FIX.json.


## 2026-11-05 — Own-policy learning, exact resume and target-origin falsification

[Evidence, models, complete trajectories and video](evidence/EXP-004/2026-11-05/online_integrity/README.md). Froma8418f9. Question: can inherited Options train credibly with own-policy data, and is a3mm feedback parameter actually a local execution change? The named ARES_SCI_RESEARCH_TASKBOOK.md was absent; requested its location and proceeded with explicitly authorized work, without claiming it was read.

Implemented immutable source/seed manifests, isolated own-arm replay, atomic complete learning-state transactions, explicit incomplete-physics detection and idempotent resume. An actual collection-before-update interruption matches uninterrupted weights/all optimizers/RNG/replay and200native steps exactly; changedseed rejected. Arbitrary mid-PhysX restoration is not claimed. Restored real learned3-grasp predecessor decisions; retained native100 and frozen shared runtime/reference.

Two64-episode online arms,5120Actor/Critic updates each: ordinary final6/16 (1rescue/4regressions), mature successful-behavior BC9/16 (1/1); nominal9, frozen historical rule8 (1/2). Ordinary intermediate10/16 is retained but not substituted for predeclared final6. Initial shared inherited9/16, one neural initialization; no convergence or multiseed claim. Data are policy-specific, not a shared-data loss-isolation experiment. Actual replay audit958decisions including128front decisions, no dev/other-arm transitions.

Mechanism4natural scenes×3real grasps: zero5/12, refresh±3mm0/12, active-target+3mm2/12, active-target−3mm5/12. Additional no-offset attachment-refresh control0/12 excludes 'only the sign of3mm' as an adequate explanation. Current target changes5.74–21.98mm; relative target3mm reduces commanded-path disruption but adds0rescues.12zero repeats reproduce1200physical steps. Grasp affects downstream outcome (759501:0/1success,2failure); not yet a learned general handoff result.

344full episodes/34400steps,2501queries,10240main update iterations plus16resume-proof iterations. All failures/models retained; exact sources, selected public model/archive SHA, full local checkpoint inventory and one RAW/English synchronized video. Retain the online/resume foundation and opt-in operator as research assets; neither BC nor relative coordinates is originality. Next keep the current cohort immutable and compare the new action interface in a separately versioned online group; require a real residual before introducing another loss. Independent gain UNVERIFIED. No hardware/sealed/restricted-data/ARES-R changes.


## 2026-11-05 — Decoupled execution references: matched online learning and frozen confirmation

[Evidence and reproducible raw archives](evidence/EXP-004/2026-11-05/decoupled_interface/README.md). Base cc0a0ff; core implementation ba7ad8c. Filled remaining_target × ordinary SAC / successful-behavior BC with64 own-policy episodes each, inherited model, same train/dev,3 learned real grasps and native100. Final seen-dev6/16 and8/16 versus historical measured6/16 and9/16: no advantage. Historical final64 unchanged. Equal additional64 BC episodes and a second128episode stochastic repeat retained all negative checkpoints; frozen dev selection chose64 for all four, while charging128episode budgets.

Frozen new32 natural conditions: measuredBC14/32,12/32 versus relativeBC17/32,18/32; nominal19/32; measured/relative frozen rules17/32,16/32. Relative reduces regressions5→2 and7→1, adds zero rescues. Two stochastic repeats share inherited neural initialization. The second16-condition cohort was declared after examining the first16; this is preliminary development confirmation, not sealed/fixed-sample significance. Own-policy datasets differ by design.

Mechanism: trained policies on759205/grasp1 share94 recorded physical steps. Continue succeeds; relative+3mm replans15points with6native steps remaining and fails. A fully reexecuted zero-offset replan also fails; four additional natural cases show unchanged complete outcomes. Replanning alone can break completion without target displacement; geometry versus retiming/velocity is not uniquely identified. No3mm-to21.98mm numerical-amplification claim.

Costs:512 new training episodes,3467effective decisions,40960Actor and40960Critic iterations (+40960unusedaux);1067complete native episodes,106700physical steps,8764planning queries. RAW/annotated101frame full state-playback video adds0physical steps. Complete raw/model/source SHA retained.

Retain representation-aware online training, model selection, all failures and real predecessor dependence. Next implement a standard progress-preserving local residual on unexecuted waypoints with shared IK/collision/dynamic checks; never compress15points into6 or extend native budget. This comparator is not yet implemented. Paper Problem/Related Work/Method/preliminary Results now updated; representation-retention is preliminary positive evidence, independent newRL and nominal superiority remain UNVERIFIED. NativeCharger200 second-task reuse prepared without incompatible replay mixing. No ARES-R/hardware/sealed/restricted-data changes.


## 2026-11-05 — atomic_handoff：可学救回成立，稳定净优势尚未成立

从9a66e94继承，完成真实own-policy训练320回合、5臂、两个独立NN初始化；共1002原生episode、101800物理步、8934规划查询。
H1单次atomic不足，H2重复阶段刷新可复现760213完整100步成功；实际首次接触87 vs98步。
761202共同63步前缀证明学到的局部动作必要且单独足够；冻结新762214同样成立，762213则有更宽的有效时间窗口。
新24确认：名义12、冻结规则15、queue12/12、macro10/12；后者分别1救回3退化、1救回1退化。两者都高估了762206的破坏性干预。
不将更多决策、标准参数化Option、83维状态增广或动作字典本身称为新算法。H3拼接10次全拒绝，尚无操作能力；第二任务Charger20016/16饱和，无迁移增益。
保留全部模型、失败、完整事务resume和历史分数。重点转为真实可执行续接下的干预价值校准与保持；当前确认集已看，不能复用为新测试。
证据：[atomic_handoff](evidence/EXP-004/2026-11-05/atomic_handoff/README.md)。完整成本、SHA、RAW/英文同步视频与论文表已归档。
