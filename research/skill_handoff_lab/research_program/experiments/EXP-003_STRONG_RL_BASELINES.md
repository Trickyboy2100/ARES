# EXP-003｜一篇 SCI 论文的双工作流：成熟RL基线 + 真正多阶段任务

**状态：RUNNING；阶段S4；继承EXP-001/EXP-002负结果。**
此卡是当前实际研究任务；允许Codex同一次会话内连续训练、诊断和合理改进，
不设日历/固定训练量，不允许只做一次Gate后交十页计划。

## 研究目标（不可变）

在有可复用名义运动规划和Skill Runtime的机器人上，RL根据任务条件、
**已实际执行的前后技能物理效果**，自主选择合法参数与必要局部动作，
比相同权限的普通RL、成熟规划/反馈更可靠地完成完整多阶段任务。
它不是“让插头终态从10/12到12/12”的单一局部插入控制论文。

## 已成立证据，作为下一轮起点

- 旧StackPyramid原生250步已实际执行PickA→PlaceA/Release/Verify
  →PickC→PlaceC/Verify，带两个技能决策边界。64开发普通同接口PPO
  三种子51/53/56，风险候选51/54/55：新方法平均无增益。强几何55。
  不再单独在这套已看参数与结果上无意义扩训。
- 最新ARES-AdaptivePlug-v0四初态×Y观测偏差−1.2/0/+1.2mm的12个条件：
  几何4、RLS6、接触搜索10、普通SAC1、匹配alpha0.02 Critic-LN SAC10、
  效果+LN2；另普通SAC同权重部署Z置零7。
  只有**四个不同场景**，没有多种子保证；LN是已有DAWN式优化技术，
  当前effect feature并无信息/表现优势，不能继续视作论文创新。
- LN-SAC两例达到过原生success条件，但因过严的**有偏观测停止条件**
  未进入hold，持续控制到200步后失败；规则修复先共享给所有方法。
  不允许用真实仿真success作为部署停止的特权oracle。

## WORKSTREAM A — 把低层学习基线真正做成熟（共享基础，不是本论文题目）

A1. **可观测稳定完成合同**：定义依据估计工件/目标相对位姿、
速度变化和少量保持(dwell)步骤的终止-保持规则，加入合理迟滞；
在相同完整200-step物理任务中对几何/search/普通SAC/LN-SAC全体共享。
报告ever-success、正确终止并保持、错误提前hold、真实终态success，
没有隐藏真值泄漏。与规则终止对照；若规则充分，不训练终止网络。
别将两个已看LN失败个案“修好”后当新独立测试。

A2. **动作权限诊断变成公平的新训练**：明确偏差只在Y的分布与独立
真实YZ/工具姿态/抓持偏差分布。使用相同信息、同名义控制、同奖励、
同预算，分别训练 (i)完全局部Y/Z/推进，(ii)物理有根据的Y-only/推进，
(iii)依据当前/后续技能合同限制合法参数的普通RL baseline。
不仅在SAC模型输出后把Z归零；各种强规则也拥有同等合法权限。
直接报告不同任务条件下应/不应开放Z，不能说Z永远是噪声。

A3. **公平RL成熟化**：以匹配alpha的Critic-LN SAC为正式已知强学习
基线，可按许可独立借用DAWN名义数据预热、归一化等思想；对照普通
SAC或PPO，但不要重复弱1024-decision初版当论文最终对照。
逐模型报告真实physical steps、完整episode、有效skill decisions、
独立训练seed、训练/测试条件、checkpoint、名义预采集/搜索预算，
逐阶段学习曲线和可解释的Actor/critic动作排序。未见开发组用于调，
sealed始终封存。不要强求Isaac16环境并行或把串行当低质量科研。

A4. 如果成熟LN-SAC/规则在原生或当前偏差族里饱和，应保留为基线，
**不要为了凑论文增益在四场景上继续调stop/alpha**；进入工作流B。

## WORKSTREAM B — 尽快恢复真实的跨技能研究对象（更高论文优先级）

B1. **候选两技能任务**优先同时小规模真实验收以下可行性：
- PullCubeTool：第一个技能Pick/GraspTool的抓持位置/方向影响
  第二技能Hook/Pull的真实接触效果与完整成功。历史增强几何已7/8
  但存在未见条件；不得回退弱原官方planner制造收益。
- StackPyramid：原生两边界Pick/PlaceA→Pick/PlaceC，保留真实
  松爪/稳定/支撑状态；当前固定参数学习空间狭窄，只有存在新的
  合理任务条件与不同后续效果时才重新用作主实验。
- 或者抽屉/工具使用/双臂合成任务中能真正让前段状态限制后段
  动作的任务。只选一个主任务及一个不同物理机理的辅助任务，
  不同时安装四套新引擎。

B2. 对候选任务形成共享且实际可执行的SkillContract接口：
`precondition, controllable_parameters, physical_execution,
measured_postcondition, next_skill_requirement, termination`.
**强约束：**在同一episode里至少两个技能级参数决策，后段必须基于
前段真实状态做一次重新决策；前段参数需可通过真实物体终态影响后段
的可行性。禁止planner waypoint当已完成skill、瞬移/重置、手工
未来成功标签进入Actor。

B3. 研究**完整任务机会**而非单步接触成功：相同真实初态下，
对有限且合法的前段参数+后段参数进行物理干预，分别统计
predecessor success、downstream conditional success、whole-task
success、相同参数重复、最强固定规则、等预算搜索/监督ranker、
已测策略成功并集（oracle仅诊断，不是policy成绩）。
最关键的是：哪些**不同任务条件**会改变最佳参数方向？规则/规划
已达到多少？能否从合法物理执行反馈判断？
若只有单次决定，不包装为多阶段Option/SMDP。

B4. **尽早恢复主论文机制研究**：当B3出现可学习的后续依赖，
与A3成熟RL基线共享训练底层。Codex可自主测试下一阶段合同相关
参数价值、执行效果适应或其他明确机制。先与简单规则、普通
同接口RL、同数据监督Q、STAP/Foresight式后续价值/搜索比较；
创新不能只是“选参数”“加历史”“一个mask”。

## 是否引入POMDP/信念估计？

**只有**存在合理的隐藏抓持/物理状态z，且相似当前可观测量出现
不同合法动作效果，并且短时行动结果能帮助区分时，才研究：
`b_{t+1}(z) ∝ p(observed_effect | z,o_t,a_t)b_t(z)`。
对照完整状态oracle、同历史长度RNN/MLP、在线RLS/自适应模型、
SCOUT类动作效果更新；额外物理探针算训练/部署预算。
这不是预先确定的论文方向。若状态已充分可观察，就坚持MDP或SMDP。

## 关键学术判断：创新机制应从“技能组合”中生长

优先检验 `successor-contract conditioned parameter policy`：
RL看到当前任务、当前skill和**下一skill实际合法条件**，在允许
动作空间内选择物理参数，并按前段真实结果更新后段动作；
学习目标是完整任务价值及实测后续可执行性，而不是只优化局部轨迹。
此概念与RAPS/STAP/Foresight高度相邻，**不是完成的方法**。
要有区别于固定skill Q乘积/离线future success predictor的具体
训练/决策机制，并在普通相同合同输入的RL和强规划对照下证明收益。
若证明不了，修改理论而不是只加算法名字。

## 本轮Codex必须交的科研证据（非论文PDF）

1. A1可观测终态合同和A2动作权限的**新场景物理配对结果**；
2. B1-B3中一个真正跨两技能、具有可重复动作互补性的任务，
   及不依赖人为弱基线的机会数据；若第一个任务缺乏空间，主动探索
   一个新物理任务，不仅返回Gate。
3. 必要时同会话完成A3成熟RL与B4候选机制的真实训练，小步实验
   继续优化，不要仅产出设计文件。
4. 从原始真实物理记录生成各一组RAW.mp4、ANNOTATED.mp4、
   可点击同步trace/物理事件、配对救回及退化图；区分重渲染和在线录制；
   对新方法必须显示前段参数、真实终态、后段动作、完整任务结果。
5. 更新experiment/STATE/DECISIONS/EVIDENCE，提交push，中文简报
   只报告实际完成、强对照、理论结论、下一项有效实验；不必改论文PDF。

历史工作单编写时尚未执行；2026-11-01实测更新如下（保留原研究要求）。


## 2026-11-01 实际执行与研究决定

A三种同合同普通LN-SAC各4096决策/3840更新：full 31/36，Y-only 33/36，task 35/36；原任务条件搜索32/36，扩展推进上限的快搜索task 30/36。完整2048/4096实际检查点、停止合同对照、模型及轨迹已归档。单训练种子、6开发场景，投影三维策略的限制明确，不宣称充分收敛或新算法收益。

B完成77原生物理任务/9200步、3任务族；完整后段TwoRobot几何8/8，相同前段正交干预2/4。工具同路径降速降低碰撞但不消除差抓持漂移。未建立强模型之外的条件排序反转，因此不启动缺乏物理依据的新方法训练。

[完整结果、视频、复现和下一项物理实验](../evidence/EXP-003/2026-11-01/README.md)。EXP-003继续作为活动实验完成B机会检验；EXP-004原创增量未验证。下一轮不扩大插入或重复强servo，执行已声明的稳定边界/工具负载小矩阵，并允许依据负结果调整任务。

## 2026-11-01 独立期刊外审后：STAP主对标及真实多技能增量

请先阅读[独立配对复算与STAP方法复现协议](../evidence/EXP-003/2026-11-01/INDEPENDENT_REVIEW_AND_STAP_PROTOCOL.md)，并运行 `python research/skill_handoff_lab/research_program/scripts/audit_exp003.py`，核对当前开发证据。

本轮研究目标不是继续提高已看PlugCharger的35/36，而是：**精确复现STAP核心算法→在共享真实两技能环境建立强规划/价值基线→找到可实测的不同后续合同下参数排序差异→提出单一可消融RL增量**。Foresight作为实际后段结果预测的次对标，RAPS负责技能参数动作表示，DAWN仅作为成熟ResidualRL训练基础。先做已声明PullCubeTool稳定抓持/适度负载8条件真实对照；没有有效物理条件时允许有研究依据地转向不同操作要求/任务版本，不能为制造增益削弱强控制。不要在下一轮只输出新的复现计划、算法名称或Gate；至少交算法级fixture实测+有实质信息的完整物理对比，有学习机会则开展实际模型训练与消融。当前真实RL新方法增益仍为UNVERIFIED。

### 执行续接｜2026-11-01：STAP核等价运行；真实后验重规划可能退化，先加强基线覆盖
- [完整证据及交互审查](evidence/EXP-003/2026-11-01/stap_followup/README.md)。首先实际独立复算历史CSV，结论不变。固定官方STAP算法体与移植12组全候选精确匹配；旧包gym缺失，未声称整套benchmark复现。
- 新增224条真实100步物理执行、22400控制步；同64条数据训练绝对/增量状态两个模型版本，各2800更新，总5600。2个训练scene、1个训练seed；普通Fitted Q/Foresight类比均3/8，不是收敛SAC/PPO强基线证明。
- 严格同首参数和逐记录物理前段的STAP开环6/8，后验重规划3/8，3退化0救回。增量状态模型位置误差约89–175mm降到10–35mm，任务仍4/8开环、3/8重规划。没有模型精度→任务增益的证据。
- 强固定100mm/−65mm为4/8；确定性预算保护也4/8，修复一个184步/剩余61步的不良选择，但不是RL创新或预算保证。训练条件固定105mm/−55mm仍8/8，更宽目标未证明抓点排序反转。
- 科学决定：不训练额外合同网络掩盖稀薄数据。先用新训练scene有界覆盖真实后段动作，固定强STAP/普通RL/监督和几何基线；只检验一个“合同裕量辅助价值决定保留或修正参数”的增量。现有新scene已用于诊断，不作为封存测试。维持UNVERIFIED。两组完整RAW/ANNOTATED视频与同步trace均实测姿态重播，零新增物理。


## 2026-11-02｜真实SkillResult配对、数值修复与最小机制否证

[结果/关键图/同步视频](../evidence/EXP-003/2026-11-02/skillresult/README.md)。保留历史，新增320真实物理回合/32000控制步、13600优化更新。96干预面板固定8/16、事后上限9/16，无严格目标排序反转。复算三例退化，区分低Q强制改参、预算与双成功合同。发现概率饱和选参并列，保留112诊断并以logits/log-Q另跑112确认。修正后ordinary1/8，generic/contract/relative/relative+contract均2/8；固定、强几何、STAP-open和3critic-UQ均3/8。组合机制没有独立增益；2个新测试初态已作开发诊断，未使用743052/53。公开模型/真实数据/四个完整视频及同帧曲线已归档。下一步优先原生多技能后继合同冲突的物理机会，不扩大窄hook网格或网络模块；保持UNVERIFIED。

## 2026-11-02｜从局部负结果累积到下一物理机制

对03c4176的原始train96物理行重新按(seed,mass,target)分组：16条件中7组六参数全成功、7组全失败、只有2组混合；其中743041正向80mm只有抓点选择改变成败，743046正向80mm只有hook选择改变成败。已测参数oracle9/16、强固定8/16，未观察到严格后继目标排序反转。由此暂缓扩大同一窄hook网格、relative+contract/generic价值头或Plug训练；这一否证仅限当前动作与任务条件，不否定真实SkillResult学习的普遍可能。新增物理0、优化0。

工作流A继续保存成熟SAC、强规划及数值修复为共同冻结底座；工作流B优先复用已有原生StackPyramid的PlaceSupport→PlaceTop、真实释放/稳定、完整250步以及576次参数网格与几何基线。先核实旧最佳固定59/64、已测上限62/64所对应的失败机制，再在不同、真实可行的上层放置目标/支撑几何/后继抓取净空要求之间做小型真实物理配对。前后技能必须有实测SkillResult、合法动作及独立后继决策；不同后继合同应可能对前段合法参数产生不同物理要求，而不是任意把阈值调难。若任务参数/几何已改变官方设置，应作为独立adaptation variant报告。

每次少量干预优先测：前段合格率、后段条件成功率、整链成功、物理裕量/预算、按初态的全成功/全失败/mixed、最佳固定与实测oracle以及任务条件是否改变参数偏好。名义强规则/几何完整后段与学习策略共享动作权限。若Pyramid依然缺乏额外学习空间，Codex可在现有TwoRobot/工具使用中自行选择更有真实后继冲突的操作，并在本会话继续物理试验，不停在机械Gate。发现可靠机会后，再比较同接口普通RL、STAP、Foresight与一个从物理结果生长出的最小方法增量。此前未验证的算法名称和辅助目标不必继承；原始证据与已验证组件必须继承。方法状态仍UNVERIFIED，密封测试不动，也不需要重编论文。

特别注意：现有16个2×3成败网格中，两个mixed上下文分别表现为单独抓点效应、单独hook效应，未看到二者在二值完整成功上的交互。因而下一任务应优先实测不同后继合同下的实际前后技能耦合，而不是仅把两个独立参数头组合起来。


## 2026-11-02｜累计物理探索：后继朝向差异被强几何解释

[关键图、同初态完整视频与原始证据](../evidence/EXP-003/2026-11-02/pyramid_contract/README.md)。先执行新增机会脚本，并核对96原始工具轨迹、576旧Pyramid终态及26历史轨迹SHA。新增90真实回合/22500控制步、0优化：单代表朝向pilot0/18；最近对称姿态29/36；四等价姿态预放置/放置可行性检查36/36。后两者36组前段记录完全相同，7救回0退化，属于确定性规划改进。强控制下2初态×2真实顶块朝向合同均9/9参数成功，固定与已测上限均4/4，无排序反转。保留SkillResult、对称可行性组件、原有RL/STAP与负结果；不在饱和条件训练。后续仅在真实布局/后继净空出现强规划后的不同物理参数偏好时训练。原生成功/250步不变，追加朝向合同另列adaptation variant；stride3及未建world-cloud碰撞约束明确披露。独立方法增益仍UNVERIFIED；密封测试未动。

## 2026-11-02｜文献+原始物理重分析后的执行顺序（紧接最近Pyramid，非新物理）

[详细理论与下一闭环](../evidence/EXP-003/2026-11-02/pyramid_contract/THEORY_REVIEW_NEXT_RESEARCH_LOOP.md)；[跨任务标准库复算脚本](../scripts/audit_cumulative_skill_opportunities.py)。

本轮独立数据再分析发现：旧576条Pyramid的64初态中9个存在有效前段参数后果、13个存在同前段后段参数效果、两类重合6个；420002和420023相对名义只有联合改变两个参数才能在已测网格取得成功。这些已看开发案例只能用于形成假设，不可作为新方法测试。新Pyramid yaw适应经过四等价姿态搜索已36/36饱和（2 scene），该技术**保留**；不能将局部新yaw负结论推广为旧Pyramid所有跨技能物理耦合消失。工具已测96条二值成败大多同值，但连续目标误差仍变化，必须区分目标/原生成功/实际预算/执行残差。

恢复类似df268c0的自主推进：首先对旧Pyramid 576中的交叉敏感和共同失败轨迹进行分类并核对已有强规划（失败是不是释放/抓持合同、夹爪扫掠、姿态对称、真实接触或时限）。随后**真实跑小型交叉物理实验**：优先已有axis-aware支撑布局与后继接近/净空合法条件，两个实际技能参数决策与原生250步/连续真实物体轨迹；强名义完整后段和四等价姿态保持。若几何方法再次解释结果，不停在Gate，而在同一会话进一步验证Tool的前碰撞附着漂移与后段时限条件，使用100Hz真实记录、稳定前段合同、合法的后段物理修正（不得简单重复旧窄hook网格）；明确命名变体并共享强控制。

在任何物理对比中逐案例保留：合法前段和后段候选、前段实测SkillResult、接触模式、剩余预算、几何规划是否可行、预测 vs 实际动作持续步数、原生全任务成功与追加合同、同前段rescue/regression、有效RL决策数和规划查询。若物理可用机会出现，**同轮继续**普通同权限两边界RL（参考SPIRE/DAWN基础）与SeqDex/ViSkill式后继feasibility、STAP/Foresight强对照；仅提出并实现一个小而可消融的执行后果/预算机制。允许连续做2–3次有区分力的物理—学习闭环，不能只做审计/写十页报告。检验没有严格反转但存在持续物理成本差异时，可先做物理因果诊断，独立测试之前再决定是否把效率列为论文主指标。没有可靠机会不强行训新网络。

额外系统基线改进：将已有Y-only 3D-SAC输出投影的混杂与真正2D actor/critic/entropy分开；在A线有界验证新action表示，但**不能替代B的多技能研究**。任何控制/规划修复为所有算法共同版本，明确保留既有训练与失败而非将之计为独立贡献。新增本轮文献/纯数据分析=物理0、优化0；方法依旧UNVERIFIED。下一次Codex应该从当下STATE与最新Pyramid README继续执行实测，而不是重复已完成yaw实验。

## 2026-11-02｜接触/时长累积闭环（contact_duration）

完成312有效物理episode/44700步，另8个FK日志API错误尝试800步单独INVALID保留；11000真实优化更新。旧Pyramid576审计与原初态counterfactual厘清C抓持/前段合同失败，原36/36几何资产保留。工具干预baseline3/12→measured/frozen refresh均5/12；严格前段一致36/36，反馈信息非必要。等时等待不充分，plan_screw Success仍有13–14mm终点FK误差；细积分下降但成功抓位交叉，不能宣称通用修复。108新设计有限网格最佳固定6/12、oracle7/12；新开发普通FittedQ、监督、STAP-open/replan、后续价值类比及固定均2/4，继承100mm参考也2/4。没有独立算法增益，UNVERIFIED。

永久保留：旧SAC/STAP/log-Q/36of36、两边界真实结果、只读100Hz接触与FK检查。下一轮先共享计划终点/实测到位/接触入口/预算合同，并用新设计物理配对验证抓位×后继调度残差，不扩训练量或叠模块。初始n2与原n16同seed不相同，54诊断条不可当旧case复现；另36条仅reset恢复记录初态，之后真实物理。详见[研究结论](../evidence/EXP-003/2026-11-02/contact_duration/README.md)、[审查页](../evidence/EXP-003/2026-11-02/contact_duration/REVIEW.html)。


## 2026-11-02｜共享执行修复与真实双边界 SAC（contract_sac）

[研究结论](../evidence/EXP-003/2026-11-02/contract_sac/README.md) · [审查页/完整同初态视频](../evidence/EXP-003/2026-11-02/contract_sac/REVIEW.html)。596完整真实物理回合/59200控制步，另76已保存部分步及首次崩溃未知0–100步。FK计划误差降低不等于实际到位；stroke2×2证明3/8→6/8由合理行程权限解释，无需后验反馈；实际起点微重规划6/8→6/8。接触前重载抓持漂移与短期稳定不矛盾，统一100Hz力/20Hz姿态定义。

恢复真实两边界SMDP SAC：保留575已完成决策、1022实测更新、SAC256/512及中断模型/缓存；645条转移物理审计误差0。第一次67回合TOPP崩溃保留原数据，不能假称恢复丢失actor。相同323回合数据训练模型/监督/后继价值10000更新。新8开发初态，固定70/105、SAC256/512、监督、STAP-open/replan、后继价值全部7/8。共同失败746204为TRANSFER规划失败，STAP概率约0.00019；不是高Q误导。1训练seed与有限检查点，不宣称收敛/已充分RL。原有A线、Pyramid36/36、STAP/log-Q保持。

36高度网格无额外后继二值机会；36加速度配对无救回且一次预算退化；拓展yaw一次退化；翻转夹爪未救回两个共同失败。原生PokeCube50步标准/直接接近各0/4，已实际追查预算与抓取，不能当弱基线制造RL收益。下一具体问题是强规划下剩余可达路径/中间工具姿态与实际跟踪/接触预算；第二任务先建立可信共享控制器。所有正负模型/物理/成本/SHA公开仿真归档，密封数据不动。独立算法增益 **UNVERIFIED**。


## 2026-11-02｜post_contact：真实动作作用与效果方向SAC负结果

[研究结论](../evidence/EXP-003/2026-11-02/post_contact/README.md) · [完整审查页](../evidence/EXP-003/2026-11-02/post_contact/REVIEW.html)。本轮858完整原生物理回合/85800控制步，另173部分步；不是858独立场景。SAC3186梯度迭代（actor/critic各一次），比较器1000联合优化。原高度审计修正为11个已选择上下文、8个实际HOOK/PULL上下文，原9的表述被原始记录推翻；高度仍无完整成败增量。746204仅3次CPU规划/IK诊断，0新物理。

继承强控制后执行四类后接触参数干预、普通SAC诊断失败同前缀追查、120条新设计物理面板与两轮时机干预。新面板14/24抓持上下文达到PULL修正边界，3个同前缀救回，但均已由另一名义抓持覆盖；最佳名义抓持与联合oracle均8/12。22个被审计状态有14个更早接触，PULL决策延迟中位19.5步。首接触与PULL使用同一个“修正后重接原航点”原语时可有方向排序变化，零动作8/8完整轨迹保持；不能将不同延续语义或时钟本身误当机制。

实际训练冻结效果方向映射+SMDP SAC，保留普通同权限、全局方向、通用效果辅助、监督评分与STAP功能核。物理尺度特征使训练场景CV效果误差0.02530→0.001301（固定效果0.003740），但最佳方向轴仅18/22→19/22。共同500更新：普通SAC/STAP6/8，方向候选/全局/通用辅助5/8。各再256实际决策后普通与候选均6/8，固定87.5mm也6/8；固定同一物理前缀三者均6/8，候选0救回0退化。普通/候选实际在线成本14170/14275步，单训练seed，不能声称完全等物理成本或充分收敛。两个576转移缓存与早期128缓存均由真实物理复算通过。效果映射Jacobian成立于共同no-op投影之前，不声称已证明执行动作原子分布的微分熵。

**决策：** 保留普通训练器、实际事件/前缀验证和所有正负权重；不将预测准确率、方向重参数化或共同控制进展认作独立RL方法。不扩当前yaw训练。下一项有价值实验必须建立最强抓持/控制后仍存在的可执行恢复收益，重点区分工具倾斜下的无效接触与真实钩入；若共享控制已覆盖，则以已有PlugCharger200成熟线研究实际抓持结果/接触条件，不降低基线、不动封存。方法继续 **UNVERIFIED**。


## 2026-11-02 — Recovery and native charger two-boundary learning

Actual continuation from 525d67e: 354 complete physical episodes / 64,400 steps plus 207 saved partial steps. Tool recovery: all six controls 6/8; early interventions all 0/4. Native charger strict-prefix factorial separates angular collision from advance/budget effects. Corrected whole-task-success SAC: 1,024 transitions / 1,792 updates, checkpoints and reconstructed buffer retained. New development: nominal and force guard 8/8, SAC256 3/8, SAC1024 7/8, same nominal grasp + SAC successor 8/8. Named mass/friction variant also covered by 15-degree nominal (8/8 conditions, two scenes). No independent algorithm gain; UNVERIFIED. Retain instrumentation and training objective correction, freeze covered recovery/pace panels. Next: genuine successor-contract constraints with residual failure after strongest shared control, not manufactured weak grasp or more old-coordinate training. [Full evidence, negative runs, videos, costs and SHA](../evidence/EXP-003/2026-11-02/recovery/README.md).


## 2026-11-03 — Parallel native Peg physics and trainable skill-value mechanism

[Complete evidence and code](../evidence/EXP-004/2026-11-03/peg_value/README.md). Active EXP-004 now runs alongside EXP-003 common execution. Official GPU Peg OBB helper used stale CPU world pose; preserve literal baseline and adapt local collision geometry to actual GPU pose. Corrected official timing1/6, explicit shared retiming3/6. Frozen-control two-boundary panel:60 trials /10 natural scenes, strongest fixed6/10, measured union8/10; same-prefix rescues and regressions are real. Prototype: paired return-difference loss, same-data FQI/absolute/suffix-value/STAP-style enumeration, discordant sampling ablation, then ordinary masked LN-SAC with paired/absolute/shuffled critic gradients. Historical early closed-selection40/69 vs ordinary32/69 largely matched generic labels; later43/69 vs44/69. Fresh actual Tool all3/6; Peg all2/4. Independent algorithm gain UNVERIFIED. Retain actual models, all failed trials, source/hash/buffer evidence and exact-frame video. Costs:160 complete100-step episodes including6 invalid instrumentation terminal holds;75600 joint value updates and51600 actor/critic updates EACH. Native horizons and success unchanged; no sealed data or ARES-R edits. Next causal question is executable contact tracking/budget under the shared Peg controller; reconstructed target is correctly +50mm along goal x, so do not re-debug a presumed reversed frame or merely train another grasp score.


## 2026-11-03 — finite physical Options, acquisition revisions and extraction mechanism

Source: [evidence/EXP-004/2026-11-03/discrete_handoff/README.md](../evidence/EXP-004/2026-11-03/discrete_handoff/README.md); code `2ac433e`, base `1ea0965`. Independent algorithm gain remains **UNVERIFIED**.

- V0 continuous projected SAC has off-support Actor/Critic targets. V1 implements exact legal finite options, real SkillResult, duration, termination and remaining budget; DSAC/FQI/IQL/CQL/supervised/functional-STAP fixtures and physical trace checks pass. This is mature algorithm infrastructure, not originality.
- Actual offline→online study uses compatible60-episode Peg prior, fresh natural training scenes, two optimizer seeds and native100.38 model histories exactly reconstruct from their actual physical buffers.2458 full episodes/245800 steps include all training/evaluation/posthoc confirmation;250400 primary critic optimizer iterations exclude audit reconstruction and toy fixtures.192 pairs reexecute and charge both full prefixes.
- Expected-update acquisition was implemented, physically trained, then revised for wasted duplicate coverage, correlated suffix outcomes and current-task anchor distribution. Joint outcome prediction improves old-scene leave-one-scene-out counterpart Brier0.313→0.182; covariance changes7/16 actual acquisition rounds, but does not yield a reliable policy gain.
- Eight predeclared fresh development scenes: original joint and EES-style4/8 for both seeds; current-anchor revision2/8 and4/8;128-episode ordinary SAC5/8 and3/8. Same-history mature supervised/FQI5/8 for both. Best fixed6/8 is a hindsight comparator, not a training-selected policy. Across14 development scenes, fixed maximum10/14 versus finite measured union12/14 preserves a conditional-adaptation opportunity.
- A further actual intervention changes only extraction of frozen Q values: current-joint seed31301 Actor2→minQ4→meanQ5; mature seed31302 Actor3→minQ2→meanQ5.48 complete native rollouts confirm it. Mean-Q rescues3 and2 respectively with0 regressions; min-Q causes1 regression in mature seed31302. Actor approximation and clipped-Q aggregation are distinct. This is explicitly posthoc development diagnosis, not held-out performance or an independent RL contribution.
- The acquisition surrogate scores greedy decisions of one critic while deployment uses a learned Actor. Positive scores do not establish actual return gain; current-anchor seed31301 has observed update returns4→4→3→4→4→3→3→3→2 on the retrospective real branch panel. Retain this falsification and align any future mechanism with the actually deployed policy before expanding it.
- Permanent assets: finite Option runtime/learners; compatibility filter; exact optimizer/prefix audits; complete real branch evaluation; actor/min/mean diagnostic; trained checkpoints and RAW/English videos with frame-synchronized curves. EES and STAP comparisons are functional adaptations with disclosed limits, not claims of full original benchmark reproduction.
- No sealed data, hardware, ARES-R modification or restricted site data. New acquisition variants were frozen before reading expansion outcomes; later extraction confirmation is labeled posthoc. Next work is shared-baseline policy/value consistency and genuinely new development validation, not relabeling a standard extraction fix as novelty.


## 2026-11-04 — Natural coverage, value improvement and measured handoff targets

[Complete evidence](../evidence/EXP-004/2026-11-04/value_improvement/README.md). From60760b8, actual96 new natural initial states and822 native100 episodes/82200 steps;374400 primary critic iterations,268 checkpoints. Shared mature finite VIAC/USAC/BEE/IQL/FQI/CQL/STAP/conditional-scoring controls; no renaming mean-Q or empirical DP as novelty. Original3x2 training fixed24/32 vs union25/32; physical split/refreshed option adds one reproducible joint rescue, giving26/32 but no development union increase: both21/32 vs hindsight fixed19/32. All positives, regressions and exact prefixes retained.

Candidate substitutes physically measured suffix values at exact SkillResult into predecessor backups; complete-coverage revision changes only activation on full legal successor support. Offline3-option original14/32,13/32; revision15/32,17/32. Six true32-episode online runs: revision19/32,19/32; ordinary mean-critic SAC19/32,19/32; exact value improvement18/32,19/32. Candidate vs SAC1 rescue/1 regression, then0/0.48 additional actual final-policy rollouts exactly match their complete100-step physical branches; other32-scene model scores are explicitly logged physical-branch evaluation. This is catch-up, not independent gain.

New diagnosis: legal newly introduced actions lack historical execution support. Ordinary SAC terminal new-action Q exceeds1 on7/29 distinct old handoffs per seed; CQL removes this audited overflow. Same-data FQI15/32,18/32 versus CQL18/32,18/32, still not an independent method. IQL18/32,19/32. Neither calibration nor an isolated control rescue proves a paper contribution. Development now seen; complete-coverage/CQL are disclosed iterative revisions, not sealed validation.

RETAIN strong shared controller, explicit action-schema compatibility, finite RL controls, all real data/weights and shared split primitive. LOCAL_NEGATIVE measured-backup/coverage revision. Next investigate a physically legal broader predecessor parameter domain with explicit grip margins and the nominal option retained; require opportunity beyond strongest fixed/search control, keep conservative/support-aware baselines, and do not expand this failed backup unchanged. Method status remains UNVERIFIED. ARES-R/read-only and hardware/network isolation preserved.


### 2026-11-04 — Persistent closed-loop successor research

See [closed-loop evidence](../evidence/EXP-004/2026-11-04/closed_loop/README.md) and latest DECISIONS. The predecessor remains task-conditioned; successor now updates bounded parameters repeatedly under native100. Ordinary/scalar/budget matched training and local curriculum are real; mature retained SAC remains10/16 versus new final5/16,4/16,6/16. Deployment-label revision9/16,7/16,7/16 does not establish a budget-method gain. Six proposed recovery schedules give zero6/6,rule6/6,mined4/6 under exact native prefixes. Keep this closed-loop asset and investigate stable feedback learning, not more fixed-branch scoring. No sealed tests; independent gain and new beneficial operation capability remainUNVERIFIED.


## 2026-11-04 — Successful-behavior retention and controlled exploration

[Complete real evidence and reproducible artifacts](../evidence/EXP-004/2026-11-04/success_retention/README.md). Fixed actual predecessor, same Peg100/feedback actions. Four32-episode SAC branches followed by4000 replay updates:8/16,9/16,10/16,9/16; nominal9/16. Controlled SAC2 rescues/1 regression. Same-replay whole-episode nominal-success anchoring restores9/16 with0/0 on both behavior datasets; weight0.05 returns1/1. Posthoc strong phase/budget feedback12/16,3/0 is control progress, not RL originality. All trajectories/models/source SHA retained.505 full episodes/50500steps,41120 primary update iterations,1336 training successor decisions,3841 planner queries;9 exact native repeat trajectories.

Retain both nominal-retention and learned-recovery endpoints, stronger feedback reference and all negative variants. No task/runtime/horizon change or sealed-test expansion. Next test stronger-reference real training data as a shared learning anchor; do not expand probability heads or repeat the same coefficient search. See DECISIONS and STATE for the continuing experiment. **UNVERIFIED independent method gain.**


### 2026-11-04 — Shared successor learning comparison

[Intervention-credit evidence](../evidence/EXP-004/2026-11-04/intervention_credit/README.md). Same-controller AWAC-style control retains17/24 where fixed-pair SAC finishes10/24; current-policy-pair SAC andAWAC both17/24. Frozen strong rule18/24 onnew natural development. These short continuation studies inherit mature weights but do not establish full RL convergence or a new method. All physical budgets, checkpoints and negative results retained.


### 2026-11-04 — Shared continuation Option baseline

[Continuation evidence](../evidence/EXP-004/2026-11-04/continuation_option/README.md): standard same-interface Option-SAC and successBC were actually trained over48 shared native paired episodes. Ordinary fixed checkpoints7/16,7/16,6/16; frozen originalrule8/16. No candidate independent gain. Retain exact physical continuation and actual-duration SMDP infrastructure; inherited initialization is mature, but these new-interface runs do not establish convergence.


## 2026-11-05 — Own-policy ordinary RL mainline

[Native results and integrity evidence](../evidence/EXP-004/2026-11-05/online_integrity/README.md). Two64-episode own-policy arms with genuine predecessor learning, isolated train/development and exact resume. Ordinary9→10→6/16; mature retention9→8→9/16. Final endpoint predeclared. These are functioning mature baseline implementations, not evidence of convergence or original method gain. See EXP-004/STATE for target-origin mechanism and next separately frozen comparison.


2026-11-05: matched online SAC/BC execution-interface study completed; see [EXP-004 evidence](../evidence/EXP-004/2026-11-05/decoupled_interface/README.md).512new training episodes; standard SAC/BC retained as mature shared learners. Frozen32-condition retention improvement does not exceed nominal19/32 and adds no rescues. Historical final64 unchanged.


## 2026-11-05 — atomic_handoff：可学救回成立，稳定净优势尚未成立

从9a66e94继承，完成真实own-policy训练320回合、5臂、两个独立NN初始化；共1002原生episode、101800物理步、8934规划查询。
H1单次atomic不足，H2重复阶段刷新可复现760213完整100步成功；实际首次接触87 vs98步。
761202共同63步前缀证明学到的局部动作必要且单独足够；冻结新762214同样成立，762213则有更宽的有效时间窗口。
新24确认：名义12、冻结规则15、queue12/12、macro10/12；后者分别1救回3退化、1救回1退化。两者都高估了762206的破坏性干预。
不将更多决策、标准参数化Option、83维状态增广或动作字典本身称为新算法。H3拼接10次全拒绝，尚无操作能力；第二任务Charger20016/16饱和，无迁移增益。
保留全部模型、失败、完整事务resume和历史分数。重点转为真实可执行续接下的干预价值校准与保持；当前确认集已看，不能复用为新测试。
证据：[atomic_handoff](../evidence/EXP-004/2026-11-05/atomic_handoff/README.md)。完整成本、SHA、RAW/英文同步视频与论文表已归档。


## 2026-11-05 shared baseline update

[rule_option](../evidence/EXP-004/2026-11-05/rule_option/README.md) adds complete own-policy training and matched cost-sensitive supervision/rehearsal to the preserved SAC assets. New24:shared paired-data initialization19/18, online policy-gradient19/18, supervision18/19, rehearsal19, versus rule16. Retain these learned references; historical SAC/Q cannot be treated as calibrated for a changed one-deviation continuation. No independent RL superiority. Tool early continuation is repaired in `program_tool_recovery_v2.py`; old negative early-mode results remain diagnostic and are not a fair learned-method comparison.


## 2026-11-06 — actual joint continuation learning (adc86dc continuation)

Three joint learning iterations;448 own episodes;1163 total native episodes/116300steps/9857queries. MC mixing did not preserve13/16; shared execution critic7/16 vs flat9/16; paired whole-return PG and SAC both9/16. Frozen prior RL21/32 vs initializer20/rule19 is preliminary, not independent robust gain.

| Arm | Own episodes | Initial → middle → final |
|---|---:|---|
| identity_flat_32001 | 64 | 9 → 8 → 9 /16 |
| identity_semantic_32001 | 64 | 9 → 7 → 7 /16 |
| paired_pg_32101 | 32 | 9 → 9 → 9 /16 |
| paired_sac_32101 | 32 | 9 → 9 → 9 /16 |
| return_mix_31901 | 64 | 13 → 12 → 11 /16 |
| return_mix_31902 | 64 | 13 → 12 → 11 /16 |
| sac_31901 | 64 | 13 → 11 → 11 /16 |
| sac_31902 | 64 | 13 → 12 → 10 /16 |

Shared physical authority and frozen-final selection within each comparison. First revision has two new-network/exploration seeds sharing pretrained weights; later revisions single seed. New data are development/confirmation, not sealed tests. No pooled cross-cohort gain. Actual two-component rescue764712 is reproduced by either component alone. Action alias equivalence is physically verified but enforcing it does not improve policy. Whole-return PG avoids injecting fixed-suffix credit into changed continuation, yet matches ordinary SAC and mostly preserves initial behavior. No convergence or independent novelty claim.

Evidence: [evidence/EXP-004/2026-11-06/joint_continuation/README.md](../evidence/EXP-004/2026-11-06/joint_continuation/README.md). Next: Test omitted legal physical Options at real handoffs to determine whether the frozen two-proposal interface leaves repeatable recovery opportunity. Share any expanded authority across warm baseline and ordinary joint RL; do not expand unsupported MC-mix, alias-critic or paired-PG mechanisms. Preserve native Peg100 and sealed isolation.


## 2026-11-07 completed action-frontier learning cycle

Five128-episode own-policy learners (640 total) follow256 shared native design executions. Two genuinely independent ordinary SAC networks reach21/24 on development; fixed reference/offset coordinates20, state-conditioned target chart19, terminal supervised scoring16. The actual state-chart revision is trained, not merely proposed. Frozen new natural confirmation: chart_state_32201 19/32; old_rl 22/32; rule 20/32; sac_32201 19/32; sac_32202 22/32; supervised_32201 18/32; warm 21/32. Fixed final128 selection, no tuning after confirmation; native100 and official success unchanged.

| Fixed policy | Native success | Rescue | Regression | Planner calls |
|---|---:|---:|---:|---:|
| confirm_chart_state_32201 | 19/32 | 1 | 3 | 276 |
| confirm_old_rl | 22/32 | 1 | 0 | 266 |
| confirm_rule | 20/32 | 0 | 1 | 259 |
| confirm_sac_32201 | 19/32 | 1 | 3 | 267 |
| confirm_sac_32202 | 22/32 | 3 | 2 | 278 |
| confirm_supervised_32201 | 18/32 | 3 | 6 | 272 |
| confirm_warm | 21/32 | 0 | 0 | 266 |

Each comparison keeps real grasp, queue, Option termination and physical budget. Shared design cost is charged to each consuming learner; prior models have separately pinned historical budgets. Total cycle 1882 native episodes/188200steps/16027queries; all positive/negative models and raw physics retained.

The fixed catalogue has measured design union14/16 versus best fixed12 and warm8, but the learned21/24 remains21 when restricted to old proposals. Catalogue opportunity does not establish deployment dependence. State-conditioned kernel sharing has mature AGILE/action-representation precedents and no independent gain here. Frozen SAC Actor21, rear-only Mean-Q13, front-only Mean-Q19 isolate a deployment extraction problem without proving a particular replacement algorithm. No claim of joint synergy or STAP superiority. An83-dimensional observation and Option durations do not establish a Markov state or a policy-improvement theorem.

Evidence: [action_frontier/README](../evidence/EXP-004/2026-11-07/action_frontier/README.md). Next study should inherit ordinary SAC and investigate policy-consistent executed Option value on fresh training prefixes, not expand unsupported coordinate-sharing networks.
