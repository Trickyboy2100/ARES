# 研究决定与原因（追加式，不覆盖真实历史）

## 决策记法
每一项写明：`ID / 触发实验 / 真实证据 / 放弃或保留什么 / 下一验证动作`。
只是程序整理不算新的物理实验；不可把“研究计划”记成正结果。
本日志保存**方法问题的演化**，原始实验在 `EVIDENCE_LEDGER.md` 中保留引用。

### D-001｜收缩旧 P2 / 执行历史似然
- 证据：R3 32例普通PPO22/25/20，候选24/24/18；强反馈26。
- 结论：梯度确实改变，但没有独立性能收益；不能沿这一机制刷种子。
- 决策：保留历史可复现成果，寻找真实可控动作与公开任务对照。

### D-002｜放弃饱和StackCube作为唯一主任务
- 证据：新64例名义60，Residual63/62/64，参数PPO62/63/60；
  风险/paired选择没胜出。Pyramid 9候选网格固定59与oracle62/64。
- 结论：有名义和普通RL正结果，但可学习额外收益狭窄。
- 决策：强化对任务真实接触失效及有效动作的筛选。

### D-003｜Pyramid后续价值/风险辅助暂不扩训
- 证据：两真实边界普通PPO51/53/56，风险候选51/54/55；
  735风险token仅13正，后续开发30/32多方法相同。
- 结论：单纯增加值函数/风险头和后验历史没有证实额外价值。
- 决策：找到有真实作用且反馈可判别的接触任务。

### D-004｜PullCubeTool作为第二任务；Charger是候选主任务
- 证据：Pull新8例强几何前瞻7/8；Charger新8例固定2、几何2、
  监督3、FittedQ1、裕量辅助FittedQ1。Charger同后验状态参数
  50/75mm在单案例中造成22.05→3.66mm、失败→成功。
- 结论：Charger可能有可学习的执行自适应空间，但现有小数据价值
  估计和规划器随机性尚未解决；不能称已实现新方法。
- 决策：从强反馈、同动作重复、物理接触纠偏和成熟RL基线推进。

### D-005｜2026-10-31建立长期研究索引
- 证据：历史work_orders相互覆盖、旧README未指向当前任务；现有
  从08到13多轮成果包，没有一个可续接的活动实验/状态源。
- 决策：将 `research_program/STATE.json` 作为当前状态唯一来源；
  论文主问题和实验证据分开管理，Codex每轮只更新少量必要记录。
- 方法/实验新增：无。正向性能提升：无。

### D-006｜EXP-001：先修强接触控制，而非增加网络
- 真实证据：[2026-10-31物理闭环](evidence/EXP-001/2026-10-31/README.md)。18组跨进程成败一致；新8例同到位保持条件下原积分1/8、有界反馈5/8（救回5、退化1）。
- 原因：对齐未完成时推进、静差、到位后继续修正及对齐门限触发整段撤回均可造成失败；不能全部归为规划随机性或学习不足。
- 后续单因素修正：当前轴向保持在4个已看诊断初态2/4→4/4；不伪装为独立8例成绩。
- 保留最强控制器和所有负结果；未训练RL，独立方法主张仍UNVERIFIED。下一轮继续EXP-001的最终控制器独立确认，然后判断EXP-002是否有真实剩余动作空间。

### D-007｜EXP-001独立确认：强后段已覆盖，不能制造RL收益
- [新8初态+4次诊断](evidence/EXP-001/confirmation-v2-2026-10-31/README.md)：积分2/8、旧servo7/8、深度保持7/8；7个到达后段案例全成功，唯一共同失败是抓持。
- 新修正相对旧servo未增加成功，但共同成功案例首次原生到位平均快18.71步。旧725002/725005的撤回与接触前停滞得到时序支持；不证明稳定性理论。
- 事后两角度×两初态均失败：真实参数有物理作用，但无完整任务救回；抓持通过后IK失败也必须记失败。
- 不进入接触RL训练；EXP-001下一步聚焦抓持几何与后续IK共同可执行合同。只有强规则之上存在真实收益再启动EXP-002；未形成新RL贡献。

### D-008｜结束EXP-001的当前固定控制研究，推进学习空间发现
- 独立记录55861de：新8初态原积分2/8、有界servo7/8、深度保持7/8；
  7/7实际进入插入后段者全部成功，剩余一例是前段抓持失败。
- 已看725002/725005可以用深度保持规则去除轴向回退；
  其时序诊断不是已学RL机制，至今没有新的方法增益。
- 科学决定：EXP-001状态DONE（强控制实验结论完备），
  EXP-002状态READY并作为活动实验。主动考察官方新初态以及
  独立命名的真实工具/接触/感知条件变化，验证有效动作与在线
  效果反馈后可以立即进入EXP-003成熟RL，而不是持续刷8例servo。
- 视频联动工具代码已提交但未执行GPU渲染；完整物理轨迹
  继续保留非Git，明确来源和零新增实验。

### D-009｜EXP-002：接触偏差有动作机会，候选反馈特征尚无增益
- [真实物理、训练、消融与同步视频](evidence/EXP-002/2026-10-31/README.md)。新原生4例3/4，唯一失败为抓持；中等质量/摩擦变化12/12被强控制解决，不构成新的RL难题。
- 独立命名ARES-AdaptivePlug-v0观测偏差造成条件化动作排序。四个新scene×三偏差下geometry4/12、RLS6/12、search10/12；强搜索完整保留。历史native-oracle停止版本与当前observed-goal停止版本分开核算。
- 实际训练普通SAC和同历史效果特征SAC各1024技能决策/896次actor与critic更新。开发成绩1/12与3/12；同效果模型prior-only仍3/12且逐例同成败，不能以特征激活或弱SAC差值宣称机制收益。
- 同SAC checkpoint仅将Z参数置零后7/12：救回6、退化0。Critic在12/12首suffix状态却更偏好原actor动作；这是价值学习/动作表示问题的诊断，不是单动作反事实真Q或新增算法。
- 既有critic LayerNorm另作强基线。首轮误用默认alpha=.1，保留全部成本且不解释为单因素；另跑alpha=.02匹配对照。最终独立结果见证据表，不能挑选或合并两个运行。
- 决策：EXP-002首轮机会分析完成，EXP-003成为活动实验。先验证任务合理的Y-only/YZ参数权限、接触持续探测和终态保持，在新条件建立有能力的普通RL。暂不扩大当前效果特征网络；EXP-004独立方法主张继续UNVERIFIED。
- 同步视频已真实GPU渲染，逐帧/曲线/点击联动QA。它是MEASURED_POSE_REPLAY，零新增物理步；不是在线拍摄、实机或额外成功实验。

- 同熵系数的critic-LN普通SAC最终10/12（相对search两救回两退化）；额外严格匹配的effect+LN仅2/12，prior-only仍2/12。因此负结论不限于原弱优化器，但仍是单训练seed有界pilot；不将特征效益推广为普遍不可能。两例普通LN失败发生在原生成功后继续servo，始终未进入hold；下一步应物理验证可观测完成合同，不能用真值oracle直接停止。
### D-009｜2026-11-01 研究总目标回审：当前单技能接触不可替代多阶段论文
- 回读原始11_PAPER_FIRST_20261028的两技能边界论文：研究主题是
  Task-Conditioned Executable Skill-Parameter Adaptation；这条主线不变。
- b116457新ARES-AdaptivePlug-v0开发4 scene×3 bias中，
  search10/12、critic-LN SAC10/12、effect+LN2/12；两个LN
  失败有真实短暂成功后继续运动而丢失。都是开发证据，
  无多seed、非官方改物理分布、没有独立新RL方法增益。
- 科学决定：EXP-003保持活动，但拆成**A强RL/共同完成合同
  和B真实多技能可学习任务**。A允许简化只做必要强基线，
  B必须尽早恢复真实两技能参数决策。EXP-004方法候选不再
  锁死历史49假设几何兼容特征；允许依据B得到的物理信息
  选择一个有独立机制的算法。POMDP仅作满足部分可观测
  信息增益时的候选。强基线一旦可信应冻结，不能无限追求100%。
- 本次仅学术审核和研究计划/指引修改：新增训练/物理测试0，
  正式方法收益仍未证实。以后Codex自行在真实GPU工作树实施。

### D-010｜2026-11-01：强RL组件已训练，跨技能差异先经强后段控制检验
- [完整实测证据](evidence/EXP-003/2026-11-01/README.md)。三种普通SAC各4096决策/3840更新，实际checkpoint和同权限强搜索完整保留；共同观察完成合同不使用native成功oracle。开发小样本不支持泛化/原创增益声明。
- 三个原生多技能任务共77条/9200步，独立强后段几何TwoRobot8/8、同前段正交干预2/4。先前大量失败来自共享控制器/预算，不能归为新RL空间。Pull降速未消除弱抓持漂移，完善前段完成定义后再检验接触适应。
- 新候选为真实SkillResult分布与声明式后续合同辅助标签；仅提出未实现，必须击败同输入RL、同数据监督、STAP式模型搜索和Foresight式实际后段价值。没有原创性证据时明确不声称创新。
- 冻结插入训练扩量；下一步为八条件稳定抓持/负载配对和强在线自适应模型，保持原生成功与时限。无条件排序反转就调整物理任务，非机械Gate。

### D-011｜2026-11-01 选择STAP主对标，摒弃难以分辨的“伪创新”
- 独立审计df268c0源CSV：task LN-SAC对最强搜索35/36 vs32/36，4救回1退化，配对p=.375，精确6^6 scene-cluster区间[-5.56,+22.22]pp；对full SAC 5/1、对Y-only SAC 3/1。仅单种子六scene、一个用于工程调试。任务mask为三维SAC环境投影，未解决Actor/critic/熵的无效动作分量。
- 多技能B的77原生执行只有真实物理机会验证，未训练RL。Pull因抓持稳定/时限未验证条件参数排序反转；TwoRobot强后段8/8。没有基于该物理实验的原创RL正结果。
- 不再优先扩大单技能PlugCharger已有35/36开发集，不追求难以辨识的所谓首次发明；主对标固定STAP技能Q+Dynamics+CEM，在算法核高颗粒度验证后迁移到同任务；Foresight是下游结果次对标，RAPS提供参数化动作入口，DAWN提供成熟训练基础。
- 本论文最小可能增量来自不同真实后续要求下SkillResult/合同条件化参数学习和数据效率；必须对相同数据STAP模型搜索、普通同权限RL及Foresight式后续价值证明差异。无该证据则不声称新方法。当前只是审阅/索引更新，0新增物理步。

### D-012｜2026-11-01：STAP核等价运行；真实后验重规划可能退化，先加强基线覆盖
- [完整证据及交互审查](evidence/EXP-003/2026-11-01/stap_followup/README.md)。首先实际独立复算历史CSV，结论不变。固定官方STAP算法体与移植12组全候选精确匹配；旧包gym缺失，未声称整套benchmark复现。
- 新增224条真实100步物理执行、22400控制步；同64条数据训练绝对/增量状态两个模型版本，各2800更新，总5600。2个训练scene、1个训练seed；普通Fitted Q/Foresight类比均3/8，不是收敛SAC/PPO强基线证明。
- 严格同首参数和逐记录物理前段的STAP开环6/8，后验重规划3/8，3退化0救回。增量状态模型位置误差约89–175mm降到10–35mm，任务仍4/8开环、3/8重规划。没有模型精度→任务增益的证据。
- 强固定100mm/−65mm为4/8；确定性预算保护也4/8，修复一个184步/剩余61步的不良选择，但不是RL创新或预算保证。训练条件固定105mm/−55mm仍8/8，更宽目标未证明抓点排序反转。
- 科学决定：不训练额外合同网络掩盖稀薄数据。先用新训练scene有界覆盖真实后段动作，固定强STAP/普通RL/监督和几何基线；只检验一个“合同裕量辅助价值决定保留或修正参数”的增量。现有新scene已用于诊断，不作为封存测试。维持UNVERIFIED。两组完整RAW/ANNOTATED视频与同步trace均实测姿态重播，零新增物理。


## 2026-11-02｜真实SkillResult配对、数值修复与最小机制否证

[结果/关键图/同步视频](evidence/EXP-003/2026-11-02/skillresult/README.md)。保留历史，新增320真实物理回合/32000控制步、13600优化更新。96干预面板固定8/16、事后上限9/16，无严格目标排序反转。复算三例退化，区分低Q强制改参、预算与双成功合同。发现概率饱和选参并列，保留112诊断并以logits/log-Q另跑112确认。修正后ordinary1/8，generic/contract/relative/relative+contract均2/8；固定、强几何、STAP-open和3critic-UQ均3/8。组合机制没有独立增益；2个新测试初态已作开发诊断，未使用743052/53。公开模型/真实数据/四个完整视频及同帧曲线已归档。下一步优先原生多技能后继合同冲突的物理机会，不扩大窄hook网格或网络模块；保持UNVERIFIED。

### D-013｜2026-11-02：旧干预数据的独立再分析（新增仿真0）
按03c4176的train96原始结果分组：16个任务条件各6种合法参数，其中7组全成功、7组全失败、2组混合成败；混合案例为743041正向目标的抓点差异和743046正向目标的钩入差异。最佳固定8/16，已测参数oracle并集9/16，未显示跨目标的严格最优参数排序反转。这只否证当前狭窄参数条件中继续堆同类辅助头的收益理由，不能外推为后继条件学习普遍无效。保留现有强基线、真实SkillResult、STAP核与数值修复。下一次研究以现有Pyramid真实双技能后继要求为物理探索对象，先分析旧数据、再做有限干预，不改变密封测试及原生成功条件。


## 2026-11-02｜累计物理探索：后继朝向差异被强几何解释

[关键图、同初态完整视频与原始证据](evidence/EXP-003/2026-11-02/pyramid_contract/README.md)。先执行新增机会脚本，并核对96原始工具轨迹、576旧Pyramid终态及26历史轨迹SHA。新增90真实回合/22500控制步、0优化：单代表朝向pilot0/18；最近对称姿态29/36；四等价姿态预放置/放置可行性检查36/36。后两者36组前段记录完全相同，7救回0退化，属于确定性规划改进。强控制下2初态×2真实顶块朝向合同均9/9参数成功，固定与已测上限均4/4，无排序反转。保留SkillResult、对称可行性组件、原有RL/STAP与负结果；不在饱和条件训练。后续仅在真实布局/后继净空出现强规划后的不同物理参数偏好时训练。原生成功/250步不变，追加朝向合同另列adaptation variant；stride3及未建world-cloud碰撞约束明确披露。独立方法增益仍UNVERIFIED；密封测试未动。

### D-014｜2026-11-02：从强几何饱和转向实际执行残差与技能级时间价值

最新物理基座是Pyramid90条：nearest-symmetry29/36、full-feasible symmetry36/36，7个严格同前段rescue、0 regression、144次候选几何查询；它是需保留的控制器能力而非RL收益。新yaw二场景学习空间局部饱和，不外推到所有Pyramid任务。独立复核历史HISTORICAL_OPPORTUNITY.json：576物理、64scene中有效前段参数效应9、后段敏感13、交集6；420002/420023在已测动作组合中相对名义必须同时更改前后参数才能成功。工具96行14/16上下文二元成败恒定，不代表物体连续结果没有差异。此行是已保存数据/文献分析，**新物理0、优化0**。

学术先例补充Sequential Dexterity/ViSkill/Foresight（后继可行与后向训练）、TASP/SPIN（异构技能运动衔接与规划蒸馏）、SPIRE/DAWN（强基座）、EES（主动练习）、FAWAM（力反馈纠偏）、随机时长规划及Action Space Design；简单合同Q/已测状态/预算输入都不是独立创新。

决定：现有EXP-003 A/B两工作流不变；保留最强四姿态搜索、STAP核和稳定log-Q，先追踪历史少量真实两阶段敏感案例的力学原因；在实际后继净空/接近或工具接触漂移/限时中做有界同权限真实物理对照，要求Codex同会话自主推进下一项有辨识力的实验，不仅返回Gate。几何解决则记RETAINED，再寻找可测执行残差；有真正剩余学习空间才训普通两边界RL和最小新机制。理论审查、原始数据与运行建议见新[研究评估](evidence/EXP-003/2026-11-02/pyramid_contract/THEORY_REVIEW_NEXT_RESEARCH_LOOP.md)。独立算法增益继续UNVERIFIED，测试密封及ARES-R边界不变。

## 2026-11-02｜接触/时长累积闭环（contact_duration）

完成312有效物理episode/44700步，另8个FK日志API错误尝试800步单独INVALID保留；11000真实优化更新。旧Pyramid576审计与原初态counterfactual厘清C抓持/前段合同失败，原36/36几何资产保留。工具干预baseline3/12→measured/frozen refresh均5/12；严格前段一致36/36，反馈信息非必要。等时等待不充分，plan_screw Success仍有13–14mm终点FK误差；细积分下降但成功抓位交叉，不能宣称通用修复。108新设计有限网格最佳固定6/12、oracle7/12；新开发普通FittedQ、监督、STAP-open/replan、后续价值类比及固定均2/4，继承100mm参考也2/4。没有独立算法增益，UNVERIFIED。

永久保留：旧SAC/STAP/log-Q/36of36、两边界真实结果、只读100Hz接触与FK检查。下一轮先共享计划终点/实测到位/接触入口/预算合同，并用新设计物理配对验证抓位×后继调度残差，不扩训练量或叠模块。初始n2与原n16同seed不相同，54诊断条不可当旧case复现；另36条仅reset恢复记录初态，之后真实物理。详见[研究结论](evidence/EXP-003/2026-11-02/contact_duration/README.md)、[审查页](evidence/EXP-003/2026-11-02/contact_duration/REVIEW.html)。


## 2026-11-02｜共享执行修复与真实双边界 SAC（contract_sac）

[研究结论](evidence/EXP-003/2026-11-02/contract_sac/README.md) · [审查页/完整同初态视频](evidence/EXP-003/2026-11-02/contract_sac/REVIEW.html)。596完整真实物理回合/59200控制步，另76已保存部分步及首次崩溃未知0–100步。FK计划误差降低不等于实际到位；stroke2×2证明3/8→6/8由合理行程权限解释，无需后验反馈；实际起点微重规划6/8→6/8。接触前重载抓持漂移与短期稳定不矛盾，统一100Hz力/20Hz姿态定义。

恢复真实两边界SMDP SAC：保留575已完成决策、1022实测更新、SAC256/512及中断模型/缓存；645条转移物理审计误差0。第一次67回合TOPP崩溃保留原数据，不能假称恢复丢失actor。相同323回合数据训练模型/监督/后继价值10000更新。新8开发初态，固定70/105、SAC256/512、监督、STAP-open/replan、后继价值全部7/8。共同失败746204为TRANSFER规划失败，STAP概率约0.00019；不是高Q误导。1训练seed与有限检查点，不宣称收敛/已充分RL。原有A线、Pyramid36/36、STAP/log-Q保持。

36高度网格无额外后继二值机会；36加速度配对无救回且一次预算退化；拓展yaw一次退化；翻转夹爪未救回两个共同失败。原生PokeCube50步标准/直接接近各0/4，已实际追查预算与抓取，不能当弱基线制造RL收益。下一具体问题是强规划下剩余可达路径/中间工具姿态与实际跟踪/接触预算；第二任务先建立可信共享控制器。所有正负模型/物理/成本/SHA公开仿真归档，密封数据不动。独立算法增益 **UNVERIFIED**。

### D-016｜2026-11-02：两边界SAC恢复后确认原动作接口的局部不可辨识性（原始数据审计）

来源：最新contract_sac/analysis/SUMMARY.json的596条完整原生物理记录与program_tool_smdp_sac.py/program_stap_tool.py源码。本项无新增物理或模型更新。可重复审计脚本：scripts/audit_smdp_action_authority.py（需要Codex在Ubuntu仓库执行）。

四个独立物理初态×3抓点×3实际hook-height：在九个具有有效后继动作的(初态,抓点)组内，三种26/32/38mm钩入高度均没有改变二元完整成功；另3组第一技能失败、没有有效第二阶段干预。加速度3/4.5/6有九个有效(初态,抓点)组，仅746001/70mm一组二元结果不同（较低加速度因剩余时限失败）。学习器在目前动作网格下主要是在选择抓点；不能仅因SAC接口有两次调用就说存在有意义的两阶段RL收益。

新开发746200–746207各有8种策略实跑，7组8/8全成功、746204八方法全失败且统一PLAN_TRANSFER；零方法可区分的开发条件。后者额外6条wide方案亦均失败，但不证明其他合法接触/路径都无解。287恢复SAC训练episode覆盖64物理初态，有24个在策略两阶段动作共同变化时成功/失败混合，这提示前段参数真实可控，但不能以非严格同首参数把作用归到第二阶段。训练只有mass1.0和单策略seed，不支持任务条件适应泛化结论。

**决策：** 继续无限修一个746204案例即使达到8/8也只是强基线饱和，不能形成方法增益。把该问题降为一次有界公共路径可行性诊断并保留其规划器修复；**主研究立即改为在真实物理事件之后给RL有效的合法控制参数**（首接触后物体相对运动、接触保持/拉动位移方向与幅值、剩余真实步数），所有方法均拥有同接口，先证实实际行动权限。若PullCubeTool100步无有效接触后再决策窗口，优先在ManiSkill现有200/250步的多技能任务上建立可信强名义，不能延长官方horizon冒充原生。与HACMan++ contact-grounded primitives、Fabrica planned residual、RecoveryChaining recovery options、SCOUT动作效果信念直接对标，不能把contact event/belief/residual换名宣称创新。保留SAC/STAP、所有原始成本、实机禁令和sealed约束；论文方法独立增益仍UNVERIFIED。


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


## 2026-11-04 — Repeated-feedback skill and budget-conditioned handoff

[Evidence and complete review](evidence/EXP-004/2026-11-04/closed_loop/README.md). From616099e, implement an actual repeated-feedback successor after ALIGN (goal-frame target corrections, measured attachment, withdrawal and phase-bounded hold), retaining the three-grasp predecessor, strong planner and native100. Same-state zero feedback exactly matches original pilot trajectories; repeated parameter IDs preserve active queues. CQL/BC warmstart is shared mature infrastructure.

Actual matched finite SAC / scalar suffix-value / budget-completion-CDF training: after96 native episodes9/16,10/16,8/16; after~10k separately costed reachable-handoff training steps5/16,4/16,6/16. Retained old SAC10/16. A further same-budget deployment-policy-label branch yields9/16,7/16,7/16. Lower prediction error is not policy superiority; budget hazard Brier0.273→0.063 is measured on different own-policy rollouts and is not a calibration-only causal claim. No independent gain.

Six local-training recovery candidates were rerun from native starts with exact physical prefixes: zero6/6,rule6/6,mined schedule4/6. Thus no new beneficial operation capability is yet established. In five targeted regressions, keeping each final learner's actual grasp but removing suffix corrections changes ordinary1/5→5/5 and budget2/5→5/5. This identifies harmful suffix intervention in those cases. Native `grasped=True` can coexist with~85mm TCP-relative drift; do not assume fixed attachment.

Retain closed-loop code, trained checkpoints, local-training initialization, physical accounting and all failures; do not retreat to fixed-branch scoring. Budget hazard and deterministic-label revision are falsifiable candidates, not original contributions. Continue stable successor policy development and controlled physical observation/action studies before expanding this handoff target. Current64 observations omit object twist/contact forces; aliasing remains a hypothesis.

Archived1431 full100-step trajectories plus7256 prefix and32640 initialized suffix steps (182996 total) include development/compatibility work and all completed physical training/evaluation, not independent samples. CUDA700 evaluation interruption resumed from saved model/optimizer; exploration RNG restart disclosed. A SIGTERM attempt was retried without overwriting evidence. Unarchived partial interrupted steps make attempted cost a lower bound. RAW/English101-frame video, synchronized curves, source/model SHA and all negative models retained. ARES-R stayed read-only; no hardware/sealed tests. **UNVERIFIED**.


## 2026-11-04 — Successful-behavior retention and controlled exploration

[Actual training, full native results and review](evidence/EXP-004/2026-11-04/success_retention/README.md). From a2fa4a6; preserve the feedback runtime, three-grasp interface, all old models and negative evidence. Separate BC from CQL with exact-default-update checks. Temporarily fix real grasp1 and train successor-only samples; native100, eight actions and strong planner unchanged. Action0 continues the active queue, never restores the original nominal path.

Four32-episode physical-training branches all start/end9/16; another4000 actual-replay updates give removed8/16, persistent-BC9/16, controlled-sampling10/16, combined9/16. Controlled SAC rescues753204/753212 but loses753209. Persistent BC is not sufficient: successful episodes also credit exploratory actions. Two same-parent/same-replay source ablations show whole-episode nominal-success anchors9/16 with0 rescue/0regression, versus all-success anchors8/16 and10/16 with regression. Lowering nominal-anchor0.2→0.05 yields9/16 with1/1; do not turn this into endless coefficient search. These are mature BC/exploration/anchoring controls, not original algorithms.

Measured failures are late handoffs73–86 versus nominal-success46–62. Parameter4 changes actual refinement geometry and duration; these causes are not yet isolated. A posthoc same-authority phase/budget rule achieves12/16 (3rescues/0regressions), while still failing the learned rescue753204. This is a new reusable closed-loop CONTROL capability on seen development, not RL novelty or unseen validation. Learners remain below this stronger reference. Nine native repeats reproduce complete100-step trajectories exactly.

Costs:505 full native episodes/50500steps, including128 training episodes/12800steps on32 distinct training starts;377 evaluation/repeat episodes on16 seen development starts;1336 effective successor training decisions;41120 primary actor/critic iterations;3841 planner queries. One inherited neural initialization. All models, raw trajectories, source/protocols, cost scopes and verified SHA archives retained, with RAW/English rescue AND regression videos and exact-frame curves. No sealed tests, hardware or ARES-R changes.

Retain the strong feedback rule, ordinary controlled-SAC endpoint, nominal-anchor retention endpoint and all failed variants. Next use actual stronger-reference training trajectories as shared data to test value grounding/retention against this stronger comparator, without changing action permissions or adding probability heads. Delay joint predecessor improvement and new evaluation expansion until learned successor improvement is stable. Independent method gain remains **UNVERIFIED**.


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


## 2026-11-05 — rule_option: learned adoption retained; waiting-credit hypothesis not engaged

From9a30507, implemented and ran single-complete-Option deviation then frozen strong rule. New64 paired training conditions give3rescues/7regressions; all7 harmful outcomes have positive old behavior-Q gaps. Completed512 own-policy training episodes across two iterations, two independent initializations and actual entry-credit/rehearsal controls. Shared500-update pretraining underfit the three positive training examples; common2000-update initialization was corrected before further matched training.

New24 fixed-final64 confirmation: rule16, fixed first proposal16 (4rescues/4regressions), shared initialization19/18, sequenceRL19/18, supervision18/19, ordinary rehearsal19. Each RL model has exactly its initializer's per-scene success outcomes. Thus paired-data conditional adoption is retained as an effective learned baseline; independent online-RL gain is not supported. All nonzero updates in the ablation seed occur at the first gate decision, so full waiting credit and entry-only credit produce identical112 complete training/development trajectories. Do not claim a new stopping-time algorithm from this comparison.

Tool's early-contact adapter had a real continuation defect: declined intervention discarded the live nominal caller, and early modes could intervene before the event. Fixed in a new version without overwriting history. Eight nominal100-step traces and three declined-intervention traces are exactly retained. Corrected early none/feedback/press=3/3/2 of8,0rescues; no second-task RL gain or blind transfer training. Whole cycle:1528 native episodes,152800physical steps,14644planner queries,40named weight checkpoints. Detailed evidence, actual scripts, all models and rawSHA: [rule_option](evidence/EXP-004/2026-11-05/rule_option/README.md).

Decision: retain the stronger learned initializer and rehearsal controls, execution contracts, own-policy data and Tool repair. Stop expanding the inactive waiting-credit mechanism. Next investigate real predecessor–successor ordering on NEW training initial conditions with this retained learned successor, then a matched ordinary two-stage learner; never reuse the current confirmation as unseen data. No independent algorithm or formal safety claim.


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

Evidence: [evidence/EXP-004/2026-11-06/joint_continuation/README.md](evidence/EXP-004/2026-11-06/joint_continuation/README.md). Next: Test omitted legal physical Options at real handoffs to determine whether the frozen two-proposal interface leaves repeatable recovery opportunity. Share any expanded authority across warm baseline and ordinary joint RL; do not expand unsupported MC-mix, alias-critic or paired-PG mechanisms. Preserve native Peg100 and sealed isolation.


### 2026-11-07 initialization erratum (historical scores unchanged)

Direct checkpoint comparison shows joint_continuation sac_31901 and sac_31902 INITIAL Actor, Critic and target tensors are identical: the shared Gate constructor resets Torch RNG to31801 after the requested seed. The two runs have distinct exploration RNG, **not independent neural initialization**. Earlier independent-new-network wording is superseded. Their actual outcomes and negative evidence remain unchanged. The next full-catalogue learner reseeds after loading shared assets and verifies different hidden/Critic weights while keeping cross-arm initialization matched.


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

Evidence: [action_frontier/README](evidence/EXP-004/2026-11-07/action_frontier/README.md). Next study should inherit ordinary SAC and investigate policy-consistent executed Option value on fresh training prefixes, not expand unsupported coordinate-sharing networks.


## Post-confirmation mechanism check: deployed action support

Both frozen SAC seeds are reexecuted on all32 now-seen confirmation scenes with only current historical proposal physical IDs plus rule allowed. No model, horizon, reward, planner or initial state changes. These64 additional native episodes test execution dependence, not new independent generalization or a retrained restricted-learning comparison.

| Network seed | Full catalogue | Old proposal support | Full-only successful scenes | Restricted-only successful scenes |
|---|---:|---:|---|---|
| 32201 | 19/32 | 20/32 | 765625 | 765608, 765626 |
| 32202 | 22/32 | 21/32 | 765625, 765627 | 765604 |

The full-policy benefits and regressions must both be retained. Any full-only success supports necessity of the extra executed choices for this frozen policy on that state, not impossibility of every other old-support policy. This does not turn the failed coordinate candidates into an independent algorithm contribution. Exact physical prefixes, raw paths and complete boundary events are in POSTCONFIRM_SUPPORT_EFFECT.json.


### Evidence-based retained capability and unresolved reliability

The frozen second SAC network uses an additional localZ− Option to rescue765625 and765627. Removing expanded action support loses both; one harmful refreshedY− intervention in765604 is also removed. Full22/32 versus restricted21/32 is therefore a capability/retention tradeoff, not robust dominance. The first seed similarly gains765625 but breaks765608/765626 through localZ+, yielding19 versus restricted20. These are post-hoc same-model physical controls on now-seen conditions, not a new independent test. Ordinary SAC owns these gains; neither coordinate candidate earns credit. The task interface has a measured new deployed capability, while reliable selection is still unresolved.
