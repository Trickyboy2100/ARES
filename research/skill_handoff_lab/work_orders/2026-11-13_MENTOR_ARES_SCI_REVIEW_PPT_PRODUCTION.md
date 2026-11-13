# ARES 累计强化学习研究汇报：导师审阅／论文讨论 PPT 制作任务书

> **性质**：只读研究证据整理与学术汇报制作；不属于新的 EXP 算法实验。  
> **执行仓库**：`Trickyboy2100/ARES`，研究分支 `research-sci-program`。  
> **事实快照**：2026-11-08，提交 `55ff6333886b8724a993d85c94d8c2e7467c5206`；开始前复查当前 HEAD，较新记录如已验证可另附更新，不悄悄改写旧结果。  
> **目标受众**：科研导师；用于汇报甲方 SCI 方法论文的累计工作、真实证据与发表差距，同时支持课题组后续独立研究方案讨论。  
> **第二份必读文件**：[2026-11-13_MENTOR_SCI_PROPOSALS_AND_QUESTIONS.md](2026-11-13_MENTOR_SCI_PROPOSALS_AND_QUESTIONS.md)。  
> **本任务的结果**：可编辑 `.pptx`、逐页 `.pdf`、讲稿/引用索引、复用图表、可追溯媒体索引/必要离线媒体包与质量复核。**不是让 Codex 先写 PPT 计划或单独交付 Markdown。**

## 1. 学术定位与叙事

主线：任务/工件条件 → 已有规划与名义技能 → **强化学习选择可执行技能参数** → 真实物理 SkillResult → 反馈条件化后续技能决策 → 完整任务成功/学习成本。展示从工业机器人模型与规划基础、真实接触在线 RL、公开基准、物理技能交接，到离散 Option/SMDP、持续策略改进的完整历程。强调**已完成工作的积累和可发表证据还缺什么**，而不是只展示最新失败。

必须在主汇报可清楚区分：
1. **工程/接口/控制修复**：URDF/夹爪、IK/cuRobo、PhysX执行、约束与规划基线；不记作新算法收益。
2. **真实 RL 能力及标准基线结果**：实际训练过的 PPO、Residual PPO、SAC、离线/在线 Option 学习，明确成功率、种子和比较范围。
3. **跨技能物理依赖与阶段性正发现**：真实前驱参数影响后继完成；已有普通 RL 救回和动作集能力，但存在退化。
4. **独立方法增量探索与反证**：阶段回报混合、配对风险、效果历史、价值回传、策略续接、Retrace、参考保持等，有实现、有真实比较，但目前**独立稳健增益尚未建立**。
5. **甲方 SCI 与课题组延展**：均必须真正包含强化学习；候选问题与方法属于讨论提案，未作为既成创新。

每个研究结果注明任务名、平台、原生/附加成功标准、实际scene组、N、seed、控制权限、对照方法、结果及证据出处。不能把 `Panda/ManiSkill` 的结果说成 `JAKA` 实机或金隅现场结果。

## 2. 必须继承的源材料（先核验，不重复制作）

### 2.1 已整理的 2026-11-13 研究素材包（如本地可获得）

先在用户提供的工作目录和 `inputs/` 下查找 `ARES_research_review_20261113.zip`（以前的聊天交付附件；**ChatGPT sandbox 路径不等于 Ubuntu Codex 本地路径**）。可解包到本项目的临时工作目录。包内：
- `00_阅读入口与研究总览.md`；`01_PPT逐页文字与素材说明.md`（**36页内容底稿，不是固定的最终页数或必须照搬的排版**）。
- `02_完整研究台账.md`（44项）、`03_方法与基线对照.md`（31项）、`04_图表与视频索引.md`。
- `05_论文证据与未解问题.md`、`06_完整数值与成本口径.md`、`07_来源索引与勘误.md`、`08_代码模型与复现资产索引.md`。
- `data/` 下的 result_panels、results_long、slide_contents、media_manifest、sources、snapshot 与 QA 结构化记录；`figures/F01–F16.png|svg`；`tools/sync_source_media.py`；`ARES_研究资料审阅.html`。

**若包不在本地**，不能停在“请提供资料”。以 2.2–2.4 的仓库固定版本和真实原始素材重建，明确在交付说明中记录“未能读取历史汇总包／重建了哪些内容”。不把汇总包视为权威证据，**原生运行、原始结果、SHA及项目原始文档高于转录汇编**。

### 2.2 权威研究总控和旧研究合同

先读 `research/skill_handoff_lab/AGENTS.md`、根部适用的指令，随后读：
- `research/skill_handoff_lab/research_program/README.md`
- `STATE.json`、`PAPER_CONTRACT.md`、`EVIDENCE_LEDGER.md`、`DECISIONS.md`、`PRIOR_ART.md`
- `paper/CLAIM_MATRIX.md`、`paper/MANUSCRIPT.md`
- `research/skill_handoff_lab/work_orders/2026-11-08_LONG_HORIZON_SKILL_RL_AUTONOMOUS_RESEARCH_DIRECTIVE.md`
- 各轮证据路径中的 `README.md`、`RESULTS*.csv|json`、`REVIEW.html`、`MANIFEST*.json`、配对/预算/源码索引。

尊重 `STATE.json` 的研究状态；**本次 PPT 工作不能覆盖或重定义 active_experiment，也不能通过写 PPT 修改“独立方法增益”结论**。如没有新的真实科学结果，不改 `STATE.json`。

### 2.3 历史研究素材（不能被最新 Peg 结果取代）

| 领域 | 仓库原始入口 | 汇报应呈现的事实 |
|---|---|---|
| 工程与真实模型/规划 | `research/skill_handoff_lab/README.md`、`reports/P2_1_SCIENCE_DECISION.md`、`reports/SERVER190_AND_CASH_SUMMARY.md` | JAKA/夹爪/URDF/USD、cuRobo、候选合法性与CASH V0接口；工程演示和CPU代理并非RL |
| Isaac 自建接触任务 | `reports/CONTACT_COMPOSITION_RESULTS_20261016.md`、`reports/CONTACT_HANDOFF_RESULTS_V6_20261016.md` | 16环境PhysX、v5 229,376交互、真实普通PPO/阶段回报结果；v6未见几何强反馈8/32，Residual PPO 13/18/23；P1/P2分层 |
| v7/R1–R3 | `deliverables/client_sci_20261027/00_README_EVIDENCE.md`、`07_MENTOR_BRIEF_CN.md`、`08_ITERATIVE_RESEARCH/` | 实际Actor更新、R3普通22/25/20，候选24/24/18，强反馈26/32；无稳定方法增益 |
| StackCube | `deliverables/client_sci_20261027/09_BENCHMARK_PIVOT/`、`10_METHOD_EVOLUTION_20261028/` | 新64：名义60；Residual PPO 63/62/64，参数PPO 62/63/60；权限/更新频率不全相同 |
| StackPyramid | `deliverables/client_sci_20261027/11_PAPER_FIRST_20261028/`、`12_SCI_OPPORTUNITY_20261029/` | 连续两真实技能边界、真实松爪/稳定；开发PPO [51,53,56] vs 风险候选[51,54,55]；64设计最佳固定59 vs所测并集62 |
| Tool / Charger | `deliverables/client_sci_20261027/13_TOOL_TASK_OPPORTUNITY_20261029/` | 抓持/钩入和插接参数的真实物理作用，强几何/强伺服竞争解释；不要把单案例或改控制称RL增益 |
| STAP与多任务 | `research_program/evidence/EXP-003/2026-11-01/stap_followup/`、`2026-11-02/pyramid_contract/` | STAP算法核功能适配、后验重规划反例；更准预测不等于成功率改善；强几何面板36/36 |
| 离散 Option | `research_program/evidence/EXP-004/2026-11-03/discrete_handoff/`、`2026-11-04/value_improvement/` | 连续训练→离散执行支持集错位已诊断；成熟离散学习器与actor/minQ/meanQ提取区别；并无独立方法优势 |
| 闭环策略 | `research_program/evidence/EXP-004/2026-11-04/success_retention/`、`intervention_credit/`、`continuation_option/` | 有救回也有退化；757300首次动作有效但后续可破坏；Option续接语义已真实修复 |
| 2026-11-05～07 | `research_program/evidence/EXP-004/2026-11-05/`、`2026-11-06/`、`2026-11-07/action_frontier/` | 真实前驱/后继联合、状态参考解耦、成功监督初始化与在线RL贡献分离、动作集扩展后普通SAC22/32 |
| 2026-11-08 | `research_program/evidence/EXP-004/2026-11-08/continuation_retention/`、`suffix_trace/` | 强固定24/32 vs新SAC17/17/32；新24：固定19、旧策略16、新SAC15、trace15、lagged14；769401首动作估值错误 |

上述数字用于**提醒复核**，作图时必须回溯对应的 source/RESULTS；不能把不同轮的分母和“新自然初态”混成一个共同测试集。明确区分 Pyramid 设计集名义52/64 与另外的开发集名义53/64；Pyramid Oracle 是**有限已执行动作事后并集**，非可部署 policy。

### 2.4 原始视频、物理轨迹、图片（优先已有，禁止虚构）

优先按历史素材包 `04_图表与视频索引.md` / `data/media_manifest.json` 寻找 MP4、PNG、轨迹和原始SHA。若包不在，则从上表原证据目录查找 `RAW.mp4`、`ANNOTATED.mp4`、`SYNC_VIEWER.html`、`TRACE.png`、`MANIFEST.json`，并查本机 `runs/` 下源数据。

必须优先放入**真实科研对照**：
- Stack同初态救回 `200003` **与**退化 `200006`（`10_METHOD_EVOLUTION_20261028/media/PAIRED_stack_*.mp4`）。
- Pyramid前段参数救回 `122007` **与**退化 `122005`；说明前驱物理后果影响后续。
- PullCubeTool抓点 `712000`；Charger同前段状态选择不同后继参数 `721000`（`13_TOOL_TASK_OPPORTUNITY_20261029/visuals/`）。
- STAP `stap_followup/video_replan_regression/`，开环成功、实测后验重规划退化。
- `intervention_credit` 的 `757300`及`suffix_trace` 的 `769401` 机制结果**分别验证视频与真正场景编号**。`suffix_trace/video_769400/ANNOTATED.mp4` 表示769400，**不可错当769401的机制对照视频**。
- 包内 `F01–F16`（若来源匹配）与原repo的科研矢量图、训练曲线、原生物理帧，优先复用；更适合再绘的图可根据原CSV/JSON重画，附字段口径。

原 MP4 如缺失可使用原数据保存的静帧或有SHA的真实状态回放，且需标注 `MEASURED_STATE_REPLAY`；不能把视频回放计为新增物理试验。禁止图片生成工具替代真实实验画面；不使用与实验无关的网上机器人照片，不复制论文图片作为自有实验。

## 3. 指定 PPT 模板：必须使用 AXIOM 中的可复用 POTX

**唯一优先模板**：`Trickyboy2100/AXIOM` 私有仓库，`main @de9c0361cb59e55fa428a967bd62727e384051a6`：
- `99_templates/academic_presentations/北工大学术汇报模板_细进度条.potx`（Git blob `c402758662b86c5257e4148e6282b27fbdbbf92d`）
- 同目录 `北工大学术汇报模板_细进度条.pdf`（15页视觉预览），`README.md`。

优先只读本机AXIOM或已授权的GitHub连接，必要时使用用户已有的本地模板拷贝。私有仓库不可访问时，不伪称读取了模板；可先构建内容草稿和可编辑slide，但最终提交前必须用真实模板重排/验证。**不修改 AXIOM main 或其原始模板。**

视觉要求：保留北京工业大学校徽、青色标题线、原先无外框细进度条式底部章节导航、页脚文献/来源区；不要改回放大加框的进度条样式；沿用学校配色，克制、专业、适合学术导师讨论。建议对应模板7个章节主题：①论文目标和证据纪律；②系统与接触RL工程基础；③公开基准与多阶段任务；④强化学习算法实现与演进；⑤真实结果与关键反例；⑥SCI证据现状；⑦Proposal and Questions。页脚当前章节高亮应正确、可编辑。可根据文字长度新增版式，但不能直接将36页MD逐页塞进模板而导致拥挤；所有正文尽可能保持可编辑。

## 4. 内容架构与可选页纲（允许按信息密度重排）

**建议正文约28–34页、附录约8–15页**，以事实完整、可讲清为准，不机械凑页。不预先生成多个版本再无止境修改。关键页：

| 章节 | 内容组 | 必须有的视觉/证据 |
|---|---|---|
| A 研究目标 | 标题；甲方SCI对象；研究演进时间轴；证据类型与成功指标 | 研究图、真实/提案标签、可追溯来源 |
| B 工程与首次在线RL | JAKA/双臂/规划资产；CASH候选；自建接触P1/P2；16环境训练与v5；v6未见几何；v7/R3 | 至少1张真实训练截图、两张三seed与P1/P2分层图，展示正负结果 |
| C 公开任务与物理依赖 | StackCube；Pyramid双真实边界；576分支机会；Tool/Charger；STAP开环vs重规划 | Stack救回/退化一组；Pyramid参数因果对照一组；跨任务比较图 |
| D RL学习机制 | 离散执行/Option合同、SMDP实际duration、Actor–Critic对齐；在线/离线数据；Success retention；757300 | 完整语义图、真实物理步/队列箭头、配对结果 |
| E 最新算法结果 | 11/05 supervised初始化分离；11/06联合学习；11/07动作权限；11/08参考保持、soft Retrace；769401 | 纵向方法台账、同一cohort逐方法对比与两个**不同**失败机制 |
| F 证据现状 | 成立的事实/被否定的方向/尚未验证的主张；同任务候选基线；论文完成条件 | 论文主张—证据矩阵，别让“UNVERIFIED”遮住已完成真实工作 |
| G 导师讨论 | 2条论文路线的对照；2–3个可反证Proposal；机制与理论近邻；优先实验与可退路；明确导师决策问题 | 见第二份工作文件，所有方案均标记“讨论提案/未验证” |
| 附录 | 全量数值/成本、源SHA、实现模块、不同评测合同、抽样/种子更正、原媒体清单 | 导师追问时可即时展示、不要把所有计数挤入主图 |

**宁愿减少冗余过程，也不能遗漏早期v5/v6的标准RL正结果、R3负结果、Pyramid真实双边界、Tool/Charger、STAP，以及11月迭代的关键发现。**

正文必须有一页**两条研究路线区分**：甲方SCI的一篇RL方法论文（延续原科学合同，不把工程优化称创新）与课题组长期多技能/长程具身智能研究（待选任务/机制/新证据）。共用工程与算法资产≠两个已经成立的论文贡献，也不预设VLA训练或实机成功。

## 5. 学术与数值纪律：制作时强制执行

- 每张数据图注明：任务、场景组/设计或开发、样本量、不同训练seed、所用基线、原生或附加合同、控制版本。不能把 `X/6`、`X/8`写成稳健期刊结论。
- Rescue／regression 按同样初态对同一明确参照；各cohort独立报告；图例区分训练、开发、已看确认、封存。已看开发集与 sealed 不等价。
- 统计不确定性如果源未报告就不能擅造；只在有原始配对索引时重新复算并注明计算方法。旧源更正（例如两个seed实际相同初始化）须反映在主图或脚注。
- 模型预测误差降低、动作幅度、Q值、梯度测试不是完整任务性能；几何/控制修正不是RL；有限oracle并集不是可部署policy；某个case强制动作成功不是学习器得分。
- 将 757300（学会首干预后仍可能被后续重规划破坏）与769401（训练策略首次选错，强制首动作后训练/旧后缀均成功）做成**一正一反的两个并列机制案例**，不统一宣称“续接毁坏”。
- 11/07普通SAC同确认32可达22/32；11/08不同确认32固定24/32 vs新候选17/32；最新不同24固定19 vs先前16／新15。不同分布**禁止画成“连续退化/进步曲线”**。
- 相关理论：Option/SMDP、参数化原语、SAC/PPO与受限动作、后继可行性/价值、恢复与闭环、部分可观测性；**Option/参数化/下游价值/BC/KL/QP/Trace本身均已有前例**。POMDP只在真实观测别名被验证时考虑。
- 不得声称已完成新算法投稿、已达到SCI质量线、已有第二任务独立RL迁移或JAKA实机方法有效性；可以展示已写工作稿和可直接用于Problem/Experimental Setup的真实证据。
- 所有视频保持真实原始时序；裁剪、拼接、圈注仅为展示，视频旁标注来源与是否MEASURED_STATE_REPLAY；不得把补帧、截图或渲染增加为新物理步数。
- PPT主体和备注保持**第三人称、非人称的学术陈述**：不写“我们”“我的”“老师您”“我觉得”；Proposal只使用“拟研究”“待验证”“可检验假设”等。

## 6. Proposal 与 Questions 的位置

必须直接读取同目录 `2026-11-13_MENTOR_SCI_PROPOSALS_AND_QUESTIONS.md`，在主汇报最后设置**4–6页**讨论模块，内容包含：
1. 由已知原生失败导出的待解决**科学问题**（不是笼统“继续调参”）。
2. **甲方SCI短期方案**及**课题组长期方案**，分别声明任务/算法/验证范围，不能将两篇视为当前成果。
3. 2–3个具体候选机制（研究依据、最邻近文献、可证伪对照、关键风险、淘汰条件）。
4. 优先任务选择的真实证据：Pyramid存在跨技能耦合但旧网格饱和；Charger有前驱交叉但后继9次合法干预未救回；TwoRobotPickCube强完整后段8/8的面板覆盖；`TwoRobotStackCube-v1` 已查到官方源代码，但**截至此快照没有该任务可训练的ARES完整执行器/实验结果，不能展示为已完成研究**。
5. 对导师有决策价值的问题（期刊定位、原创性主张、主任务/第二任务、主要度量、感知/实机要求与资源分配等）；只列关键可决策问题，不把所有未解研究问题堆成模糊清单。

**提案页不需预填“创新已经成立”，也不应写“所有历史均无结果”。**

## 7. 媒体嵌入、可编辑性和交付结构

主交付目录：`research/skill_handoff_lab/deliverables/mentor_sci_review_20261113/`（如当前branch出现同名目录，保留现有并另建v2，不覆盖）。

至少交付：
- `ARES_SCI_MENTOR_REVIEW_20261113.pptx`：真实 .potx 派生、可编辑文字/表格/流程图；小体量视频缩略图配链接，必要时用原MP4离线相对路径；不能把整个幻灯片栅格化。
- `ARES_SCI_MENTOR_REVIEW_20261113.pdf`：逐页可阅读审阅版，版式与PPT同源。
- `SPEAKER_NOTES_CN.md`：逐页完整说明（核心事实、场景组、对照口径、理论/不确定性、讨论问题），不代替实际PPT；如可写入PPT备注同时写入。
- `SLIDE_SOURCE_MAP.csv`：一页一行，页码、结论、源路径、Git commit/运行SHA、实验或提案、媒体、真实性/许可证注记。
- `MEDIA_MANIFEST.csv`：每段视频原始路径、SHA（已记录则校验）、场景ID、原生/回放、截取范围、失败/成功、离线可达性、在PPT哪页。
- `FIGURE_SOURCE_MAP.csv`、`REVIEW_QA.md`、`BUILD_README.md`及可重建的制作源（如 .py/.js/markdown）。
- 可选独立 `media_offline/`：供离线播放的少量MP4，按SHA与许可证检查；不要重复复制整个runs存档或敏感现场素材。大型原始运行留原处，提供哈希/路径索引。
- 单独 `PROPOSAL_AND_DECISION_QUESTIONS_CN.md`：与PPT提案页严格一致，作为会后研究决策记录底稿。

封面/页眉应明确“2026-11-13整理，证据截至2026-11-08”；新验证（如果用户后来添加）另行标清版本。不要把“导师讨论提案”附在已完成实验页里模糊边界。

## 8. 执行步骤与最终验收

1. **只读盘点**：先检查 ARES 当前branch/dirty、更改权限、AXIOM真实POTX与本机媒体；列简短source map。不清理用户文件，不改ARES-R/AXIOM；不启动新训练，也不碰GPU物理任务。
2. **重核关键统计**：用源csv/json回对主图和极容易混淆的三类数字（v6 13/18/23 vs 8、Pyramid52/53两组、11/08两个不同32/24cohort）；异常记录在QA，不静默改写。
3. **制作第一版完整PPT**：真实模板、7章节导航、早期到最新、完整正负与阶段演进；不少于约6张具有原始结果的学术比较图，至少3组有效物理案例对照，包含失败/退化。
4. **补Proposal/Questions**：按第二任务文件形成讨论部分，确保每页有前提、假设/决策、既有近邻与研究成本/退出条件。
5. **媒体核验**：离线视频仅在真实源文件存在且校验时声明可播放；本机找不到时用实际源图/截图与固定链接替代，记录缺失；不要空白视频框。
6. **逐页视觉复核**：导出PDF并生成整页缩略图接触表，人工/程序检查版式重叠、表格过密、中文字体、色差、视频/图片变形、进度条对应章节、引用可见性；至少逐页打开所有PDF页核实关键文字可读。
7. **链接/一致性检验**：验证内/外部链接，PPT与PDF页数一致，图表数值一致，引用落到正确source，视频不是错场景，所有结论限定在正确cohort。
8. **git提交与反馈**：新增展示资料进入独立交付目录，优先在 `research-sci-program` 当前干净分支上独立提交（如与其他Codex任务并行则新建命名工作分支/PR）；大文件如受Git大小/隐私限制只留本地并给出确切路径和SHA。**不推送覆盖实验历史，不修改 STATE 科学结论**。
9. **最终报告**：提供PPT/PDF实际可打开路径、总页数、模板来源、引用与媒体核验汇总、仍不可用的媒体项、用于导师讨论的4–8个核心问题、commit SHA；不能把“计划已写”当作PPT完成。

完成标准：在没有口头解释时也能看懂**已做了什么 → 主要成功/失败证据 → 物理与算法解释 → 当前SCI论文证据缺口 → 有根据的下一步研究问题**；既适合给导师审阅，也可被实际用来讨论和作出研究决策。

## 9. 供 Codex 直接执行的单段启动指令

> 从当前 ARES `research-sci-program` 分支读取本任务书及同目录 `2026-11-13_MENTOR_SCI_PROPOSALS_AND_QUESTIONS.md`，按照真实AXIOM学术汇报 `.potx` 模板，基于完整历史原始结果、图表、视频以及可获得的 `ARES_research_review_20261113.zip`，**直接完成可编辑PPTX、PDF、讲稿、媒体/来源清单和QA**。科研结果与论文假设严格分开，补齐导师讨论的Proposal与Questions。重点复用真实对照视频、逐种子结果和错误干预案例；优先交付成品，不要只写大纲，不做新训练、不动真实机器人、不修改研究结论。提交独立成果并给出精确路径和SHA。