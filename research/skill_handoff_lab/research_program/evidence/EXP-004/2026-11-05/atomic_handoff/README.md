# 从执行权限到真实可学救回：2026-11-05

基点 `9a66e94`；公共执行接口提交 `181ee33`。以下全部是网络隔离的原生物理仿真；
ARES-R未改动，未操作实机，未使用封存测试。每个完整结果由原始终末状态复算。

**真正获得的能力：** 参数化闭环Option已实现并真实训练，学到的局部修正能救回名义失败，
包括冻结后的新自然初态。**未获得的能力：** 两独立初始化未稳定保留名义能力，
新24条件上候选10/24、12/24，名义12/24，冻结规则15/24，不能宣称净性能或独立算法优势。

## 训练与冻结确认

共同83维实际观测、12动作、强共享规划器、三种真实抓持、原生100步；所有臂从共同名义
预热开始，各自采集64完整训练episode，20更新/episode，成功BC系数固定0.2。
16条共享预热物理轨迹供两个独立网络初始化，每个512次actor预热、零critic预热。
每臂own-policy replay从空开始。队列/阶段/参数化Option改变的是时间策略类别；相同动作
字典不代表相同重开时机。全部是标准SAC/Option构件，不包装为新SAC。

| Temporal interface / NN seed | Dev0 | Dev32 | Dev64 (primary) | Final rescue / regression | Own train successes | Effective decisions | Actor / critic updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| macro_bc64 | 10/16 | 11/16 | 8/16 | 1 / 3 | 19/64 | 348 | 1280 / 1280 |
| macro_seed31702 | 10/16 | 11/16 | 12/16 | 2 / 0 | 26/64 | 339 | 1280 / 1280 |
| phase_bc64 | 10/16 | 6/16 | 6/16 | 0 / 4 | 19/64 | 443 | 1280 / 1280 |
| queue_bc64 | 10/16 | 10/16 | 9/16 | 0 / 1 | 27/64 | 300 | 1280 / 1280 |
| queue_seed31702 | 10/16 | 9/16 | 7/16 | 0 / 3 | 20/64 | 346 | 1280 / 1280 |

最初两臂比较队列与阶段重开；阶段结果变差后，第二迭代训练保持同一修正参数、在细化阶段
刷新参考的闭环Option。出现开发救回后，队列与该Option均以独立NN31702重复；未只延长候选。
训练数据761016–079，预热761000–015，开发761200–215；均非封存集。

以下使用预先登记的max(dev0,32,64)、同分取早规则选模，冻结后才运行新762200–223。
队列两种子选中checkpoint0，候选选中32/64；每臂均计满64训练成本。
历史final64主终点不改。24场景是新的开发确认，不称正式未见封存测试。

| Frozen method | Success | Rescue | Regression |
|---|---:|---:|---:|
| confirm_queue_bc64 | 12/24 | 0 | 0 |
| confirm_macro_bc64 | 10/24 | 1 | 3 |
| confirm_queue_seed31702 | 12/24 | 0 | 0 |
| confirm_macro_seed31702 | 12/24 | 1 | 1 |
| confirm_rule | 15/24 | 4 | 1 |
| nominal | 12/24 | 0 | 0 |

候选31701救回762214、退化762201/204/206；31702救回762213、退化762206。
规则救回762203/207/213/216、退化762204。762214是本轮冻结规则未覆盖的真实救回，
但不能以单案例取代总体比较。完整种子和模型SHA见FROZEN_SELECTION.json。

## 机制与竞争解释

- H1单次原子修正不足以解释760213。按历史精确阈值全原生重执行，共享76步记录前缀；
  单次atomic与refresh-then-local均失败，重复atomic+phase成功。早72步探针另存，未删除或冒称原见证。
- H2闭环参数Option重复刷新同一参数后，100步记录与原成功规则完全一致。接触仪器测得
  phase首次非零接触87步，queue98步；力信号只用于诊断，不是模型额外输入。
  新16dev上chunk/phase/queue冻结规则及保守预算保护均8/16，名义10/16：更多决策并非普遍更好。
- 学到的761202修正：共同63步前缀；learned成功、删除该次修正失败、nominal失败、只做该次修正成功。
  实际为local -Z，不能将此部署收益归因于atomic动作本身。
- 冻结新762214：共同58步前缀，learned/only-first成功，suppress-first失败。
  新762213的74步动作单独执行可成功；删除后模型稍后介入也成功，说明存在时间窗口而非唯一关键帧。
- 在共同退化762206，两模型对第一次干预相对continue的Mean-Q差分别约+0.201/+0.174，
  仍真实失败。Q-filter不会凭这一估值自动保护成功；Q不是成功概率。
- H3十个固定时间网格拼接全部拒绝，8次加速度检查、1次抓持丢失、1次队列不足。
  759205因此保留旧队列成功；零偏移重规划失败。此算子尚无新增操作能力。
  实测起点到位置命令的有限差分混有PD跟踪滞后，不能当实际电机加速度或普遍不可行证明。
  未认证接触安全，也未将此拒绝算子加入本轮在线训练。
- H4旧32条件posthoc交叉前驱物理实验：固定grasp1时旧/新后继15/32、17/32；
  交换前驱则13/32、17/32，原联合14/32、17/32。后继差异存在，前驱也影响归因。
  本轮三抓持均真实探索，但确认策略几乎都选grasp1；只有31702在762209选grasp0。
  该案例joint、front-fixed、rear-fixed均成功，尚无新增跨阶段协同收益证据。

## 第二物理机制

复用真实抓持→强局部IK servo的原生PlugCharger200，四新初态×两抓持×两策略；无Y/Z偏置、
无修改物理或成功判据，不虚构规划队列，也不混用Peg replay。这里是能力研究，不是方法迁移训练。

| Actual grasp pitch | Nominal pace | Feedback pace |
|---|---:|---:|
| 15 degrees | 4/4 | 4/4 |
| 30 degrees | 4/4 | 4/4 |

## 成本与限制

完整原生episode **1002**；物理步 **101800**；规划查询 **8934**。
五臂实际在线训练320episode/32000物理步；6400次actor、6400次critic及6400次未参与决策的
legacy auxiliary优化，另有两个初始化共1024次actor预热。每次配对前缀重执行均计入物理成本。
IK拼接内部查询另见原始queries.ik_queries。wall时间求和不等于并行经过时间。

| Run group | Native episodes | Physical steps | Planner queries | Sum episode wall seconds |
|---|---:|---:|---:|---:|
| ablation_front_fixed | 1 | 100 | 8 | 6.5 |
| ablation_rear_fixed | 1 | 100 | 7 | 8.1 |
| charger200 | 16 | 3200 | 714 | 94.1 |
| confirm_macro_bc64 | 24 | 2400 | 212 | 109.7 |
| confirm_macro_seed31702 | 24 | 2400 | 225 | 153.5 |
| confirm_nominal | 24 | 2400 | 166 | 109.6 |
| confirm_queue_bc64 | 24 | 2400 | 166 | 154.4 |
| confirm_queue_seed31702 | 24 | 2400 | 166 | 150.9 |
| confirm_rule | 24 | 2400 | 204 | 97.1 |
| confirmation_effect | 6 | 600 | 50 | 38.5 |
| contact_sequence | 2 | 200 | 18 | 7.7 |
| control_budget_guard_chunk | 16 | 1600 | 127 | 101.6 |
| control_rule_chunk | 16 | 1600 | 133 | 76.6 |
| control_rule_insertion_queue | 16 | 1600 | 119 | 103.9 |
| control_rule_phase | 16 | 1600 | 133 | 77.7 |
| cross_measured | 64 | 6400 | 501 | 299.7 |
| cross_relative | 64 | 6400 | 539 | 402.9 |
| h1_witness | 9 | 900 | 78 | 34.4 |
| h1_witness76 | 9 | 900 | 78 | 55.0 |
| learned_rescue | 4 | 400 | 32 | 16.0 |
| macro_bc64 | 112 | 11200 | 981 | 655.8 |
| macro_seed31702 | 112 | 11200 | 961 | 689.2 |
| macro_witness | 1 | 100 | 10 | 6.4 |
| phase_bc64 | 112 | 11200 | 1005 | 693.0 |
| progress_controls | 15 | 1500 | 115 | 92.8 |
| progress_splice | 10 | 1000 | 80 | 64.4 |
| queue_bc64 | 112 | 11200 | 947 | 685.2 |
| queue_seed31702 | 112 | 11200 | 922 | 691.7 |
| training_rule | 16 | 1600 | 126 | 84.6 |
| warm31701 | 16 | 1600 | 111 | 62.4 |

## 可直接查看与复现

- [Review](REVIEW.html)：学习曲线、成败配对、完整视频与同步曲线。
- [开发救回同步视频](video/SYNC_VIEWER.html)、[冻结新初态救回](video_confirmation/SYNC_VIEWER.html)。
  RAW与英文版均为真实测得状态的101帧可视化回放，不是重新跑出的物理或实机视频。
- SUMMARY.json / COMPARISONS.json / RAW_INDEX.json：全量终态核验、成本、每条真实轨迹SHA。
- PRIOR_ART.md：精读及开源实现的具体区别；标准Option、参考刷新、状态增广和BC不单独主张原创。
- RAW_MANIFEST.json / RAW_*.tar.gz：完整原始轨迹、模型、当前事务resume、源快照和协议。
  中间事务resume全部本地保留并在LOCAL_INVENTORY_SHA.json索引；公开包保留最终可续训事务及所有命名检查点。

仓库tools内的analyze_atomic_handoff.py、summarize_atomic_comparisons.py可从解压包复算；
verify_decoupled_archive.py验证每个包及成员SHA。固定训练命令和源版本在各MANIFEST/source中；
重跑物理必须经handoff_offline.sh，不可直接调用现场或生产入口。

## 研究决策

保留明确动作语义、83维执行状态、三种时间策略、真实救回及可持续own-policy训练资产。
不继续把阶段重开本身当创新，也不以当前拼接拒绝结果宣称局部轨迹修正成功。
下一项关键问题是：在真实可执行救回窗口与容易破坏的正常进展状态之间，如何校准
部署干预价值并约束策略改进，而不是仅增大Q或全局BC。应使用冻结前缀续接、训练期
真实局部干预数据和同权限成熟保持对照；新24组已看，不再冒充新测试。

本轮可进入论文的是执行语义、H1/H2交互、真实可学且可反证救回、独立初始化与冻结确认结果；
稳定净收益、跨阶段协同、跨任务学习迁移仍缺证据。独立算法优势继续UNVERIFIED，
但这些能力与负结果都已形成可复用的Method/Experiments材料。
