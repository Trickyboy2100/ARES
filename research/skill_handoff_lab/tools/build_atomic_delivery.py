"""Publish concise, claim-specific research evidence from completed raw analyses."""
import argparse,json,shutil,html
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;e=a.out;e.mkdir(parents=True,exist_ok=True)
s=json.loads((r/'SUMMARY.json').read_text());c=json.loads((r/'COMPARISONS.json').read_text());charger=json.loads((r/'charger200/RESULTS.json').read_text());assert len(charger)==16
assert all(v['episodes']==64 for v in s['training'].values())
for f in r.glob('*.json'):shutil.copy2(f,e/f.name)
for name in ['figures','video','video_confirmation']:shutil.copytree(r/name,e/name,dirs_exist_ok=True)
training='| Temporal interface / NN seed | Dev0 | Dev32 | Dev64 (primary) | Final rescue / regression | Own train successes | Effective decisions | Actor / critic updates |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for n,v in sorted(s['training'].items()):
 z=v['curve'];f=z['64'];training+=f"| {n} | {z['0']['success']}/16 | {z['32']['success']}/16 | {f['success']}/16 | {len(f['rescue'])} / {len(f['regression'])} | {v['success']}/64 | {v['effective_decisions']} | {v['actor_updates']} / {v['critic_updates']} |\n"
confirmation='| Frozen method | Success | Rescue | Regression |\n|---|---:|---:|---:|\n'
for n,v in c['confirmation'].items():confirmation+=f"| {n} | {v['success']}/{v['n']} | {len(v.get('rescue',[]))} | {len(v.get('regression',[]))} |\n"
charger_table='| Actual grasp pitch | Nominal pace | Feedback pace |\n|---|---:|---:|\n'
for angle in [15,30]:charger_table+=f"| {angle} degrees | "+' | '.join(f"{sum(x['success']for x in charger if x['angle']==angle and x['policy']==p)}/4"for p in ['nominal','feedback'])+' |\n'
cost='| Run group | Native episodes | Physical steps | Planner queries | Sum episode wall seconds |\n|---|---:|---:|---:|---:|\n'
for n,v in sorted(s['groups'].items()):cost+=f"| {n} | {v['episodes']} | {v['steps']} | {v['planning_queries']} | {v['wall_s']:.1f} |\n"
text='''# 从执行权限到真实可学救回：2026-11-05

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

''' + training + '''
最初两臂比较队列与阶段重开；阶段结果变差后，第二迭代训练保持同一修正参数、在细化阶段
刷新参考的闭环Option。出现开发救回后，队列与该Option均以独立NN31702重复；未只延长候选。
训练数据761016–079，预热761000–015，开发761200–215；均非封存集。

以下使用预先登记的max(dev0,32,64)、同分取早规则选模，冻结后才运行新762200–223。
队列两种子选中checkpoint0，候选选中32/64；每臂均计满64训练成本。
历史final64主终点不改。24场景是新的开发确认，不称正式未见封存测试。

''' + confirmation + '''
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

''' + charger_table + '''
## 成本与限制

''' + f"完整原生episode **{s['total']['episodes']}**；物理步 **{s['total']['steps']}**；规划查询 **{s['total']['planning_queries']}**。\n" + '''五臂实际在线训练320episode/32000物理步；6400次actor、6400次critic及6400次未参与决策的
legacy auxiliary优化，另有两个初始化共1024次actor预热。每次配对前缀重执行均计入物理成本。
IK拼接内部查询另见原始queries.ik_queries。wall时间求和不等于并行经过时间。

''' + cost + '''
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
'''
(e/'README.md').write_text(text)
# Tables remain available as manuscript-ready source without a full manuscript rebuild.
(e/'RESULT_TABLES.md').write_text(training+'\n'+confirmation+'\n'+charger_table)
rows=''.join(f'<tr><td>{html.escape(n)}</td><td>{v["success"]}/{v["n"]}</td><td>{len(v.get("rescue",[]))}</td><td>{len(v.get("regression",[]))}</td></tr>'for n,v in c['confirmation'].items())
page='''<!doctype html><meta charset="utf-8"><title>ARES: executable adaptation evidence</title><style>body{font:18px/1.55 system-ui;max-width:1160px;margin:36px auto;padding:0 20px;background:#f6f8fb;color:#172a3a}h1,h2{line-height:1.2}img,video{width:100%;border-radius:8px}section{background:white;padding:25px;margin:22px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}a{color:#176f91}.note{border-left:5px solid #b57600;padding:15px;background:#fff3da}</style><h1>从执行权限到真实可学救回</h1><p>Native PhysX · Peg100 / Charger200 · 2026-11-05 · no hardware / no sealed test</p><p class="note">真实救回成立，稳定净性能优势未成立。新24条件：名义12，冻结规则15，候选10/12。保留全部负结果。</p><p><a href="README.md">完整结论与成本</a> · <a href="PRIOR_ART.md">近邻区别</a> · <a href="COMPARISONS.json">逐例配对</a> · <a href="RAW_MANIFEST.json">原始包SHA</a></p><section><h2>完整学习曲线</h2><img src="figures/learning.png"><p>每臂64回合自身策略采集；共同强执行器和动作字典，时间策略类别不同。0/32/64完整开发评价，未隐藏后期退化。</p></section><section><h2>冻结后的新24初态</h2><table><tr><th>Method</th><th>Success</th><th>Rescue</th><th>Regression</th></tr>'''+rows+'''</table><p>同一预登记选模规则。队列基线选择训练前模型，但仍计全部64回合成本。非封存测试。</p></section><section><h2>学到的真实修正</h2><video controls preload="metadata" src="video_confirmation/ANNOTATED.mp4"></video><p>762214：名义失败；冻结策略第58步修正成功；移除该次修正失败。</p><p><a href="video_confirmation/RAW.mp4">完整RAW</a> · <a href="video_confirmation/SYNC_VIEWER.html">逐帧可点击同步曲线</a> · <a href="video/SYNC_VIEWER.html">761202开发机理对照</a></p><p>视频由真实记录的关节/物体状态逐帧渲染，非新物理采样或实机。模型回读误差在视频MANIFEST披露。</p></section><section><h2>为何单次干预不够</h2><img src="figures/witness_timing.png"><p>760213：同76步前缀；单次atomic失败，重复阶段刷新成功；首次接触87 vs98步。标准Option机制，不冒称新算法。</p></section><section><h2>真实能力与论文边界</h2><p>保留83维执行观测、12动作、参数化闭环Option、独立NN初始化、真实前驱抓持和全部模型。下一步针对必要干预的估值与能力保持；当前无稳定跨阶段协同或第二任务学习迁移证据。</p></section>'''
(e/'REVIEW.html').write_text(page)
print(e/'REVIEW.html')
