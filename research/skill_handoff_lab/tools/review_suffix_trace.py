"""Build a compact review from completed actual native runs."""
import argparse,json,html,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=json.loads((a.out/'SUMMARY.json').read_text());ad=json.loads((a.out/'ABLATION_SUMMARY.json').read_text())
assert len(d['arms'])==3 and all(v['own_episodes']==64 and v['curve']['64']['n']==12 for v in d['arms'].values())
assert len(d['controls'])==8 and all(v['n']==24 for v in d['controls'].values())
assert len(ad['arms'])==2 and all(v['own_episodes']==32 and v['curve']['32']['n']==12 for v in ad['arms'].values())
prior=d['controls']['confirm_prior']['outcomes'];ct=[]
for name,v in d['controls'].items():
 r=[s for s,x in v['outcomes'].items()if x and not prior[s]];g=[s for s,x in v['outcomes'].items()if not x and prior[s]]
 ct.append(dict(method=name.removeprefix('confirm_'),success=v['success'],n=24,rescue=r,regression=g))
cost={k:d['cost'][k]+ad['cost'][k]for k in d['cost']};result=dict(confirmation=ct,total_new_cost=cost,primary=d['arms'],entropy_ablation=ad['arms'],scope='exploratory one shared pretrained initialization; not sealed and not independent network seeds')
(a.out/'RESULTS.json').write_text(json.dumps(result,indent=2))
for name in ['PROTOCOL.json','ENTROPY_ABLATION_PROTOCOL.json','FROZEN_CONFIRMATION.json','FORK_PHYSICAL_RECHECK.json','HISTORICAL_WITNESS_RECHECK.json','TRACE32_DIAG.json','ADOPTION32_PROBE.json','ENTROPY_SCOPE_PROBE.json','PRIOR_SOURCES.json','PRIOR_IMPLEMENTATIONS.json','ABLATION_FROZEN.json']:
 shutil.copy2(a.root/name,a.out/name)
for name in ['PREFIX.json','QUEUE_MATCH.json','VALUES64.json','MANIFEST.json']:
 shutil.copy2(a.root/'mechanism'/name,a.out/('MECHANISM_'+name))
rows=''.join(f'<tr><td>{html.escape(x["method"])}</td><td>{x["success"]}/24</td><td>{len(x["rescue"])}</td><td>{len(x["regression"])}</td></tr>'for x in ct)
curve=''.join(f'<tr><td>{n}</td><td>{v["curve"]["0"]["success"]}</td><td>{v["curve"]["32"]["success"]}</td><td>{v["curve"]["64"]["success"]}</td></tr>'for n,v in d['arms'].items())
ab=''.join(f'<tr><td>{n}</td><td>{v["curve"]["0"]["success"]}</td><td>{v["curve"]["32"]["success"]}</td><td>{d["arms"][n]["curve"]["64"]["success"]}</td></tr>'for n,v in ad['arms'].items())
page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ARES · Executed continuation study</title><style>
body{{max-width:1150px;margin:40px auto;padding:0 24px;font:17px/1.65 system-ui;background:#f4f6f8;color:#162536}}h1{{font-size:34px}}h2{{margin-top:38px}}section{{background:white;padding:24px;border-radius:12px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dbe2eb}}a{{color:#1764a5}}img,video{{max-width:100%}}.note{{border-left:5px solid #ce8a2b;padding:15px;background:#fff4dc}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:700px){{.grid{{display:block}}}}</style>
<h1>实测续接修正，是否改善完整技能执行？</h1><p>ARES SCI · PegInsertionSide-v1 · 原生100步 · 真实抓持与强共享规划 · 2026-11-08</p>
<p class="note">从保留的强SAC权重和优化器继续，未退回弱初始化。标准Retrace和周期策略迭代均有成熟先例；不因更新公式或值拟合改变而宣称独立方法收益。封存测试未打开。</p>
<section><h2>冻结新24场景：完整原生结果</h2><table><tr><th>策略</th><th>成功</th><th>相对继承SAC救回</th><th>退化</th></tr>{rows}</table><p>训练769100–163；开发769400–411；确认769600–623。确认前固定final64与模型SHA。固定localZ−参照有一位自身动作历史，不是网络结构匹配消融。规则和固定控制不使用未来结果。</p><img src="figures/CONFIRMATION.png"></section>
<section><h2>真实在线学习与单因素修订</h2><table><tr><th>主臂</th><th>初始 /12</th><th>32 /12</th><th>64 /12</th></tr>{curve}</table><div class="grid"><img src="figures/LEARNING.png"><img src="figures/PAIRED.png"></div><p>三个主臂各64次自身策略采集、2560轮Q/Actor更新。相同继承基础，不是三个独立网络初始化。</p><table><tr><th>零熵消融</th><th>精确分叉32 /12</th><th>新增32 /12</th><th>原熵最终64 /12</th></tr>{ab}</table><p>仅把Bellman与Actor熵系数.01改为0，保留10%随机探索、.1成功BC及相同后32训练初态。每臂额外32次真实执行；不把分叉前32回合重复计入唯一物理成本。</p></section>
<section><h2>实际改变了什么</h2><p>保存实际Option选择概率和相邻执行边界，用截断重要性修正传播真实后续奖励；旧数据缺少行为概率时只做一步更新。候选每8个自身回合固定一次所评估的Actor版本，物理执行仍可在合法事件后再次决策。它改变价值目标及随后的Actor更新，不改变执行权限，也没有新增预测头。</p><img src="figures/TRACE_ACTIVITY.png"><p>图示只是实际多步更新覆盖长度，不是算法收益。<a href="METHOD.md">公式与限制</a> · <a href="PRIOR_ART.md">原论文和源码比较</a></p></section>
<section><h2>真实失败与完整视频</h2><p>769400：继承策略成功，Retrace续训后失败；32回合诊断在共同56步后产生不同修正。769401：有效首个Option被放弃；不能把所有失败都解释成“正确首次干预后的续接破坏”。记录状态上的Q排序并不等于该状态被每条部署轨迹访问。</p><video controls preload="metadata" src="video_769400/ANNOTATED.mp4"></video><p><a href="video_769400/RAW.mp4">完整RAW</a> · <a href="video_769400/SYNC_VIEWER.html">逐帧同步曲线</a> · <a href="video_769400/MANIFEST.json">原始轨迹SHA</a></p><p>明确标注MEASURED_STATE_REPLAY：渲染保存的真实关节/物体状态，新增物理步0；成功取自源原生执行。历史757300的100步成功见证已重新核对，不重复记作本轮实验。</p></section>
<section><h2>反证：新的失败不全是续接破坏</h2><p>769401额外3次原生执行：学习策略失败；强制首次selector6后续接学习策略成功；同一首次动作后续接继承策略成功。边界83维观测、活动队列和目标一致，记录物理前缀一致；不宣称完整隐藏PhysX状态一致。人工强制仅是机制探针，不计学习器成绩。</p><p>final64在该真实边界给selector0概率0.8645、平均Q=1.0667，selector6概率0.1011、平均Q=0.9731；与实际救回排序相反。checkpoint32曾有相反Q排序，不能把旧模型诊断冒充final64结果。<a href="MECHANISM_QUEUE_MATCH.json">队列核对</a> · <a href="MECHANISM_VALUES64.json">冻结模型数值</a></p></section><section><h2>完整成本与复现</h2><p>{cost['own_episodes']}次新增自身策略训练，{cost['updates']}轮更新；总计{cost['episodes']}次原生执行、{cost['steps']}物理步、{cost['queries']}次记录规划查询。继承数据和模型的历史成本另列；设备精确峰值未连续测量。</p><p><a href="README.md">中文结论与下一项实验</a> · <a href="RESULTS.json">原始汇总</a> · <a href="REPRODUCE.md">复现</a> · <a href="RAW_MANIFEST.json">全部模型/轨迹归档SHA</a> · <a href="FROZEN_CONFIRMATION.json">冻结协议</a></p></section></html>'''
(a.out/'REVIEW.html').write_text(page)
print(json.dumps({'confirmation':ct,'cost':cost},indent=2))
