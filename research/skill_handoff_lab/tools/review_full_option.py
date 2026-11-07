"""Small review page from independently recounted native evidence."""
import argparse,csv,html,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
s=json.loads((a.root/'LEARNING_SUMMARY.json').read_text());f=json.loads((a.root/'FRONTIER.json').read_text());g=s['groups']
for name in ['LEARNING_SUMMARY.json','FRONTIER.json','RESCUE_MECHANISMS.json','REFERENCE_MECHANISM.json','REFERENCE_EFFECT_SHARING.json','PROTOCOL.json','FROZEN_CONFIRM_PROTOCOL.json','INITIALIZATION_ERRATUM.json','PROPOSAL_SUPPORT_AUDIT.json','CANDIDATE_REVISION.json','TARGET_CHART_CHECKS.json','LEARNER_REGRESSION.json','NORMALIZATION_AUDIT.json','STARTUP_REPAIR.json','ACTUAL_INITIALIZATIONS.json','EXTRACTION_DECOMPOSITION_PROTOCOL.json','CANDIDATE_FAILURE_EXAMPLE.json','CANDIDATE_FAILURE_PREFIX.json','TARGET_CHART_PHYSICAL_COUNTEREXAMPLES.json','LEGAL_MASK_SCOPE.json','PREDECESSOR_DEPLOYMENT_CHANGES.json','CONFIRMATION_BEHAVIOR.json','VIAC_SOURCE_REVIEW.json','POSTCONFIRM_SUPPORT_PROTOCOL.json','POSTCONFIRM_SUPPORT_EFFECT.json']:
 if (a.root/name).exists():shutil.copy2(a.root/name,a.out/name)
def table(headers,rows):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table>'
def scores(names,ref):
 rows=[]
 for k in names:
  v=g[k];b=g.get(ref,{}).get('outcomes',{});out=v['outcomes'];same=set(b)==set(out)
  rescue=sum(out[x] and not b[x]for x in out)if same else '—';reg=sum(b[x] and not out[x]for x in out)if same else '—'
  rows.append([k,f"{v['success']}/{v['n']}",rescue,reg,v['planning_queries'],round(v['planner_wall_s'],3)])
 return table(['Frozen policy / checkpoint','Native success','Rescue','Regression','Planner calls','Planner time (s)'],rows)
dev=[k for k in g if k.startswith('dev_')]
for run in s['learning']:
 keys=[k for k in g if k.startswith(run+'@') and g[k]['n']==24]
 if keys:dev.append(max(keys,key=lambda k:int(k.rsplit('@',1)[1])))
confirm=[k for k in g if k.startswith('confirm_')]
postconfirm=[k for k in g if k.startswith('postconfirm_')]
with (a.out/'RESULTS.csv').open('w',newline='')as handle:
 writer=csv.writer(handle);writer.writerow(['policy','cohort','success','n','reference','rescue','regression','planner_calls','planner_seconds'])
 for k in sorted(dev+confirm+postconfirm):
  v=g[k];ref='confirm_warm'if 'confirm_'in k else'sac_32201@0';b=g.get(ref,{}).get('outcomes',{});o=v['outcomes'];same=set(b)==set(o)
  writer.writerow([k,'posthoc_support_ablation'if k.startswith('postconfirm_')else('new_confirmation'if k.startswith('confirm_')else'seen_development'),v['success'],v['n'],ref,sum(o[x]and not b[x]for x in o)if same else'',sum(b[x]and not o[x]for x in o)if same else'',v['planning_queries'],v['planner_wall_s']])
tex=['% Generated from independently recounted LEARNING_SUMMARY.json.','% Development is seen; confirmation is not sealed testing.',r'\begin{tabular}{lrrr}',r'\hline',r'Policy & Success & Scenes & Planner calls \\',r'\hline']
for k in sorted(dev+confirm+postconfirm):
 v=g[k];tex.append(k.replace('_',r'\_')+' & '+str(v['success'])+' & '+str(v['n'])+' & '+str(v['planning_queries'])+r' \\')
tex += [r'\hline',r'\end{tabular}'];(a.out/'results_table.tex').write_text('\n'.join(tex)+'\n')
training=table(['Own-policy learner','Episodes','Exploratory successes','Physical steps','Option decisions','Updates','Planner calls','Optimizer time (s)'],[[k,v['episodes'],v['success'],v['physical_steps'],v['effective_decisions'],v['updates'],v['planning_queries'],round(v['optimizer_wall_s'],1)]for k,v in s['learning'].items()])
design=table(['Natural design scene','Warm success','Finite success union','First rear step','Remaining','Outside measured old support'],[[k,v['warm_success'],v['finite_union'],v['first_rear_step'],v['remaining'],v['only_new_physical_success']]for k,v in f['scenes'].items()])
figs=[('skill_chain','Continuous real grasp → SkillResult → recurrent executable Options'),('physical_frontier','Measured finite action frontier: training-design only'),('reference_conditional_effect','Same signed offset, opposite reference-dependent outcomes'),('learning_curves','Fixed checkpoint deployment; no retrospective best selection'),('online_behavior','Exploratory physical training behavior, separate from deployment'),('frozen_pairings','Paired rescue and regression on seen development'),('confirmation_pairings','Frozen natural-condition confirmation'),('learned_rescue_physics','Two complete native trajectories; identical physical prefixes'),('postconfirm_support_physics','Expanded support: new capability and harmful intervention, same frozen policy')]
gallery=''.join(f'<section><h2>{html.escape(title)}</h2><a href="figures/{name}.pdf">PDF</a><img src="figures/{name}.png" alt="{html.escape(title)}"></section>'for name,title in figs if (a.out/'figures'/f'{name}.png').exists())
video=''
if(a.out/'video_765423/ANNOTATED.mp4').exists():video='<section><h2>Observed SAC rescue: scene765423</h2><p>101 recorded frames, original native execution. Visualization is MEASURED_STATE_REPLAY with zero new physics; it is not independent method validation. The retained historical RL also rescues this scene; this is not exclusive new capability.</p><video controls preload="metadata" src="video_765423/ANNOTATED.mp4"></video><p><a href="video_765423/RAW.mp4">Complete RAW</a> · <a href="video_765423/SYNC_VIEWER.html">Frame-synchronized curves</a> · <a href="video_765423/MANIFEST.json">Source SHA</a></p></section>'
if(a.out/'video_765416/ANNOTATED.mp4').exists():video+='<section><h2>Negative mechanism evidence: scene765416</h2><p>Identical actual prefix through step68. The chart candidate misses a useful local Y+ intervention and fails; flat SAC retains success. MEASURED_STATE_REPLAY, zero new physics.</p><video controls preload="metadata" src="video_765416/ANNOTATED.mp4"></video><p><a href="video_765416/RAW.mp4">Complete RAW</a> · <a href="video_765416/SYNC_VIEWER.html">Frame-synchronized curves</a> · <a href="video_765416/MANIFEST.json">Source SHA</a></p></section>'
if(a.out/'video_765625/ANNOTATED.mp4').exists():video='<section><h2>New natural condition: scene765625</h2><p>Frozen ordinary SAC32202 rescues a warm failure. Same actual physical prefix through step64. This is real native execution evidence visualized as MEASURED_STATE_REPLAY, not a novel-algorithm claim.</p><video controls preload="metadata" src="video_765625/ANNOTATED.mp4"></video><p><a href="video_765625/RAW.mp4">Complete RAW</a> · <a href="video_765625/SYNC_VIEWER.html">Frame-synchronized curves</a> · <a href="video_765625/MANIFEST.json">Source SHA</a></p></section>'+video
cost=html.escape(json.dumps(s['cost'],indent=2))
headline='；'.join(k.removeprefix('confirm_')+f" {g[k]['success']}/{g[k]['n']}" for k in sorted(confirm))
page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ARES | Executable action frontier and actual online learning</title>
<style>body{{margin:0;background:#edf2f5;color:#172d3c;font:16px/1.6 system-ui}}main{{max-width:1250px;margin:auto;padding:30px}}header,section{{background:white;border-radius:10px;padding:26px;margin:18px 0}}header{{border-top:7px solid #287a98}}h1{{font-size:30px;line-height:1.25}}h2{{font-size:22px}}small,.muted{{color:#536875}}.tag{{display:inline-block;background:#dcecf1;padding:4px 10px;margin-right:7px;border-radius:4px}}.note{{border-left:4px solid #bf7746;padding:12px 18px;background:#fff5e9}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #dce4e9}}th{{background:#e8f0f4}}img,video{{width:100%;height:auto;margin-top:10px}}a{{color:#166c8d}}pre{{white-space:pre-wrap;font-size:13px}}.scroll{{overflow:auto}}</style>
<main><header><span class="tag">Native Peg100</span><span class="tag">Real PhysX execution</span><span class="tag">No physical robot motion</span><h1>可执行动作机会 → 实际在线技能学习</h1><p>ARES SCI · 2026-11-07 · c50123e continuation</p><p>保留强控制器、真实抓持、原生100步和成功判据。先检查有限动作机会，再训练同权限的普通RL、监督对照与可证伪的表示候选。</p><p><a href="README.md">研究结论</a> · <a href="PRIOR_ART.md">论文与源码对照</a> · <a href="THEORY.md">机制与反例</a> · <a href="LEARNING_SUMMARY.json">复算结果</a> · <a href="RAW_MANIFEST.json">原始证据 / SHA</a></p></header>
<section><h2>已完成的实质研究</h2><p>256次设计物理执行后，完成5个128回合own-policy训练：普通SAC两种子、两版表示候选、终局监督评分。固定最终检查点，保留全部中途退化。</p><p>冻结新自然初态：{html.escape(headline)}</p><p>两种子在已看开发集同为21/24并不保证泛化一致；动作目录机会、坐标计算准确与独立方法增益分别判断。</p></section><section><h2>必须保留的证据边界</h2><div class="note">动作并集不是部署策略分数；当前开发场景已看。冻结新条件确认也不是密封测试。普通SAC、动作参数化、核函数和状态条件化关系均有成熟先例，候选优势必须由同权限实际执行及消融支持。</div><p>初始化勘误：旧31901/31902初始Actor/Critic相同，只有探索随机性不同。本轮在加载预热资产后重设网络种子，历史分数不改。</p><p>新动作使用不等于救回原因：765402在第93步已成功，随后第94步的Z−不能直接获记救回信用。冻结提议权限消融与所有失败均保留。</p></section>
<section><h2>Seen development ·765400–765423</h2><p>救回/退化相对同一预热初始化19/24。每个新学习臂显示其已完成的固定终点；全部中间checkpoint见学习曲线和JSON。</p><div class="scroll">{scores(sorted(dev),'sac_32201@0')}</div></section>
<section><h2>Frozen new-condition confirmation ·765600–765631</h2><p>以下仅显示实际已执行结果。空表意味着仍保留未开启；不能用开发分数补填。</p><div class="scroll">{scores(sorted(confirm),'confirm_warm')}</div></section>
<section><h2>Post-hoc catalogue-support ablation</h2><p>同一32场景现已看：冻结两个SAC模型，仅限制旧提议动作支持；不是重新训练或新的独立确认。</p><div class="scroll">{scores(sorted(postconfirm),'confirm_warm')}</div></section><section><h2>Own-policy physical training</h2><p>各臂额外共享256次原生设计回放；不能把共享数据成本当作零。网络优化在CPU、物理仿真在单张本地RTX3080。训练随机行为成功与贪心部署成功分开统计。</p><div class="scroll">{training}</div></section>{gallery}{video}
<section><h2>Design-only opportunity</h2><p>16个自然初态，256次完整重执行。Warm8/16、设计集最佳固定动作12/16、有限实测并集14/16。不是全局上界，也未证明必须联合改变两个技能。</p><div class="scroll">{design}</div></section><section><h2>Actual cost and failures</h2><pre>{cost}</pre><p>两次采样启动故障的200物理步计入总成本，未加入有效学习经验；原始故障文件和修复说明均保留。耗时是累计执行时间，不是并发后的墙钟历时。</p></section></main></html>'''
(a.out/'REVIEW.html').write_text(page)
print(a.out/'REVIEW.html')
