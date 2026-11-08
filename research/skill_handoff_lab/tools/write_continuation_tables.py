"""Tables from finalized physical records; no best-checkpoint or seed selection."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();s=json.loads((a.root/'analysis/SUMMARY.json').read_text());g=s['groups'];frozen=json.loads((a.root/'FROZEN_COMPARISON.json').read_text());assert all(g.get('confirm_'+x['name'],{}).get('episodes')==32 for x in frozen['items']);assert not s['pending_files'];a.out.mkdir(parents=True,exist_ok=True)
labels={'sac':'SAC + success BC','retention':'Reference KL + paired ranking','ranking':'Supervised + paired ranking','witness_soft':'Soft witness support','witness_qp':'QP witness update'}
rows=[]
for name,label in [('fixed5','Frozen first localZ-minus'),('warm','Warm'),('rule','Frozen feedback rule')]:
 v=g['confirm_'+name];rows.append([label,'frozen',f"{v['success']}/32",len(v['rescue']),len(v['regression']),0,0,0,0,0])
for arm,label in labels.items():
 for seed in [32301,32302]:
  name=f'{arm}_{seed}';v=g['confirm_'+name];t=s['training'][name];rows.append([label,seed,f"{v['success']}/32",len(v['rescue']),len(v['regression']),t['own']['episodes'],t['own_decisions'],t['updates'],t['refresh']['episodes'],t['own']['planner_calls']+t['refresh']['planner_calls']+s['shared']['planner_calls']])
header=['Learner','NN seed','Success','Rescue','Regression','Own episodes','Own decisions','Updates','Refresh episodes','New train planning calls incl. shared']
md='| '+' | '.join(header)+' |\n|'+ '|'.join(['---']*len(header))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)
(a.out/'PRIMARY_TABLE.md').write_text(md)
tex='\\begin{tabular}{lrrrrr}\n\\toprule\nLearner & Seed & Success & Rescue & Regression & Own episodes \\\\\n\\midrule\n'
for r in rows:tex+=' & '.join(map(str,r[:6]))+' \\\\\n'
tex+='\\bottomrule\n\\end{tabular}\n% Same32 natural scenes; two network seeds. Each learner also pays189 shared episodes and its reported refresh episodes.\n'
(a.out/'PRIMARY_TABLE.tex').write_text(tex)
other=[]
for x in frozen['items']:
 if x['name'].rsplit('_',1)[0]in labels and x['own_budget']==128:continue
 v=g['confirm_'+x['name']];other.append([x['name'],f"{v['success']}/32",len(v['rescue']),len(v['regression']),x['own_budget'],x['scope']])
(a.out/'CONTROL_TABLE.md').write_text('| Policy | Success | Rescue | Regression | Own budget | Scope |\n|---|---|---|---|---|---|\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in other))
print('Final tables from',len(frozen['items']),'complete32-scene evaluations')
