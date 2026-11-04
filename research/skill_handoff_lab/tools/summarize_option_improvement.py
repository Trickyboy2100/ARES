"""Quantify native branch opportunity, real cost and frozen-policy learning curves."""
import argparse,json,gzip,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text())
def savefig(fig,path):
 fig.savefig(path.with_suffix('.png'),dpi=190,bbox_inches='tight');fig.savefig(path.with_suffix('.svg'),bbox_inches='tight');f=path.with_suffix('.svg');f.write_text('\n'.join(x.rstrip()for x in f.read_text().splitlines())+'\n');plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;figs=r/'figures';figs.mkdir(exist_ok=True);opportunity={}
 for name in ['design','development','extended_design','extended_development']:
  rows=read(r/name/'ROWS.json');scenes=sorted({x['seed']for x in rows});nb=1+max(int(x['tag'][-1])for x in rows);mat=np.array([[int(next(x for x in rows if x['seed']==s and x['tag']==f'fixed{aa}{b}')['success'])for aa in range(3)for b in range(nb)]for s in scenes]);fixed=mat.sum(0)
  opportunity[name]=dict(n_scenes=len(scenes),fixed=fixed.tolist(),best_fixed=int(fixed.max()),measured_union=int(mat.max(1).sum()),all_success=int(mat.min(1).sum()),all_fail=int((mat.max(1)==0).sum()),mixed=int(((mat.max(1)>0)&(mat.min(1)==0)).sum()),fixed_selection='hindsight bound, not train-selected deployment')
  if name.startswith('extended'):
   fig,ax=plt.subplots(figsize=(7.5,10));ax.imshow(mat,cmap=matplotlib.colors.ListedColormap(['#ead9d7','#1b8568']),vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(3*nb),[f'{[-10,0,10][aa]}mm\n{["original","refresh","split"][b]}'for aa in range(3)for b in range(nb)]);ax.set_yticks(range(len(scenes)),[str(s)for s in scenes],fontsize=8);ax.set_title(f'{name}: native100 complete task\nGreen=success, pale=fail; measured branches, no oracle policy');savefig(fig,figs/f'{name}_physical_matrix')
 tables={};paired={}
 for mode,base,cp in [('two_options',r,192),('three_options',r/'threeway',288)]:
  folder=base/('eval_controls'if (base/'eval_controls').exists()else 'eval_development');f=folder/'SCORES.json'
  if not f.exists():continue
  scores=read(f);final=[x for x in scores if x['checkpoint']==cp and x['rule']=='deployed'];tables[mode]=final
  pred=read(folder/'PREDICTIONS.json');lookup={(x['method'],x['seed']):x['rows']for x in pred if x['checkpoint']==cp and x['rule']=='deployed'}
  for seed in (31401,31402):
   cand=lookup['handoff',seed]
   for competitor in ['sac_min','sac_mean','vi_mean','vi_expectile','usac','lower_bound','handoff_complete','bee','fqi','iql','cql','supervised','stap']:
    if (competitor,seed)not in lookup:continue
    comp=lookup[competitor,seed];assert [x['scene']for x in cand]==[x['scene']for x in comp]
    delta=np.array([int(x['success'])-int(y['success'])for x,y in zip(cand,comp)]);rng=np.random.default_rng(914);means=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    paired[f'{mode}:{seed}:{competitor}']=dict(rescues=int((delta==1).sum()),regressions=int((delta==-1).sum()),paired_difference=float(delta.mean()),scene_bootstrap_95=np.quantile(means,[.025,.975]).tolist(),note='exploratory development, no multiplicity-adjusted superiority claim')
  methods=['sac_mean','vi_mean','vi_expectile','usac','handoff','lower_bound','bee','supervised','stap'];fig,ax=plt.subplots(figsize=(9,5))
  for method in methods:
   sub=[x for x in scores if x['method']==method and x['rule']=='deployed'];ks=sorted({x['checkpoint']for x in sub});ys=[np.mean([x['success']/x['n']for x in sub if x['checkpoint']==k])for k in ks];ax.plot(ks,ys,marker='o',label=method)
  ax.set(xlabel='Shared real training episodes (plus60 prior)',ylabel='Native100 success /32 development scenes',ylim=(0,1),title=f'{mode}: two optimization seeds, same32 scenes\nExact logged physical branch evaluation; not extra rollouts');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);savefig(fig,figs/f'{mode}_learning_curve')
 online={}
 f=r/'threeway/online_evaluation/SCORES.json'
 if f.exists():
  rows=read(f);online['scores']=rows;pred=read(f.with_name('PREDICTIONS.json'));pairs={}
  for seed in (31401,31402):
   cand=next(x['rows']for x in pred if x['method']=='handoff_complete'and x['seed']==seed and x['checkpoint']==32)
   for kind in ['sac_mean','vi_mean']:
    other=next(x['rows']for x in pred if x['method']==kind and x['seed']==seed and x['checkpoint']==32);d=np.array([int(a['success'])-int(b['success'])for a,b in zip(cand,other)]);pairs[f'{seed}:{kind}']=dict(rescues=int((d>0).sum()),regressions=int((d<0).sum()),difference=float(d.mean()))
  online['pairs']=pairs
  fig,ax=plt.subplots(figsize=(8,4.8))
  for kind in ['sac_mean','vi_mean','handoff_complete']:
   sub=[x for x in rows if x['method']==kind];xs=sorted({x['checkpoint']for x in sub});yy=np.array([[next(x['success']/x['n']for x in sub if x['seed']==seed and x['checkpoint']==k)for k in xs]for seed in (31401,31402)])
   ax.plot(xs,yy.mean(0),marker='o',label=kind);ax.fill_between(xs,yy.min(0),yy.max(0),alpha=.12)
  ax.axhline(opportunity['extended_development']['best_fixed']/32,color='gray',ls='--',label='best fixed (hindsight)');ax.axhline(opportunity['extended_development']['measured_union']/32,color='black',ls=':',label='finite union (not a policy)');ax.set(xlabel='Additional own-policy native100 episodes',ylabel='Native success rate (32 development scenes)',ylim=(0,1),title='Actual online learning after equal shared physical pretraining\nTwo seeds; band=seed range, not confidence interval');ax.legend(fontsize=8);ax.grid(alpha=.2);savefig(fig,figs/'online_physical_learning')
 support=read(r/'UNIQUE_SUPPORT_AUDIT.json');fig,ax=plt.subplots(figsize=(9,5));rng=np.random.default_rng(41)
 for j,kind in enumerate(['fqi','cql','vi_mean','handoff_complete']):
  for i,seed in enumerate((31401,31402)):
   rec=next(x for x in support if x['kind']==kind and x['seed']==seed and x['checkpoint']==288);vals=np.array(rec['values'])[:,2];xx=j+(-.15 if i==0 else .15);ax.scatter(xx+rng.uniform(-.07,.07,len(vals)),vals,s=16,alpha=.65,color=['#3476b2','#e18b39'][i],label=str(seed)if j==0 else None);ax.plot([xx-.08,xx+.08],[np.median(vals)]*2,color='black',lw=2)
 ax.axhline(1,color='#b84742',ls='--',label='terminal binary return upper bound');ax.axhline(0,color='gray',ls=':');ax.set_xticks(range(4),['FQI','CQL','Exact value\nimprovement','Complete-handoff\nbackup']);ax.set(ylabel='Mean critic value of historically unexecuted Option2',title='Legal action does not imply historical execution support\n29 distinct handoffs from10 historical scenes; actual Option2 outcomes unknown');ax.legend(fontsize=8);savefig(fig,figs/'unsupported_value_extrapolation')
 # Count physical sources once, excluding merged reference tables and pose videos.
 physics=[]
 for f in r.rglob('physics/RESULTS.json'):physics+=read(f)
 paths=[x['path']for x in physics];assert len(paths)==len(set(paths));errors=[]
 for row in physics:
  f=Path(row['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'];d=json.load(gzip.open(f,'rt'));assert len(d['steps'])==100;assert bool(d['steps'][-1]['evaluation']['success'][0])==row['success']
  if row['error']not in (None,'HORIZON','PLAN_FAILED'):errors.append(dict(path=str(f),error=row['error']))
 costs=dict(actual_episodes=len(physics),control_steps=sum(x['steps']for x in physics),planner_queries=sum(len(x['queries'])for x in physics),planner_wall_s=sum(q['wall_s']for x in physics for q in x['queries']),effective_option_decisions=sum(len(x['decisions'])for x in physics),errors=errors,wall_note='concurrent single-GPU processes; timings are not isolated latency benchmarks')
 updates=0
 for f in r.rglob('train_*/HISTORY.json'):updates+=sum(x['updates']for x in read(f))
 costs['shared_data_primary_critic_updates']=updates;costs['own_policy_additional_critic_updates']=sum(25*len(read(f))for f in r.glob('threeway/online_*/HISTORY.json'));costs['total_primary_critic_updates']=updates+costs['own_policy_additional_critic_updates'];costs['fixture_and_audit_updates_excluded']=True
 out=dict(claim='UNVERIFIED',opportunity=opportunity,final_tables=tables,paired=paired,online=online,cost=costs);(r/'RESULTS_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(opportunity=opportunity,cost=costs),indent=2))
if __name__=='__main__':main()
