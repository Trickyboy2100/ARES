"""Independent recomputation of intervention outcomes and policy paired results."""
import argparse,gzip,json,csv,itertools
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[]
 for path in sorted(a.root.rglob('TRAJECTORY.json.gz')):
  if 'datasets' in path.parts:continue
  d=json.load(gzip.open(path,'rt'));r=d['summary'];m=json.loads((path.parent/'MANIFEST.json').read_text());last=d['steps'][-1];dist=float(np.linalg.norm(np.array(last['cube'][:2])-[-.045,r['target_y']]));native=last['cube_base_distance_m']<.6;ok=bool(native and dist<=.04);assert ok==r['task_success'];assert len(d['steps'])==100
  decisions=d['decisions'];grasp=decisions[0]['parameters']['grasp_x_m'];hook=decisions[1]['parameters']['hook_y_m'] if len(decisions)>1 else None
  phase=last['phase'];failure='SUCCESS' if ok else 'PREFIX' if len(decisions)<2 else 'BUDGET_'+phase if r['error'] and 'HORIZON' in r['error'] else 'NATIVE_REGION' if not native else 'DECLARED_TARGET'
  logs=d.get('selector_logs',[]);lastlog=logs[-1] if logs else {};vals=lastlog.get('all_values',[]);pred=max(vals) if vals else lastlog.get('p_success');pred=pred if lastlog.get('stage')==1 else None
  if r.get('method')=='stap_uq' or (hook is not None and lastlog.get('normalized_action') is not None and abs(hook-(-.065+.01*lastlog['normalized_action']))>1e-7):pred=None
  rows.append(dict(predicted_success=pred,group=path.relative_to(a.root).parts[0],seed=r['seed'],mass=r['density_scale'],target=r['target_y'],method=r.get('method','intervention'),grasp=grasp,hook=hook,success=int(ok),native=int(native),distance_m=dist,failure=failure,wall_s=r['wall_s'],path=str(path),initial_sha256=r['initial_sha256']))
 fields=list(rows[0]);f=(a.out/'EPISODES.csv').open('w');w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows);f.close()
 intervention=[r for r in rows if r['group']=='train96'];contexts=defaultdict(list)
 for r in intervention:contexts[(r['seed'],r['target'])].append(r)
 common={h:sum(r['success'] for r in intervention if abs(r['grasp']-.1)<1e-6 and abs(r['hook']-h)<1e-6) for h in [-.055,-.065,-.075]}
 reversals=[]
 for seed in sorted(set(r['seed'] for r in intervention)):
  ys={y:{(round(r['grasp'],4),round(r['hook'],4)):r['success'] for r in contexts[(seed,y)]} for y in [-.08,.08]}
  for left,right in itertools.combinations(sorted(set(ys[-.08])&set(ys[.08])),2):
   da=ys[-.08][left]-ys[-.08][right];db=ys[.08][left]-ys[.08][right]
   if da*db<0:reversals.append(dict(seed=seed,action_a=left,action_b=right,minus_target_difference=da,plus_target_difference=db))
 methods=defaultdict(list)
 for r in rows:
  if r['group']=='corrected112':methods[r['method']].append(r)
 totals={k:dict(n=len(v),success=sum(r['success'] for r in v),native=sum(r['native'] for r in v),mean_wall_s=float(np.mean([r['wall_s'] for r in v])),failures=dict(Counter(r['failure'] for r in v)),selected_action_brier=None if any(r['predicted_success'] is None for r in v) else float(np.mean([(r['predicted_success']-r['success'])**2 for r in v]))) for k,v in methods.items()}
 paired={}
 if 'ordinary' in methods:
  base={(r['seed'],r['mass'],r['target']):r for r in methods['ordinary']}
  for name,rs in methods.items():
   rescue=regress=prefix_matches=0
   for r in rs:
    b=base[(r['seed'],r['mass'],r['target'])];assert r['initial_sha256']==b['initial_sha256'];rescue+=r['success']>b['success'];regress+=r['success']<b['success']
    if name in ['ordinary','generic','contract','relative','relative_contract']:
     rd=json.load(gzip.open(r['path'],'rt'));bd=json.load(gzip.open(b['path'],'rt'));k=rd['decisions'][1]['boundary_step'] if len(rd['decisions'])>1 else 100;assert rd['steps'][:k]==bd['steps'][:k];prefix_matches+=1
   paired[name]=dict(rescue=rescue,regression=regress,exact_prefix_matches=prefix_matches)
 diagnostic={m:dict(n=sum(r['group']=='eval112' and r['method']==m for r in rows),success=sum(r['success'] for r in rows if r['group']=='eval112' and r['method']==m)) for m in sorted(set(r['method'] for r in rows if r['group']=='eval112'))}
 report=dict(diagnostic_probability_rank=diagnostic,episodes=len(rows),physics_steps=100*len(rows),intervention_success=sum(r['success'] for r in intervention),contexts=len(contexts),common_fixed100_success=common,retrospective_oracle_context_success=sum(max(r['success'] for r in v) for v in contexts.values()),strict_goal_ranking_reversals=reversals,methods=totals,paired_vs_ordinary=paired,limits='Oracle is retrospective measured-action upper bound, not policy. 96 rows are 8 scenes, not 96 independent trials. Mass is between-scene balanced. Evaluation is a bounded pilot, not final multi-seed proof.')
 (a.out/'SUMMARY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 fig,axes=plt.subplots(1,3,figsize=(17,5.7),layout='constrained');fig.suptitle('Actual physical evidence | Independent method gain: UNVERIFIED',fontsize=14)
 regress=json.loads((a.out/'REGRESSIONS.json').read_text());bad=[r for r in regress if r['open_success'] and not r['posterior_success']]
 for r in bad:axes[0].semilogy(r['grid_hook_mm'],r['grid_q'],label=r['case'].replace('_stap',''));axes[0].scatter([r['prior_hook_mm']],[r['posterior_q_retained']],marker='x',s=55)
 axes[0].set(xlabel='Hook offset (mm)',ylabel='Posterior predicted success',title='A | Low-Q argmax replaces successful plan');axes[0].legend(fontsize=7,title='Cross = retained prior action',title_fontsize=8)
 keys=sorted(contexts);data=[]
 for key in keys:
  rs=contexts[key];gs=sorted(set(r['grasp'] for r in rs));data.append([next(r['success'] for r in rs if r['grasp']==g and abs(r['hook']-h)<1e-6) for g in gs for h in [-.075,-.065,-.055]])
 axes[1].imshow(data,vmin=0,vmax=1,cmap='Blues',aspect='auto');axes[1].set(xticks=range(6),xticklabels=['low/-75','low/-65','low/-55','high/-75','high/-65','high/-55'],yticks=range(len(keys)),yticklabels=[f'{s}: {y:+.2f}' for s,y in keys],title='B | Actual outcomes (blue = success)');axes[1].tick_params(axis='x',rotation=60,labelsize=8);axes[1].tick_params(axis='y',labelsize=7)
 if totals:
  names=list(totals);axes[2].barh(names,[totals[k]['success']/totals[k]['n'] for k in names],color='#287a9b');axes[2].set(xlim=(0,1),xlabel='Actual full-task success',title='C | New-scene physical policy comparison')
  axes[2].tick_params(axis='y',labelsize=8)
  for i,k in enumerate(names):axes[2].text(.02,i,f"{totals[k]['success']}/{totals[k]['n']}",va='center')
 fig.savefig(a.out/'KEY_FIGURE.png',dpi=300);fig.savefig(a.out/'KEY_FIGURE.pdf');plt.close(fig)
if __name__=='__main__':main()
