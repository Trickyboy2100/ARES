"""Publish auditable episode classifications and standalone scientific figures."""
import argparse,gzip,json,csv,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True);rows=[];summary={};phase_rows=[]
 for path in sorted(a.root.glob('*/RESULTS.json')):
  rs=json.loads(path.read_text());g=path.parent.name;summary[g]={}
  for r in rs:
   t=load(path.parent/f"{r['seed']}_{r['mode']}.json.gz");ss=t['steps'];lift=[s for s in ss if s['phase']=='LIFT'];active=[s for s in ss if s['phase']!='TERMINAL_HOLD'];end=ss[-1];phase=active[-1]['phase'] if active else 'NO_EXECUTED_PREFIX';charger='charger' in g;prefix=bool(lift and lift[-1]['grasped']);reached=any(s['phase'].startswith('INSERT') if charger else s['phase']=='PULL' for s in ss)
   # Operational categories, NOT a causal claim about why a skill failed.
   failure='SUCCESS' if r['success'] else ('PREFIX_NOT_VERIFIED' if not prefix and r['mode']!='official' else ('PREFIX_OR_TRANSFER_BUDGET' if not reached else ('SUFFIX_NO_GRASP_SIGNAL' if not end['grasped'] else 'SUFFIX_GOAL_MISS_WITH_GRASP')))
   for ph in dict.fromkeys(z['phase'] for z in ss):
    phase_rows.append(dict(group=g,seed=r['seed'],method=r['mode'],phase=ph,native_steps=sum(z['phase']==ph for z in ss),simulated_s=sum(z['phase']==ph for z in ss)/20,planning_wall_s=sum(z.get('planning_wall_s',0) for z in t['plans'] if z['phase']==ph)))
   final=r.get('final',{});dist=float(final['obj_to_goal_dist'][0]) if charger else end['cube_base_distance_m'];ang=float(final['obj_to_goal_angle'][0]) if charger else None
   row=dict(group=g,seed=r['seed'],method=r['mode'],native_success=int(r['success']),error=r['error'],classification=failure,last_active_phase=phase,verified_lift=prefix,reached_final_skill=reached,final_grasped=end['grasped'],position_metric_m=dist,angle_error_rad=ang,wall_s=r['wall_s'],planning_wall_s=r['planning_wall_s'],steps=r['steps']);rows.append(row)
   mm=summary[g].setdefault(r['mode'],dict(n=0,success=0,failures=collections.Counter(),wall_s=[],planning_s=[]));mm['n']+=1;mm['success']+=int(r['success']);mm['failures'][failure]+=1;mm['wall_s'].append(r['wall_s']);mm['planning_s'].append(r['planning_wall_s'])
 for methods in summary.values():
  for v in methods.values():
   for k in ['wall_s','planning_s']:v[k]=dict(mean=float(np.mean(v[k])),median=float(np.median(v[k])),max=float(np.max(v[k])))
 with (a.out/'FAILURE_CLASSES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with (a.out/'PHASE_COSTS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=phase_rows[0]);w.writeheader();w.writerows(phase_rows)
 ev=[r for r in rows if r['group']=='charger_evaluation8'];paired={}
 if ev:
  seeds=sorted({r['seed'] for r in ev});methods=list(dict.fromkeys(r['method'] for r in ev));lookup={(r['seed'],r['method']):r for r in ev}
  for m in methods:
   pairs=[(lookup[s,'fixed_grid_search']['native_success'],lookup[s,m]['native_success']) for s in seeds if (s,m) in lookup];paired[m]=dict(n=len(pairs),rescues=sum(not b and y for b,y in pairs),regressions=sum(b and not y for b,y in pairs),both_success=sum(b and y for b,y in pairs),both_fail=sum(not b and not y for b,y in pairs))
  fig,ax=plt.subplots(figsize=(8,4.5));vals=[sum(r['native_success'] for r in ev if r['method']==m) for m in methods];ax.bar(range(len(methods)),vals,color=['#64748b','#94a3b8','#60a5fa','#818cf8','#0d9488']);ax.set_xticks(range(len(methods)),['Fixed grid\nsearch','Geometry\nreplan','Supervised','Ordinary\nfitted Q','Margin\nfitted Q']);ax.set_ylim(0,len(seeds)+1);ax.set_ylabel(f'Native successes / {len(seeds)} new initial states');ax.set_title('Matched action permissions and physics data; one training seed')
  for i,v in enumerate(vals):ax.text(i,v+.12,str(v),ha='center')
  fig.tight_layout();fig.savefig(a.out/'heldout_results.png',dpi=200);plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(methods));ax.bar(x-.18,[paired[m]['rescues'] for m in methods],.35,label='Rescues vs fixed',color='#0d9488');ax.bar(x+.18,[-paired[m]['regressions'] for m in methods],.35,label='Regressions vs fixed',color='#dc2626');ax.axhline(0,c='black',lw=.6);ax.set_xticks(x,['Fixed','Geometry','Supervised','Fitted Q','Margin FQ']);ax.set_ylabel('Paired cases');ax.set_title('Net success must not hide damage to successful cases');ax.legend();fig.tight_layout();fig.savefig(a.out/'paired_tradeoff.png',dpi=200);plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4));ax.bar(x,[summary['charger_evaluation8'][m]['wall_s']['median'] for m in methods],label='Whole episode wall time',color='#94a3b8');ax.bar(x,[summary['charger_evaluation8'][m]['planning_s']['median'] for m in methods],label='Planner calls within episode',color='#0d9488');ax.set_xticks(x,['Fixed','Geometry','Supervised','Fitted Q','Margin FQ']);ax.set_ylabel('Measured median seconds');ax.set_title('Single RTX3080; native horizon 200 = 10 simulated seconds');ax.legend();fig.tight_layout();fig.savefig(a.out/'compute_cost.png',dpi=200);plt.close(fig)
 design=[]
 for group in ['charger_design4','charger_train8']:
  design.extend(json.loads((a.root/group/'RESULTS.json').read_text()))
 design=[r for r in design if r['mode'].startswith('a')];seeds=sorted({r['seed'] for r in design});methods=list(dict.fromkeys(r['mode'] for r in design));lookup={(r['seed'],r['mode']):r for r in design};mat=np.array([[int(lookup[s,m]['success']) for m in methods] for s in seeds]);fig,ax=plt.subplots(figsize=(9,5));ax.imshow(mat,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(methods)),[m.replace('a','').replace('_d',' deg / ')+' m' for m in methods],rotation=40,ha='right');ax.set_yticks(range(len(seeds)),[str(s) for s in seeds]);ax.set_title('Training-design outcomes: measured union is NOT a deployable policy')
 for i in range(len(seeds)):
  for j in range(len(methods)):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j] else '#222')
 fig.tight_layout();fig.savefig(a.out/'charger_parameter_grid.png',dpi=200);plt.close(fig)
 pairs=[('pull','design8_v4_retry',712000,['g0.035_y0.00','g0.105_y0.00']),('charger','charger_design4',721000,['a45.0_d0.05','a45.0_d0.075'])]
 for task,g,seed,ms in pairs:
  fig,axes=plt.subplots(2,1,figsize=(7.5,6),sharex=True)
  for m in ms:
   t=load(a.root/g/f'{seed}_{m}.json.gz');ss=t['steps'];tt=np.arange(1,len(ss)+1)/20;toolx=[s['tcp_T_tool'][0]*1000 for s in ss];goal=np.array(t['initial_measure']['cube'][:3]);yy=[np.linalg.norm(np.array(s['tool'][:3])-goal)*1000 for s in ss] if task=='charger' else [(t['initial_measure']['cube'][0]-s['cube'][0])*1000 for s in ss];axes[0].plot(tt,toolx,label=m);axes[1].plot(tt,yy,label=m)
  axes[0].set_ylabel('Measured tool in TCP: x [mm]');axes[1].set_ylabel('Tool-to-goal distance [mm]' if task=='charger' else 'Cube displacement [mm]');axes[1].set_xlabel('Simulated seconds');axes[0].legend();axes[0].set_title(f'{task}: actual paired trajectories, seed {seed}')
  if task=='charger':axes[1].axhline(5,c='black',ls='--',label='5 mm goal tolerance');axes[1].legend()
  fig.tight_layout();fig.savefig(a.out/f'{task}_measured_states.png',dpi=200);plt.close(fig)
 result=dict(groups=summary,paired_vs_fixed=paired,completed_recorded_steps=sum(r['steps'] for r in rows),completed_episodes=len(rows),additional_debug_steps='one unsaved early charger empty-path crash: positive, at most 200; excluded from exact recorded count',training_positive_episodes=int(mat.sum()),training_grid_episodes=int(mat.size),training_best_fixed=int(mat.sum(0).max()),training_finite_union=int(mat.max(1).sum()))
 (a.out/'CORE_METRICS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))
if __name__=='__main__':main()
