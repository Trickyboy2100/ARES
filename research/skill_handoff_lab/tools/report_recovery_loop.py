"""Source-only research report/figures. No fabricated metrics or simulator calls."""
import argparse,json,gzip,hashlib,collections,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from program_charger_mechanism import arrays

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;figs=root/'figures';figs.mkdir(exist_ok=True);seen={};groups={};sources=[]
 for f in sorted(root.rglob('RESULTS.json')):
  if 'ORIGINAL_METADATA' in f.relative_to(root).parts:continue
  rr=json.loads(f.read_text())
  if not isinstance(rr,list):continue
  for row in rr:
   if not row.get('path'):continue
   path=Path(row['path'])
   if str(path)in seen:continue
   assert sha(path)==row['sha256'];d=json.load(gzip.open(path,'rt'));h=100 if 'tool_'in path.parent.name else 200
   assert len(d['steps'])==h;assert bool(d['steps'][-1]['native_success'])==bool(row['success']);group=str(path.relative_to(root).parts[0]);method=row.get('method',row.get('mode',row.get('policy','unknown')))
   if group=='charger_eval':method=path.parent.name
   seen[str(path)]=dict(row,group=group,method=method);sources.append(dict(path=str(path.relative_to(root)),sha256=sha(path),steps=h))
 for z in seen.values():
  k=z['group']+'/'+z['method'];g=groups.setdefault(k,dict(episodes=0,success=0,steps=0,planner_queries=0,planning_wall_s=0.,wall_s=0.,scene_seeds=set(),errors=collections.Counter(),selected_rl_decisions=0));g['episodes']+=1;g['success']+=bool(z['success']);g['steps']+=z['steps'];g['planner_queries']+=z.get('planner_queries',z.get('plan_calls',0));g['planning_wall_s']+=z.get('query_wall_s',z.get('planning_wall_s',0));g['wall_s']+=z['wall_s'];g['scene_seeds'].add(z['seed']);g['errors'][str(z.get('error'))]+=1;g['selected_rl_decisions']+=z.get('rl_decisions',0)
 for g in groups.values():g['scene_seeds']=sorted(g['scene_seeds']);g['errors']=dict(g['errors'])
 train={};partial=[]
 for folder in sorted(root.glob('charger_sac*')):
  checkpoints=[]
  for f in sorted(folder.glob('*.zip')):
   d=json.loads(zipfile.ZipFile(f).read('data'));checkpoints.append(dict(file=str(f.relative_to(root)),sha256=sha(f),decisions=d.get('num_timesteps'),updates=d.get('_n_updates')))
  if checkpoints:train[folder.name]=checkpoints
  f=folder/'PARTIAL.json.gz'
  if f.exists():
   d=json.load(gzip.open(f,'rt'));partial.append(dict(file=str(f.relative_to(root)),sha256=sha(f),steps=len(d['steps']),planning_queries=len(d.get('calls',[]))))
 summary=dict(method_claim='UNVERIFIED_INDEPENDENT_ALGORITHM_GAIN',groups=groups,complete_episodes=len(seen),complete_control_steps=sum(z['steps']for z in seen.values()),partial=partial,training=train,sources=sources,scope='Public PhysX: native tasks and explicitly named construction-time variant; no hardware, sealed data or ARES-R edits. Development, diagnostic and training episodes are not independent tests.')
 (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
 # Exact same-prefix contact mechanism, not an algorithm comparison.
 rr=json.loads((root/'charger_matched/RESULTS.json').read_text());selected=[next(r for r in rr if r['seed']==749101 and r['policy']==m)for m in ['nominal','fast','soft']];colors=['#187b80','#d95442','#7162a5'];fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
 for row,col in zip(selected,colors):
  d=json.load(gzip.open(row['path'],'rt'));aa=arrays(d);t=np.arange(1,201)
  for ax,key in zip(axs,['x','angle','force']):ax.plot(t,aa[key],label=row['policy']+f" (native final={int(row['success'])})",color=col,lw=1.8)
 for ax,label in zip(axs,['Axial position (mm)','Orientation error (rad)','20 Hz contact proxy (N)']):ax.set_ylabel(label);ax.axvspan(0,131,color='#cbd5e1',alpha=.35);ax.axvline(131,ls=':',color='black');ax.grid(alpha=.2)
 axs[0].legend(loc='best');axs[-1].set_xlim(125,200);axs[-1].set_xlabel('Actual native control step (20 Hz; contact-window zoom; full horizon 200)');fig.suptitle('Same physical grasp + approach prefix; different insertion consequences\nSeed 749101, 45° grasp | DATA_DERIVED_PLOT | NO NEW RL CLAIM');fig.savefig(figs/'CAUSAL_TRACE.png',dpi=220);fig.savefig(figs/'CAUSAL_TRACE.svg');plt.close(fig)
 # Compact key result updated when real evaluation exists.
 fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
 modes=['none','rehook','retract','press','hold','feedback'];vv=[groups['tool_pilot/'+m]['success']for m in modes];axs[0].bar(range(len(modes)),vv,color='#397b9e');axs[0].set_xticks(range(len(modes)),['Continue','Rehook','Retract','Press','Wait','Feedback'],rotation=35,ha='right');axs[0].set_ylim(0,8.8);axs[0].set_title('Tool: shared strong 87.5 mm grasp');axs[0].set_ylabel('Native final successes / 8 scenes')
 for i,v in enumerate(vv):axs[0].text(i,v+.12,f'{v}/8',ha='center')
 factorial=json.loads((root/'charger_factorial/RESULTS.json').read_text());vals=[next(r for r in factorial if r['seed']==749101 and r['policy']==m)['final']['obj_to_goal_dist'][0]*1000 for m in ['nominal','advance_fast','rotate_fast','rotate_soft']];axs[1].bar(['Nominal','Advance\n4 mm','Rotate\n0.08 rad','Rotate\n0.01 rad'],vals,color=['#187b80','#4094a8','#d95442','#7162a5']);axs[1].axhline(5,ls=':',color='#333',label='native position limit');axs[1].set_ylabel('Terminal position error (mm)');axs[1].set_title('Charger: one exact-prefix causal case');axs[1].legend(fontsize=8)
 for i,v in enumerate(vals):axs[1].text(i,v+1,f'{v:.1f}',ha='center')
 ev=[(k.split('/')[-1],v)for k,v in groups.items()if k.startswith('charger_eval/')]
 if ev:
  axs[2].bar(range(len(ev)),[v['success']for k,v in ev],color=['#a57b3a'if 'sparse'in k or 'dense'in k else '#397b9e'for k,v in ev]);axs[2].set_xticks(range(len(ev)),[{'nominal':'Nominal','force_guard':'Force rule','dense_256':'Legacy reward\n256','sparse_256':'Native SAC\n256','sparse_1024':'Native SAC\n1024','matched_sparse_1024':'Same grasp\nSAC1024'}.get(k,k)for k,v in ev],rotation=45,ha='right');axs[2].set_ylim(0,max(v['episodes']for _,v in ev)+1)
  for i,(k,v)in enumerate(ev):axs[2].text(i,v['success']+.12,f"{v['success']}/{v['episodes']}",ha='center',fontsize=8)
  axs[2].set_title('Charger: whole-task development');axs[2].set_ylabel('Native final successes')
 else:axs[2].axis('off');axs[2].text(.05,.5,'Ordinary RL evaluation pending\nNo estimated outcome displayed',fontsize=12)
 fig.suptitle('Recovery and cross-skill physics | Independent method gain UNVERIFIED',fontsize=14);fig.savefig(figs/'KEY_RESULT.png',dpi=220);fig.savefig(figs/'KEY_RESULT.svg');plt.close(fig)
 print(json.dumps({k:v for k,v in summary.items()if k not in ['sources','groups','training']},indent=2))
if __name__=='__main__':main()
