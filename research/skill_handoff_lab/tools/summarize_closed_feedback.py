"""Recompute continuous native evaluations and separately cost local training."""
import argparse,collections,gzip,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
allcost=collections.defaultdict(lambda:collections.Counter());errors=[]
for f in sorted(r.rglob('*.json.gz')):
 if 'prior'in f.relative_to(r).parts:continue
 try:d=json.load(gzip.open(f,'rt'))
 except Exception as ex:errors.append(dict(path=str(f),error=str(ex)));continue
 if 'steps'not in d:continue
 s=d.get('summary',d);scope=s.get('scope','FULL_NATIVE');cc=allcost[scope];cc['files']+=1;cc['physical_steps']+=len(d['steps']);cc['decisions']+=len(s.get('decisions',[]));cc['planning_queries']+=len(s.get('queries',[]));cc['planning_wall_s']+=sum(q.get('wall_s',0)for q in s.get('queries',[]))
 if s.get('error')and s['error']not in ['HORIZON','PLAN_FAILED']:errors.append(dict(path=str(f.relative_to(r)),error=s['error']))
def score(rows):
 rear=[z for x in rows for z in x['decisions']if z['stage']==1]
 return dict(n=len(rows),success=sum(x['success']for x in rows),success_scenes=[x['seed']for x in rows if x['success']],front_actions={str(x['seed']):x['decisions'][0]['id']for x in rows},rear_actions=dict(collections.Counter(z['id']for z in rear)),active_correction_scenes=[x['seed']for x in rows if any(z['stage']==1 and z['id']in range(1,7)for z in x['decisions'])],physical_steps=sum(x['steps']for x in rows),planning_queries=sum(len(x['queries'])for x in rows),planning_wall_s=sum(q['wall_s']for x in rows for q in x['queries']))
methods={};comparisons={}
for mode in ['ordinary','suffix','budget']:
 d=r/f'train_{mode}_v4_31501';local=r/f'local_{mode}_31501';ss={}
 for f in sorted(d.glob('EVAL_*.json')):ss[f.stem[5:]]=score(json.loads(f.read_text()))
 loc={}
 for f in sorted(local.glob('EVAL_*.json')):loc[f.stem[5:]]=score(json.loads(f.read_text()))
 methods[mode]=dict(full_native_training_evaluations=ss,local_training_native_evaluations=loc,local_training_cost=json.loads((local/'COST.json').read_text())if(local/'COST.json').exists()else None)
 if (local/'LOCAL_ROWS.json').exists():
  rr=json.loads((local/'LOCAL_ROWS.json').read_text());methods[mode]['local_training_success']=dict(n=len(rr),success=sum(x['success']for x in rr),note='Training initialized suffixes; NOT full-task benchmark success.')
# Failed evaluation interrupted suffix training after a valid checkpoint.
resume=r/'local_suffix_resume_31501'
if resume.exists():
 v=methods['suffix'];recovery=json.loads((r/'local_suffix_recovered/RECOVERY.json').read_text());offset=recovery['actual_training_steps']
 for f in sorted(resume.glob('EVAL_*.json')):
  k=f.stem[5:];v['local_training_native_evaluations'][str(int(k)+offset)if k!='FINAL'else k]=score(json.loads(f.read_text()))
 if (resume/'COST.json').exists():
  c=json.loads((resume/'COST.json').read_text());c['resumed_segment_steps']=c['actual_training_steps'];c['actual_training_steps']+=offset;c['optimizer_updates']+=2560;c['recovery']=recovery;v['local_training_cost']=c
for mode in methods:
 d=r/f'deployment_labels_{mode}';methods[mode]['deployment_label_ablation']={f.stem[5:]:score(json.loads(f.read_text()))for f in d.glob('EVAL_*.json')}
 if (d/'COST.json').exists():methods[mode]['deployment_label_cost']=json.loads((d/'COST.json').read_text())
for name in ['96','local_FINAL']:
 base=methods['ordinary']['full_native_training_evaluations'].get(name)if not name.startswith('local')else methods['ordinary']['local_training_native_evaluations'].get('FINAL')
 if base:
  for mode in ['suffix','budget']:
   z=methods[mode]['full_native_training_evaluations'].get(name)if not name.startswith('local')else methods[mode]['local_training_native_evaluations'].get('FINAL')
   if z:
    aa=set(base['success_scenes']);bb=set(z['success_scenes']);comparisons[mode+'_'+name]=dict(rescued=sorted(bb-aa),regressed=sorted(aa-bb),note='Same native initial states; different predecessors/policies, not identical handoffs.')
result=dict(status='UNVERIFIED_INDEPENDENT_ALGORITHM_GAIN',methods=methods,paired= comparisons,actual_physics_by_scope={k:dict(v)for k,v in allcost.items()},cost_completeness='Archived steps exactly counted; failed CUDA evaluation contains an unarchived partial attempt, so total attempted physical steps are a lower bound.',exceptions_preserved=errors,note='Local initialization is training only, charged prefix once and every actual suffix step. Full evaluation has no handoff reset. Seen development scenes, one training seed; no sealed tests.')
(r/'RESULTS_SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:{'native':{n:z['success']for n,z in v['full_native_training_evaluations'].items()},'local':{n:z['success']for n,z in v['local_training_native_evaluations'].items()}}for k,v in methods.items()},indent=2))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
(r/'figures').mkdir(exist_ok=True);fig,axs=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for mode,color in [('ordinary','#2865a0'),('suffix','#33835a'),('budget','#bd671c')]:
 v=methods[mode];ss=v['full_native_training_evaluations'];ks=sorted(map(int,ss));axs[0].plot([n*100 for n in ks],[ss[str(n)]['success']/16 for n in ks],'o-',color=color,label=mode)
 loc=v['local_training_native_evaluations'];ks=sorted(int(k)for k in loc if k!='FINAL');xx=[0]+ks;yy=[ss['96']['success']/16]+[loc[str(k)]['success']/16 for k in ks]
 if 'FINAL'in loc:xx.append(v['local_training_cost']['actual_training_steps']);yy.append(loc['FINAL']['success']/16)
 axs[1].plot(xx,yy,'o-',color=color,label=mode)
for ax,title in zip(axs,['Continuous native-prefix training','Additional reachable-handoff training']):ax.set(xlabel='Actual additional training control steps',ylabel='Native100 development success (16 scenes)',ylim=(0,1),title=title);ax.grid(alpha=.2);ax.legend()
fig.suptitle('Closed-loop skill learning | real physics | one seed | no sealed tests');fig.tight_layout();fig.savefig(r/'figures/LEARNING.png',dpi=180);plt.close(fig)
