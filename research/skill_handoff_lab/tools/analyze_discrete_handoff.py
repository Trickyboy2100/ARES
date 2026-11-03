"""Raw-source verification and independent physical/cost/selection accounting."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np
from program_discrete_options import episode_transitions

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;allrows=[];runs={};buffers=[];pairs=[]
 if(root/'DISTRIBUTION_REVISION_PROTOCOL.json').exists():assert len(json.loads((root/'anchor_eval/SCORES.json').read_text()))==14,'REVISION_EVALUATION_NOT_COMPLETE'
 for res in sorted(root.glob('*/physics/RESULTS.json')):
  name=res.parent.parent.name;rows=json.loads(res.read_text());raws={}
  for row in rows:
   path=Path(row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];d=json.load(gzip.open(path,'rt'));assert len(d['steps'])==row['steps']==100;assert bool(d['steps'][-1]['evaluation']['success'][0])==row['success'];assert row['error']in [None,'HORIZON','PLAN_FAILED'];raws[str(path)]=d
  allrows+=rows;training=[r for r in rows if r['tag'].startswith(('train','warmup','pair'))];evals=[r for r in rows if r['tag'].startswith('dev')]
  run=dict(episodes=len(rows),physical_steps=sum(r['steps']for r in rows),training_episodes=len(training),training_steps=sum(r['steps']for r in training),training_decisions=sum(len(r['decisions'])for r in training),eval_steps=sum(r['steps']for r in rows if not r['tag'].startswith(('train','warmup','pair'))),eval_scope='includes fixtures and fixed-branch assessment; separate from training',planner_queries=sum(len(r['queries'])for r in rows),planning_s=sum(q['wall_s']for r in rows for q in r['queries']),wall_s=sum(r['wall_s']for r in rows),first_actions=dict(collections.Counter(r['decisions'][0]['id']for r in training)),second_actions=dict(collections.Counter(r['decisions'][1]['id']for r in training if len(r['decisions'])>1)))
  run['training_handoffs']=sum(len(x['decisions'])==2 for x in training)
  run['training_handoffs_not_grasped']=sum(len(x['decisions'])==2 and x['decisions'][1]['obs'][42]<.5 for x in training)
  run['training_native_successes']=sum(x['success']for x in training)
  hfile=res.parent.parent/'HISTORY.json'
  if hfile.exists():
   hh=json.loads(hfile.read_text());run['optimizer_iterations']=hh[-1]['update'];run['training_compute_s']=sum(x['wall_s']for x in hh)
  acqs=[json.loads(x.read_text())for x in res.parent.parent.glob('ACQUIRE_*.json')];run['acquisition_compute_s']=sum(x.get('virtual_wall_s',x.get('wall_s',0.))for x in acqs if isinstance(x,dict))
  cf=res.parent.parent/'CURVE.json'
  if cf.exists():
   curve=json.loads(cf.read_text());run['curve']=curve;initial={r['seed']:r['success']for r in evals if r['tag']=='dev0'};changes=[]
   for c in curve:
    rs=[r for r in evals if r['tag']==f"dev{c['episodes']}"];assert c['success']==sum(r['success']for r in rs)
    changes.append(dict(episodes=c['episodes'],rescues=sum(r['success']and not initial[r['seed']]for r in rs),regressions=sum(not r['success']and initial[r['seed']]for r in rs),by_scene={r['seed']:dict(success=r['success'],actions=[d['id']for d in r['decisions']])for r in rs}))
   run['paired_to_initial']=changes
  bf=res.parent.parent/'ONLINE.json'
  if bf.exists():
   online=json.loads(bf.read_text());reconstructed=[]
   for r in training:
    d=raws[r['path']];act=r['decisions'];offset=[-.01,0.,.01][act[0]['id']];reconstructed+=episode_transitions(d,offset,act[-1]['id']if len(act)>1 else 0,r['path'])
   # A running job may not yet have written its final episode buffer. Final audit
   # requires exact lengths and values; call only after all jobs complete.
   assert online==reconstructed,(name,len(online),len(reconstructed));buffers.append(dict(run=name,transitions=len(online),exact=True))
  pairids=sorted({r['tag'].rsplit('_',1)[0]for r in rows if r['tag'].startswith('pair')})
  for pairid in pairids:
   rr=[r for r in rows if r['tag'].startswith(pairid+'_')];assert len(rr)==2
   dd=[raws[r['path']]for r in rr];n=rr[0]['decisions'][1]['step']if len(rr[0]['decisions'])>1 else 100;assert dd[0]['initial']==dd[1]['initial']and dd[0]['steps'][:n]==dd[1]['steps'][:n];pairs.append(dict(run=name,pair=pairid,scene=rr[0]['seed'],prefix_steps=n,outcomes=[r['success']for r in rr],cost=200,exact=True))
  runs[name]=run
 # Classify training conditions by actually measured choices only; no oracle use
 # during acquisition. Distinguish fixed coverage from hindsight union.
 by=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in allrows:
  if r['tag'].startswith(('train','warmup','pair')):
   acts=tuple(d['id']for d in r['decisions']);by[r['seed']][str(acts)].append(int(r['success']))
 conditions={str(s):dict(actions=dict(v),observed_union=max(max(x)for x in v.values()),mixed=len({y for x in v.values()for y in x})>1)for s,v in by.items()}
 summary=dict(runs=runs,total_complete_episodes=len(allrows),total_physics_steps=sum(r['steps']for r in allrows),new_training_scenes=sorted(by),conditions=conditions,buffers=buffers,strict_pairs=pairs,claim='UNVERIFIED',native_task='PegInsertionSide-v1 native100',cost_note='all pair prefixes charged; dev and training separated; planning seconds overlap concurrent jobs walltime, not isolated speed benchmark')
 (root/'ANALYSIS.json').write_text(json.dumps(summary,indent=2));print('verified',len(allrows),'episodes',len(buffers),'buffers',len(pairs),'pairs')
if __name__=='__main__':main()
