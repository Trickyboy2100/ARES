"""Declared single-factor acquisition-distribution revision; no eval-label fitting.
All revision model hashes frozen before loading the previous real branch panel.
"""
import argparse,json,hashlib,time,gzip
from pathlib import Path
import torch
from program_discrete_options import Learner,prior,normalizer,episode_transitions
from program_peg_option_runtime import Runtime
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'anchor_eval';out.mkdir(exist_ok=False);torch.set_num_threads(2);models=[]
for seed in [31301,31302]:
 for mode in ['joint','independent_kernel']:models.append((f'frontier_current_{mode}',seed,r/f'frontier_current_{mode}_{seed}/FINAL.pt'))
while not all(p.exists()for _,_,p in models):time.sleep(5)
data,_=prior(a.prior);training=[]
for seed in [31301,31302]:
 online=json.loads((r/f'frontier_current_joint_{seed}/ONLINE.json').read_text());paths=list(dict.fromkeys(t['source']for t in online))
 for kind in ['supervised','fqi','iql','cql','stap']:
  m=Learner(kind,seed,normalizer(data));m.train(data,steps=800);buf=[]
  for k,path in enumerate(paths):
   buf.extend(t for t in online if t['source']==path)
   if k<8:m.train(data,buf,steps=50)
   elif k%2==1:m.train(data,buf,steps=100)
  name=f'matched_current_joint_{kind}';p=out/f'{name}_{seed}.pt';m.save(p);models.append((name,seed,p));training.append(dict(kind=kind,seed=seed,updates=m.updates,history=m.history,transitions=len(buf)))
(out/'MATCHED_TRAINING.json').write_text(json.dumps(training,indent=2))
(out/'FROZEN_MODELS.json').write_text(json.dumps(dict(protocol='additional current-distribution revision; all hashes frozen before reading existing branch outcomes; not part of original54-model freeze',models=[dict(method=n,seed=s,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for n,s,p in models]),indent=2))
# Original evaluation finishes first; no simultaneous evaluation GPU load.
while True:
 try:
  if len(json.loads((r/'expansion_eval/SCORES.json').read_text()))==60:break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(5)
panel={}
for row in json.loads((r/'expansion_eval/physics/RESULTS.json').read_text()):
 if not row['tag'].startswith('fixed'):continue
 aa,bb=map(int,row['tag'][-2:]);d=json.load(gzip.open(row['path'],'rt'));panel[row['seed'],aa,bb]=(row,episode_transitions(d,[-.01,0,.01][aa],bb,row['path']),d)
rt=Runtime(out/'physics');pred=[];scores=[]
try:
 for name,seed,p in models:
  m=Learner.load(p);rows=[];direct=name.startswith('frontier_')
  for scene in range(751220,751228):
   aa=m.choose(panel[scene,0,0][1][0]['s']);ts=panel[scene,aa,0][1];bb=m.choose(ts[1]['s'])if len(ts)==2 else 0;reference,_,raw=panel[scene,aa,bb];entry=dict(method=name,seed=seed,scene=scene,actions=[aa,bb]if len(ts)==2 else[aa],success=reference['success'],reference=reference['path'],reference_sha=reference['sha256'],direct_checked=False)
   if direct:
    row,_,d=rt.run(scene,m.choose,f'{name}_{seed}');assert d['initial']==raw['initial']and d['steps']==raw['steps'];entry.update(direct_checked=True,direct_source=row['path'],all100steps_equal=True);rows.append(row)
   else:rows.append(reference)
   pred.append(entry)
  scores.append(dict(method=name,seed=seed,success=sum(x['success']for x in rows),n=8,sources=[x['path']for x in rows],evaluation='DIRECT_NATIVE_PHYSICS_EXACT_BRANCH_CHECK'if direct else'FROZEN_POLICY_ON_REAL_FINITE_BRANCH_PANEL_NOT_NEW_ROLLOUT',protocol='additional current-anchor revision'))
  (out/'SCORES.json').write_text(json.dumps(scores,indent=2));(out/'BRANCH_PREDICTIONS.json').write_text(json.dumps(pred,indent=2))
finally:rt.close()
