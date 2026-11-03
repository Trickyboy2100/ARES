"""Independently recompute raw physics, paired-prefix and optimizer accounting."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;physics={};sources=[];invalid=[]
for f in sorted(r.glob('*/RESULTS.json')):
 rows=json.loads(f.read_text())
 for z in rows:
  path=Path(z['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==z['sha256'];d=json.load(gzip.open(path,'rt'));assert len(d['steps'])==100
  if 'evaluation'in d['steps'][-1]:assert bool(d['steps'][-1]['evaluation']['success'][0])==bool(z['success'])
  else:assert bool(d['steps'][-1]['native_success'])==bool(z['success'])
  group=z.get('method',f.parent.name);m=physics.setdefault(group,dict(n=0,success=0,steps=0,wall_s=0.));m['n']+=1;m['success']+=bool(z['success']);m['steps']+=100;m['wall_s']+=z['wall_s'];sources.append(dict(path=str(path.relative_to(r)),sha256=z['sha256']))
  if f.parent.name=='official_gpu_obb':invalid.append(str(path.relative_to(r)))
pairs=[];matrix=[]
for prefix,offsets in [('grid_', ['-0.01','0.01']),('new_', ['-0.01','0','0.01'])]:
 for offset in offsets:
  left=r/f'{prefix}{offset}_nominal/RESULTS.json';right=r/f'{prefix}{offset}_refresh/RESULTS.json'
  for x,y in zip(json.loads(left.read_text()),json.loads(right.read_text())):
   d=json.load(gzip.open(x['path'],'rt'));e=json.load(gzip.open(y['path'],'rt'));assert d['initial']==e['initial'];n=next((c['start']for c in x['calls']if c['phase']=='INSERT'),100);assert d['steps'][:n]==e['steps'][:n]
   pairs.append(dict(seed=x['seed'],offset=float(offset),prefix_steps=n,exact_prefix=True,nominal=x['success'],refresh=y['success'],rescued=not x['success']and y['success'],regressed=x['success']and not y['success']))
for name in ['gpu_retime','gpu_refresh','grid_-0.01_nominal','grid_-0.01_refresh','grid_0.01_nominal','grid_0.01_refresh']+[f'new_{g}_{m}'for g in ['-0.01','0','0.01']for m in ['nominal','refresh']]:
 manifest=json.loads((r/name/'MANIFEST.json').read_text());offset=manifest.get('grasp_offset',0.);refresh=manifest.get('refresh_terminal',name=='gpu_refresh')
 for z in json.loads((r/name/'RESULTS.json').read_text()):matrix.append(dict(seed=z['seed'],offset=offset,refresh=refresh,success=z['success']))
fixed=collections.defaultdict(int);oracle={}
for x in matrix:fixed[str((x['offset'],x['refresh']))]+=x['success'];oracle[x['seed']]=max(oracle.get(x['seed'],0),x['success'])
training={};sacsteps=0;othersteps=0
for folder in sorted(r.iterdir()):
 if not folder.is_dir()or not(folder/'TRAINING.json').exists():continue
 logs=json.loads((folder/'TRAINING.json').read_text());latest={}
 for z in logs:
  key=(z.get('method'),z.get('fold'),z.get('seed'));latest[key]=z
 count=sum(z.get('updates',z.get('critic_updates',0))for z in latest.values());training[folder.name]=dict(runs=len(latest),update_iterations=count,wall_s=sum(z['wall_s']for z in latest.values()),kind='actor_and_critic_each'if any('critic_updates'in z for z in logs)else 'joint_value_optimizer')
 if training[folder.name]['kind']=='actor_and_critic_each':sacsteps+=count
 else:othersteps+=count
out=dict(method_claim='UNVERIFIED',physics=physics,complete_episodes=sum(x['n']for x in physics.values()),control_steps=sum(x['steps']for x in physics.values()),invalid_instrumentation_episodes=invalid,training=training,sac_actor_steps=sacsteps,sac_critic_steps=sacsteps,joint_value_optimizer_steps=othersteps,paired_insert=pairs,peg_matrix=matrix,best_fixed_success=max(fixed.values()),fixed_success=dict(fixed),parameter_oracle=sum(oracle.values()),scene_count=len(oracle),raw_sources=sources,notes=['Native success and native100 preserved','Not all episodes independent: cluster by physical scene','Official GPU OBB compatibility failure is not valid nominal control evidence','Peg call wall includes execution, not isolated planning time','Logged candidate CV is not fresh physical evaluation'])
(r/'ANALYSIS.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items()if k in ['complete_episodes','control_steps','sac_actor_steps','joint_value_optimizer_steps','best_fixed_success','parameter_oracle','scene_count']})
