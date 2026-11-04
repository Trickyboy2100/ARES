"""Shared-data Option SAC / witness BC / suffix-consistent improvement pilot.
No claim of novelty: pair preference and reference-KL have mature antecedents.
Only fresh physical pairs constrain updates; snapshots remain immutable.
"""
import argparse,json,copy,hashlib,shutil
from pathlib import Path
import numpy as np,torch
from program_budget_feedback_sac import FeedbackSAC,completion_labels
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rounds',type=int,default=8);p.add_argument('--resume',type=Path);p.add_argument('--joint',action='store_true');p.add_argument('--compare-joint',action='store_true');p.add_argument('--smdp-credit',action='store_true');p.add_argument('--dev-count',type=int,default=16);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004')
parent=root/'success_retention_20261104/controlled/FINAL.pt'
reference=FeedbackSAC.load(parent);models={k:FeedbackSAC.load(parent) for k in ['option_sac','witness_bc','consistent']}
if a.resume:
 for name in models:models[name]=FeedbackSAC.load(a.resume/f'{name}_FINAL.pt')
if a.compare_joint:
 assert a.resume
 models={name:FeedbackSAC.load(a.resume/'option_sac_FINAL.pt') for name in ['option_sac','witness_bc','consistent','consistent_joint']}
if a.smdp_credit:
 assert a.resume
 models={name:FeedbackSAC.load(a.resume/'option_sac_FINAL.pt') for name in ['option_sac','consistent','consistent_smdp']}
reference_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
rt=FeedbackRuntime(a.out/'physics');data=[];labels=[];pairs=[];history=[];evaluations=[]
offset=0
if a.resume:
 data=json.loads((a.resume/'TRANSITIONS.json').read_text());labels=json.loads((a.resume/'LABELS.json').read_text());offset=json.loads((a.resume/'PROTOCOL.json').read_text())['train_seeds'][1]-758000+1
protocol=dict(resume=str(a.resume),joint=a.joint,compare_joint=a.compare_joint,smdp_credit=a.smdp_credit,reference_sha=reference_sha,termination='insertion_queue',train_seeds=[758000+offset,758000+offset+a.rounds-1],dev_seeds=[758200,758200+a.dev_count-1],fixed_grasp=1,shared_dataset=True,old_primitive_replay_used=False,updates_per_round=80,auxiliary_actor_updates=40,all_native_steps=100,claim='UNVERIFIED',selection='first rear boundary; action cycles 1..7 independent of outcome')
(a.out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));src=a.out/'source';src.mkdir()
for f in [Path(__file__).name,'program_continuation_option.py','program_peg_feedback.py','program_budget_feedback_sac.py','program_peg_option_runtime.py']:shutil.copy2(Path(__file__).parent/f,src/f)

def run(seed, actor, tag):
 policy=ContinuationOption(lambda s:1 if s[44]<.5 else actor(s),lambda s:reference.choose(s),'insertion_queue')
 r,t,d=rt.run(seed,policy,tag);ts=policy.transitions(t)
 for x in ts:x['continuation_sha']=reference_sha
 return r,ts,d,policy

def evaluate(checkpoint):
 for seed in range(758200,758200+a.dev_count):
  for name,m in models.items():
   r,t,d,p=run(seed,m.choose,f'{name}_{checkpoint}');evaluations.append(dict(checkpoint=checkpoint,method=name,**r))
 (a.out/'EVALUATION.json').write_text(json.dumps(evaluations))

for i in range(a.rounds):
 seed=758000+offset+i;frozen=copy.deepcopy(models['option_sac']);branch=[]
 # Exact same original prefix; first rear callback is the intervention boundary.
 for action in [0,1+i%7]:
  first=[True]
  def actor(s):
   if first[0]:first[0]=False;return action
   return frozen.choose(s)
  r,ts,d,policy=run(seed,actor,f'train_{i}_a{action}');branch.append((r,ts,d))
  data.extend(ts);labels.extend(completion_labels(d,ts))
 (b,bt,bd),(v,vt,vd)=branch
 bx=next(x for x in bd['summary']['decisions'] if x['stage']==1);vx=next(x for x in vd['summary']['decisions'] if x['stage']==1);step=bx['step']
 assert bx['queue_before']==vx['queue_before'] and bx['obs']==vx['obs'] and bd['initial']==vd['initial']
 for x,y in zip(bd['steps'][:step],vd['steps'][:step]):
  for key in ['action','q','qvel','tcp','peg']:assert x[key]==y[key]
 delta=int(v['success'])-int(b['success']);pair=dict(s=bx['obs'],a=1+i%7,delta=delta,scene=seed,prefix_steps=step,base_path=b['path'],action_path=v['path'],reference_sha=reference_sha)
 pairs.append(pair)
 winner=vt if delta>0 else bt
 suffix=[t for t in winner if t['stage']==1 and t['step']>step] if delta else []
 anchors=[t for t in data if t['stage']==1 and t['outcome']]
 for name,m in models.items():
  before=m.updates
  log=m.train(data,labels,80,successor_only=True,success_bc=.2 if name=='witness_bc' else 0,bc_anchor=anchors or None)
  # Fixed additional iteration budget for all models. SAC gets normal updates;
  # candidate gets fresh signed intervention preference AND suffix-policy KL.
  snap=copy.deepcopy(frozen);credit_trace=[]
  for _ in range(40):
   rear=[t for t in data if t['stage']==1]
   batch=[rear[j] for j in m.rng.integers(len(rear),size=64)]
   xx=m.x([t['s'] for t in batch]);pp=m.probs(xx)
   with torch.no_grad():q=torch.stack([net(xx) for net in m.q]).mean(0)
   loss=(pp*(m.alpha*pp.clamp_min(1e-9).log()-q)).sum(1).mean()
   if name=='witness_bc' and anchors:
    aa=[anchors[j] for j in m.rng.integers(len(anchors),size=64)]
    ap=m.probs(m.x([t['s'] for t in aa]));ids=torch.tensor([t['a'] for t in aa])
    loss=loss-.2*ap[torch.arange(64),ids].clamp_min(1e-9).log().mean()
   if name.startswith('consistent'):
    x=m.x([pair['s']]);z=m.actor(x)
    extra=abs(delta)*torch.nn.functional.softplus(-np.sign(delta)*(z[0,pair['a']]-z[0,0]))
    if suffix and not (a.joint or name=='consistent_joint'):
     sx=m.x([t['s'] for t in suffix]);old=snap.probs(snap.x([t['s'] for t in suffix])).detach();new=m.probs(sx)
     extra=extra+(old*(old.clamp_min(1e-9).log()-new.clamp_min(1e-9).log())).sum(1).mean()
    if (a.joint or name=='consistent_joint') and delta:
     # Joint suffix likelihood ratios: credit remains tied to both measured
     # executable continuations, not solely the first intervention log-odds.
     ratios=[]
     for trajectory in [vt,bt]:
      tail=[t for t in trajectory if t['stage']==1 and t['step']>step]
      if tail:
       tx=m.x([t['s'] for t in tail]);ids=torch.tensor([t['a'] for t in tail]);j=torch.arange(len(tail))
       current=m.actor(tx).log_softmax(-1)[j,ids]
       with torch.no_grad():old=snap.actor(snap.x([t['s'] for t in tail])).log_softmax(-1)[j,ids]
       ratios.append((current-old).sum())
      else:ratios.append(z.sum()*0)
     odds=z[0,pair['a']]-z[0,0]+ratios[0]-ratios[1]
     extra=abs(delta)*torch.nn.functional.softplus(-np.sign(delta)*odds)
    if name=='consistent_smdp':
     # Both targets end at their ACTUAL Option termination. Never substitute
     # a frozen-policy terminal outcome for the changing actor's tail value.
     targets=[]
     with torch.no_grad():
      for trajectory in [vt,bt]:
       t=next(t for t in trajectory if t['stage']==1)
       nx=m.x([t['next']]);np_=m.probs(nx);nq=torch.stack([net(nx)for net in m.target]).mean(0)
       value=(np_*(nq-m.alpha*np_.clamp_min(1e-9).log())).sum()
       targets.append(t['r']+(0 if t['done'] else value))
      advantage=torch.as_tensor(targets[0]-targets[1]).clamp(-1,1)
     credit_trace.append(dict(native_delta=delta,smdp_advantage=float(advantage),option_targets=[float(x)for x in targets]))
     extra=advantage.abs()*torch.nn.functional.softplus(-advantage.sign()*(z[0,pair['a']]-z[0,0]))
    loss=(loss+extra)/2
   m.aopt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.actor.parameters(),10);m.aopt.step()
  log['credit_trace']=credit_trace
  log['extra_actor_updates']=40;log['paired_delta']=delta;log['suffix_boundaries']=len(suffix)
  history.append(dict(round=i,method=name,success_pair=[b['success'],v['success']],log=log))
  m.save(a.out/f'{name}_CHECKPOINT_{i+1}.pt')
 for key,obj in [('TRANSITIONS',data),('LABELS',labels),('PAIRS',pairs),('HISTORY',history)]: (a.out/(key+'.json')).write_text(json.dumps(obj))
 print('ROUND',i,delta,'shared transitions',len(data),flush=True)
 if i+1 in [4,a.rounds]:evaluate(i+1)
for name,m in models.items():m.save(a.out/f'{name}_FINAL.pt')
rt.close()
