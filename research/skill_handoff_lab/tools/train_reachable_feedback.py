"""Local RL at actually reached handoffs; benchmarks always full native starts.
State initialization is explicit TRAINING ONLY. Prefix is physically executed
once, charged once, archived once; every suffix step/query is separately charged.
"""
import argparse,copy,gzip,hashlib,json,time,shutil
from pathlib import Path
import numpy as np,torch,sapien
from program_peg_feedback import FeedbackRuntime,features,potential
from program_peg_native import Horizon,val
from program_budget_feedback_sac import FeedbackSAC,completion_labels
class Captured(Exception):pass
class LocalRuntime(FeedbackRuntime):
 def __init__(self,out):
  self.capture_requested=False;super().__init__(out)
 def execute_feedback(self,planner,pose,*args,**kw):
  if self.capture_requested:
   self.capture_requested=False;e=self.env;u=e.unwrapped
   self.cap=dict(state=copy.deepcopy(u.get_state_dict()),elapsed=u._elapsed_steps.clone(),records=copy.deepcopy(e.records),calls=copy.deepcopy(e.calls),actions=copy.deepcopy(self.actions),queries=copy.deepcopy(self.queries),pose=copy.deepcopy(pose),insert_pose=copy.deepcopy(planner.original_insert_pose),planner=planner)
   self.cap['wrapper_elapsed']=[];w=e
   while hasattr(w,'env'):
    if '_elapsed_steps'in vars(w):self.cap['wrapper_elapsed'].append((w,copy.deepcopy(vars(w)['_elapsed_steps'])))
    w=w.env
   raise Captured()
  return super().execute_feedback(planner,pose,*args,**kw)
 def capture(self,seed,policy,directory):
  self.callback=policy;self.actions=[];self.queries=[];self.capture_requested=True;self.cap=None;error=None
  try:self.solve(self.env,seed=seed,vis=False)
  except Captured:pass
  except Horizon:error='HORIZON'
  except Exception as ex:error=type(ex).__name__+':'+str(ex)
  self.capture_requested=False;e=self.env
  if self.cap is None:
   q=np.array(val(e.unwrapped.agent.robot.get_qpos()[0,:7]));e.phase='CAPTURE_FAILED_HOLD'
   while len(e.records)<100:e.step(np.r_[q,-1])
  initial=copy.deepcopy(e.initial);prefix=copy.deepcopy(e.records);source=directory/'PREFIX.json.gz';scope='ACTUAL_REACHABLE_PREFIX'if self.cap else'FULL_NATIVE_CAPTURE_FAILURE'
  with gzip.open(source,'wt')as f:json.dump(dict(scope=scope,seed=seed,initial=initial,steps=prefix,error=error,decisions=self.actions,queries=self.queries),f)
  if self.cap is not None:
   self.cap.update(initial=initial,seed=seed,source=str(source));torch.save(self.cap['state'],directory/'REACHABLE_STATE.pt')
  return len(prefix),len(self.queries),scope
 def local(self,policy,path,restore=True,label_policy="exploratory_behavior"):
  c=self.cap;e=self.env;u=e.unwrapped;start=len(c['records']);planner=c['planner']
  if restore:
   u.set_state_dict(copy.deepcopy(c['state']))
   if 'controller'in c['state']:u.agent.controller.set_state(copy.deepcopy(c['state']['controller']))
   u._elapsed_steps=c['elapsed'].clone()
   for w,n in c['wrapper_elapsed']:w._elapsed_steps=copy.deepcopy(n)
  e.records=copy.deepcopy(c['records']);e.calls=copy.deepcopy(c['calls']);e.initial=copy.deepcopy(c['initial']);self.actions=[];self.queries=[];self.callback=policy;planner.elapsed_steps=start;planner.gripper_state=-1;error=None
  pose=copy.deepcopy(c['pose'])
  # A state restore does not restore PhysX contact-force caches. Refresh them
  # with ONE charged physical hold step before observing grasp/contact state.
  # This is local-training initialization only, never benchmark execution.
  if restore:
   e.phase='LOCAL_INITIAL_SETTLE';e.step(np.r_[val(u.agent.robot.get_qpos()[0,:7]),-1])
  try:
   for i in range(3):
    if len(e.records)>=100:break
    if i:
     offset=sapien.Pose([-float(u.peg_half_sizes[0,0])-.01,0,0]);pose=u.goal_pose*offset*u.peg.pose.inv()*pose
    ret=super().execute_feedback(planner,pose)
    if isinstance(ret,int)and ret==-1:break
   else:
    if len(e.records)<100:ret=super().execute_feedback(planner,c['insert_pose']*sapien.Pose([.05,0,0]))
  except Horizon:error='HORIZON'
  except Exception as ex:error=type(ex).__name__+':'+str(ex)
  q=np.array(val(u.agent.robot.get_qpos()[0,:7]));e.phase='LOCAL_TERMINAL_HOLD'
  while len(e.records)<100:e.step(np.r_[q,-1])
  y=bool(u.evaluate()['success'][0]);ts=[]
  for i,z in enumerate(self.actions):
   end=self.actions[i+1]['step']if i+1<len(self.actions)else 100;done=i+1==len(self.actions);nxt=self.actions[i+1]['obs']if not done else features(e.records[-1],0,1).tolist();m=e.records[z['step']-1];n=e.records[end-1]
   ts.append(dict(s=z['obs'],next=nxt,a=z['id'],r=(float(y)if done else 0)+.2*((0 if done else potential(n))-potential(m)),done=done,duration=end-z['step'],stage=1,outcome=float(y),scene=c['seed'],source=str(path),controller=self.controller_id,step=z['step'],scope='LOCAL_REACHABLE_HANDOFF'))
  assert sum(t['duration']for t in ts)==100-start-int(restore)
  # Training labels use the actual suffix; prefix is a reference, NOT copied
  # into archived new physical steps or charged for every local rollout.
  full=dict(summary=dict(success=y),steps=e.records);labels=completion_labels(full,ts)
  row=dict(label_policy=label_policy,scope='LOCAL_REACHABLE_HANDOFF',seed=c['seed'],success=y,actual_steps=100-start,start_step=start,native_deadline=100,state_reset=restore,initialization_steps=int(restore),error=error,prefix_source=c['source'],decisions=self.actions,queries=self.queries)
  with gzip.open(path,'wt')as f:json.dump(dict(summary=row,steps=e.records[start:]),f)
  front=None
  if not restore:
   z=c['actions'][0];b=self.actions[0];front=dict(s=z['obs'],next=b['obs'],a=z['id'],r=.2*(potential(c['records'][-1])-potential(c['initial'])),done=False,duration=start,stage=0,outcome=float(y),scene=c['seed'],source=c['source'],controller=self.controller_id,step=0,scope='ONCE_EXECUTED_PREFIX')
  return row,ts,labels,front

p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--budget',type=int,default=10000);p.add_argument('--start',type=int,default=755000);p.add_argument('--dev-count',type=int,default=16);p.add_argument('--fidelity-only',action='store_true');p.add_argument('--head-label-policy',choices=['behavior','deterministic'],default='behavior');p.add_argument('--rollouts',type=int,default=8);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);m=FeedbackSAC.load(a.parent/'FINAL.pt');data=json.loads((a.parent/'TRANSITIONS.json').read_text());oldlabels=json.loads((a.parent/'LABELS.json').read_text());rt=LocalRuntime(a.out/'full_evaluation');cost=0;queries=0;rows=[];captures=[];recent=[];updates_start=m.updates;rng=np.random.default_rng(m.seed+900);m.save(a.out/'INITIAL.pt');shutil.copy2(__file__,a.out/'SOURCE.py')
(a.out/'PROTOCOL.json').write_text(json.dumps(dict(arguments={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},parent_sha256=hashlib.sha256((a.parent/'FINAL.pt').read_bytes()).hexdigest(),controller=rt.controller_id,training_only_state_initialization=True,restored_contact_refresh_steps=1,full_evaluation_resets_at_handoff=False,restore_exactness='NOT_CLAIMED: serialized state omits PhysX contact caches; restored rollouts are local training, not exact counterfactual pairs'),indent=2))
def exploratory(s):
 if a.fidelity_only:return m.choose(s)
 if rng.random()<.1:return int(rng.integers(3 if s[44]<.5 else 8))
 return m.choose(s,stochastic=True)
def evaluate(tag):
 result=[]
 for seed in range(753200,753200+a.dev_count):result.append(rt.run(seed,lambda s:m.choose(s),f'local_eval_{tag}')[0])
 (a.out/f'EVAL_{tag}.json').write_text(json.dumps(result,indent=2));print('LOCAL_EVAL',tag,sum(x['success']for x in result),flush=True)
for scene in range(a.start,a.start+1000):
 if a.budget-cost<100:break
 d=a.out/f'capture_{scene}';d.mkdir();n,q,scope=rt.capture(scene,exploratory,d);cost+=n;queries+=q;captures.append(dict(seed=scene,actual_steps=n,planner_queries=q,scope=scope));
 if rt.cap is None:
  e=rt.env;z=rt.actions[0];y=float(e.unwrapped.evaluate()['success'][0]);data.append(dict(s=z['obs'],next=features(e.records[-1],0,1).tolist(),a=z['id'],r=y-.2*potential(e.initial),done=True,duration=100,stage=0,outcome=y,scene=scene,source=str(d/'PREFIX.json.gz'),controller=rt.controller_id,step=0,scope='FULL_NATIVE_CAPTURE_FAILURE'))
  labels=[z for ll in recent[-48:]for z in ll]or oldlabels;m.train(data,labels,0 if a.fidelity_only else 40);continue
 for j in range(a.rollouts):
  if a.budget-cost<100-n:break
  deterministic_head=(a.head_label_policy=='deterministic'and j%4==0)
  row,t,l,front=rt.local((lambda s:m.choose(s))if deterministic_head else exploratory,d/f'suffix_{j}.json.gz',restore=j>0,label_policy='deterministic_deployment'if deterministic_head else 'exploratory_behavior')
  row['label_policy']='deterministic_deployment'if deterministic_head else 'exploratory_behavior';rows.append(row);cost+=row['actual_steps'];queries+=len(row['queries']);data+=t
  if a.head_label_policy=='behavior'or deterministic_head:recent.append(l)
  if front:data.append(front)
  labels=[z for ll in recent[-48:]for z in ll]or oldlabels;m.train(data,labels,0 if a.fidelity_only else 40)
  print('LOCAL_TRAIN',m.mode,scene,j,row['success'],cost,m.updates,row['error'],flush=True)
 rt.cap['planner'].close();rt.cap=None
 if len(captures)%8==0:
  m.save(a.out/f'CHECKPOINT_{cost}.pt')
  (a.out/'TRANSITIONS.json').write_text(json.dumps(data));(a.out/'LABELS.json').write_text(json.dumps(labels));(a.out/'LOCAL_ROWS.json').write_text(json.dumps(rows))
  evaluate(cost)
 (a.out/'COST.json').write_text(json.dumps(dict(actual_training_steps=cost,planner_queries=queries,local_rollouts=len(rows),captures=captures,optimizer_updates=m.updates-updates_start),indent=2))
(a.out/'COST.json').write_text(json.dumps(dict(actual_training_steps=cost,planner_queries=queries,local_rollouts=len(rows),captures=captures,optimizer_updates=m.updates-updates_start),indent=2))
(a.out/'LABELS.json').write_text(json.dumps([z for ll in recent[-48:]for z in ll]or oldlabels));m.save(a.out/'FINAL.pt')
if (a.out/f'EVAL_{cost}.json').exists():
 shutil.copy2(a.out/f'EVAL_{cost}.json',a.out/'EVAL_FINAL.json');print('LOCAL_EVAL FINAL reused identical final checkpoint',flush=True)
else:evaluate('FINAL')
rt.close();(a.out/'LOCAL_ROWS.json').write_text(json.dumps(rows,indent=2));(a.out/'TRANSITIONS.json').write_text(json.dumps(data));(a.out/'TRAINING.json').write_text(json.dumps(m.history,indent=2));print('LOCAL_DONE',flush=True)
