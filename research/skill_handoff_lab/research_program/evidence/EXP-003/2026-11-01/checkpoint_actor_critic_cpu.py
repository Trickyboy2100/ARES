"""NOT_PHYSICS_OUTPUT: frozen CPU Q/action diagnostics on earlier training states."""
import argparse,hashlib,json,gzip,itertools,sys
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sb3',type=Path,required=True);p.add_argument('--tools',type=Path,required=True);p.add_argument('--max-decisions',type=int,default=2048);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not a.checkpoint.exists():raise FileNotFoundError('Checkpoint not yet available; no diagnostic generated')
 sys.path.insert(0,str(a.sb3));sys.path.insert(0,str(a.tools));from stable_baselines3 import SAC
 model=SAC.load(a.checkpoint,device='cpu');model.policy.set_training_mode(False);assert model.observation_space.shape==(92,)
 ep=json.loads((a.run/'EPISODES.json').read_text());eligible=[];total=0
 for row in ep:
  total+=row['decisions']
  if total<=a.max_decisions and not row.get('truncated_training_tail'):eligible.append(row)
 selected=[]
 for tag,rows in [('first10',eligible[:10]),('recent10_before_checkpoint',eligible[-10:])]:
  for row in rows:selected.append((tag,row))
 data=[]
 for tag,row in selected:
  assert sha(row['path'])==row['sha256']
  with gzip.open(row['path'],'rt') as f:trace=json.load(f)
  ts=trace['transitions']
  if not ts:continue
  history=[np.array(ts[0]['obs'],dtype=np.float32).copy() for _ in range(4)]
  contact=next((i for i,t in enumerate(ts) if np.linalg.norm(np.array(t['obs'])[9:12])>.1),None)
  samples=sorted({0}|({contact} if contact is not None else set()))
  for i,t in enumerate(ts):
   if i>0:history=(history+[np.array(t['obs'],dtype=np.float32)])[-4:]
   if i not in samples:continue
   # ln_sac always disables derived features: zero slots, shared four raw frames.
   family=row['task_family'];obs=np.r_[np.concatenate(history),np.zeros(6),[family=='Y',family=='YZ']].astype(np.float32);assert len(obs)==92
   actor=np.asarray(model.predict(obs,deterministic=True)[0]);zero=np.zeros(3);zeroz=actor.copy();zeroz[1]=0
   grid=[np.array([y,z,0.],dtype=np.float32) for y,z in itertools.product([-.8,0,.8],repeat=2)]
   actions=np.array([zero,actor,zeroz]+grid,dtype=np.float32);ob=torch.tensor(np.repeat(obs[None],len(actions),axis=0))
   with torch.no_grad():q=torch.cat(model.critic(ob,torch.tensor(actions)),dim=1).numpy()
   assert np.isfinite(q).all();v=q.min(1);best=int(np.argmax(v[3:]))+3
   act=torch.tensor(actor[None],requires_grad=True);val=torch.cat(model.critic(torch.tensor(obs[None]),act),dim=1).min();val.backward()
   data.append(dict(window=tag,seed=row['seed'],family=family,bias_mm_analysis_only=row['bias_mm'],training_episode_final_success=row['success'],transition_index=i,physical_step=t['step'],sample_role='first_suffix' if i==0 else 'first_contact',observed_force_norm=float(np.linalg.norm(np.array(t['obs'])[9:12])*10),raw_obs_sha256=hashlib.sha256(obs.tobytes()).hexdigest(),actor_action=actor.tolist(),historical_behavior_action=t['action'],q_zero=float(v[0]),q_actor=float(v[1]),q_zero_z=float(v[2]),actor_over_zero=bool(v[1]>v[0]),actor_over_zero_z=bool(v[1]>v[2]),best_grid_action=actions[best].tolist(),best_grid_q=float(v[best]),actions=actions.tolist(),q1=q[:,0].tolist(),q2=q[:,1].tolist(),minQ_action_gradient=act.grad.numpy()[0].tolist(),trace_sha256=row['sha256']))
 summary=[]
 for tag in sorted({x['window'] for x in data}):
  rows=[x for x in data if x['window']==tag and x['sample_role']=='first_suffix']
  summary.append(dict(window=tag,episodes=len(rows),historical_training_success=sum(x['training_episode_final_success'] for x in rows),mean_current_actor=np.mean([x['actor_action'] for x in rows],axis=0).tolist(),actor_over_zero=sum(x['actor_over_zero'] for x in rows),actor_over_zero_z=sum(x['actor_over_zero_z'] for x in rows)))
 out=dict(classification='NOT_PHYSICS_OUTPUT_CPU_FROZEN_CHECKPOINT_DIAGNOSTIC',physics_steps=0,optimizer_updates=0,checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),checkpoint_training_decisions=a.max_decisions,eligible_completed_episodes=len(eligible),observation='four real21dim frames + six zeros + two known family bits; no bias labels or future success fed to policy',summary=summary,samples=data,script_sha256=sha(__file__),limitations=['First/recent windows are training data, not held-out evaluation.','Historical success belongs to historical exploratory behavior, not this checkpoint prediction.','Q grid is model inference, not actual executed alternatives; soft Q is not success probability.','Future final outcome is attached for analysis only and is absent from reconstructed policy input.'])
 a.out.write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
