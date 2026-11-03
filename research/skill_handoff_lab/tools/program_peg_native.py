"""Instrument pinned official PegInsertionSide and explicit shared-control ablations."""
import argparse,gzip,hashlib,json,time,inspect,shutil
from pathlib import Path
import numpy as np, gymnasium as gym,torch
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.solutions import peg_insertion_side as official
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

def val(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().numpy().tolist()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:val(v)for k,v in x.items()}
 if isinstance(x,(np.floating,np.integer)):return x.item()
 return x
class Horizon(Exception):pass
class Measure(gym.Wrapper):
 def reset(self,**kw):
  out=self.env.reset(**kw);self.records=[];self.calls=[];self.policy_decisions=[];self.phase='RESET';self.initial=self.measure();return out
 def measure(self):
  u=self.unwrapped
  return dict(q=val(u.agent.robot.get_qpos()[0]),qvel=val(u.agent.robot.get_qvel()[0]),tcp=val(u.agent.tcp.pose.raw_pose[0]),peg=val(u.peg.pose.raw_pose[0]),goal=val(u.goal_pose.raw_pose[0]),half_sizes=val(u.peg_half_sizes[0]),grasped=bool(u.agent.is_grasping(u.peg)[0]),evaluation=val(u.evaluate()))
 def step(self,a):
  if len(self.records)>=100:raise Horizon()
  out=self.env.step(a);self.records.append(dict(step=len(self.records)+1,phase=self.phase,action=val(a),**self.measure()));return out
class Planner(PandaArmMotionPlanningSolver):
 speed=None
 refresh_terminal=False
 model=None
 def __init__(self,*args,**kw):
  if self.speed is not None:kw.update(joint_vel_limits=self.speed,joint_acc_limits=self.speed)
  super().__init__(*args,**kw)
 def move_to_pose_with_screw(self,pose,*args,**kw):
  refresh=self.refresh_terminal
  if len(self.env.calls)==6 and self.model is not None:
   from program_paired_skill_value import peg_features
   obs=peg_features(self.env.measure(),100-len(self.env.records),1);action=self.model.predict(obs,deterministic=True)[0];refresh=bool(action[0]>=0);self.env.policy_decisions.append(dict(stage=1,step=len(self.env.records),obs=obs.tolist(),requested=action.tolist(),applied_refresh=refresh))
  if len(self.env.calls)==6 and refresh:
   import sapien
   u=self.env.unwrapped;pose=u.goal_pose.sp*u.peg.pose.sp.inv()*u.agent.tcp.pose.sp*sapien.Pose([0.05,0,0])
  phases=['REACH','GRASP','ALIGN','REFINE1','REFINE2','REFINE3','INSERT'];i=len(self.env.calls);self.env.phase=phases[min(i,6)];t=time.monotonic();c=dict(phase=self.env.phase,start=len(self.env.records),target=val(pose.raw_pose)if hasattr(pose,'raw_pose')else val(np.r_[pose.p,pose.q]));self.env.calls.append(c)
  try:
   out=super().move_to_pose_with_screw(pose,*args,**kw);c['failed']=isinstance(out,int)and out==-1;return out
  finally:c.update(end=len(self.env.records),wall_s=time.monotonic()-t)
 def close_gripper(self,*args,**kw):
  self.env.phase='CLOSE';return super().close_gripper(*args,**kw)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path);p.add_argument('--grasp-offset',type=float,default=0.);p.add_argument('--refresh-terminal',action='store_true');p.add_argument('--gpu-obb',action='store_true');p.add_argument('--measured-insert',action='store_true');p.add_argument('--speed',type=float,default=None);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);Planner.speed=a.speed;Planner.refresh_terminal=a.refresh_terminal
 source=Path(inspect.getfile(official));
 if a.measured_insert or a.grasp_offset or a.model:
  code=source.read_text().replace('-max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01)',f'(-max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01)+{a.grasp_offset!r})')
  if a.measured_insert:code=code.replace('for i in range(3):','for i in range(0):').replace('insert_pose * (sapien.Pose([0.05, 0, 0]))','env.goal_pose.sp * env.peg.pose.sp.inv() * env.agent.tcp.pose.sp * sapien.Pose([0.05, 0, 0])')
  if a.model:code=code.replace(f'(-max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01)+{a.grasp_offset!r})','(-max(0.05, env.peg_half_sizes[0, 0].item() / 2 + 0.01)+choose_grasp(env))')
  exec(compile(code,str(source),'exec'),official.__dict__);(a.out/'SHARED_SOLVER.py').write_text(code)
 shutil.copy2(source,a.out/'OFFICIAL_SOLVER.py');shutil.copy2(__file__,a.out/'EXECUTION_SOURCE.py')
 (a.out/'MANIFEST.json').write_text(json.dumps(dict(task='PegInsertionSide-v1',horizon=100,solver_sha=hashlib.sha256(source.read_bytes()).hexdigest(),seeds=a.seeds,controller='UNMODIFIED_OFFICIAL'if a.speed is None else 'OFFICIAL_GEOMETRY_SHARED_RETIMING',speed=a.speed,measured_insert=a.measured_insert,gpu_obb=a.gpu_obb,grasp_offset=a.grasp_offset,refresh_terminal=a.refresh_terminal,notes='Stop at native horizon; terminal hold after solver returns. Explicit flagged shared control/parameter changes; no physical state setters.'),indent=2))
 if a.gpu_obb:
  original_obb=official.get_actor_obb
  def gpu_obb(actor,*args,**kw):
   import trimesh
   from scipy.spatial.transform import Rotation
   box=original_obb(actor,to_world_frame=False);pose=np.array(val(actor.pose.raw_pose[0]),dtype=float);T=np.eye(4);T[:3,:3]=Rotation.from_quat(np.r_[pose[4:],pose[3]]).as_matrix();T[:3,3]=pose[:3]
   return trimesh.primitives.Box(extents=np.array(box.primitive.extents).copy(),transform=T@np.array(box.primitive.transform))
  official.get_actor_obb=gpu_obb
 if a.model:
  import program_contact_sac as sac
  sac.mask=lambda obs:torch.stack([torch.ones_like(obs[:,44]),torch.zeros_like(obs[:,44])],dim=1)
  Planner.model=sac.BoundarySAC.load(a.model,device='cpu')
 env=Measure(gym.make('PegInsertionSide-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));official.PandaArmMotionPlanningSolver=Planner;rows=[]
 if a.model:
  def choose_grasp(u):
   from program_paired_skill_value import peg_features
   obs=peg_features(env.initial,100,0);action=Planner.model.predict(obs,deterministic=True)[0];offset=float(np.clip(np.rint(action[0]),-1,1)*.01);env.policy_decisions.append(dict(stage=0,step=0,obs=obs.tolist(),requested=action.tolist(),applied_grasp_offset=offset));return offset
  official.choose_grasp=choose_grasp
 for seed in a.seeds:
  t=time.monotonic();error=None
  try:
   out=official.solve(env,seed=seed,vis=False);error='PLAN_FAILED'if isinstance(out,int)and out==-1 else None
  except Horizon:error='HORIZON'
  except Exception as e:error=type(e).__name__+':'+str(e)
  env.phase='TERMINAL_HOLD';q=np.array(val(env.unwrapped.agent.robot.get_qpos()[0,:7]));grip=-1 if any(x['phase']=='CLOSE'for x in env.records)else 1
  while len(env.records)<100:env.step(np.r_[q,grip])
  row=dict(policy_decisions=env.policy_decisions,model_sha256=None if a.model is None else hashlib.sha256(a.model.read_bytes()).hexdigest(),seed=seed,success=bool(env.unwrapped.evaluate()['success'][0]),error=error,steps=len(env.records),wall_s=time.monotonic()-t,calls=env.calls)
  f=a.out/f'{seed}.json.gz'
  with gzip.open(f,'wt')as h:json.dump(dict(summary=row,initial=env.initial,steps=env.records),h)
  rows.append(dict(row,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()));(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(seed,row['success'],error,[(c['phase'],c['start'],c['end'])for c in env.calls],flush=True)
 env.close()
if __name__=='__main__':main()
