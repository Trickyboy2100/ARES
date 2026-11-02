"""Actual measured-attachment feedback vs shared native-timed tool control.
No new task criterion; bounded replanning consumes the SAME native100 budget.
"""
import argparse,gzip,json,hashlib,time,random
from pathlib import Path
import program_stap_tool as task
from program_stap_tool import *
OriginalPlanner=task.TracePlanner
class EffectPlanner(OriginalPlanner):
 def follow_path(self,result,refine_steps=0):
  chunk=getattr(self,'active_chunk',0)
  self.cut=False
  if chunk and len(result['position'])>chunk:
   result={**result,'position':result['position'][:chunk],'velocity':result['velocity'][:chunk]};self.cut=True
  return super().follow_path(result,refine_steps)
 def move_to_pose_with_screw(self,pose,**kwargs):
  mode=self.env.effect_mode
  if self.env.phase!='TRANSFER' or mode=='baseline':return super().move_to_pose_with_screw(pose,**kwargs)
  if mode.startswith('settle'):
   result=super().move_to_pose_with_screw(pose,**kwargs)
   if isinstance(result,int) and result==-1:return result
   count=int(mode.replace('settle',''));q=np.array(self.env.plans[-1]['position'][-1]);self.env.phase='TRANSFER_SETTLE'
   self.env.plans.append(dict(phase='ARRIVAL_SETTLE',step=len(self.env.records),hold_steps=count,target_q=q.tolist()))
   try:
    for _ in range(count):result=self.env.step(np.r_[q,self.gripper_state])
   finally:self.env.phase='TRANSFER'
   return result
  if mode.startswith('slow'):
   previous=self.planner.joint_acc_limits.copy();self.planner.joint_acc_limits=np.ones_like(previous)*float(mode.replace('slow',''))
   try:return super().move_to_pose_with_screw(pose,**kwargs)
   finally:self.planner.joint_acc_limits=previous
  attached=sp(self.env.unwrapped.agent.tcp).inv()*sp(self.env.unwrapped.l_shape_tool);initial_attachment=attached;goal=pose*attached
  chunk=int(mode.replace('refresh','').replace('frozen',''));self.active_chunk=chunk
  try:
   for repeat in range(12):
    attached=sp(self.env.unwrapped.agent.tcp).inv()*sp(self.env.unwrapped.l_shape_tool)
    target=goal*(initial_attachment if mode.startswith('frozen') else attached).inv();self.env.plans.append(dict(phase='ATTACHMENT_REFRESH',step=len(self.env.records),repeat=repeat,chunk=chunk,tool_goal=rawpose(goal),actual_attachment=rawpose(attached)))
    result=super().move_to_pose_with_screw(target,**kwargs)
    if isinstance(result,int) and result==-1:return -1
    if not self.cut:return result
   raise Failure('BOUNDED_TRANSFER_REFRESH_EXHAUSTED')
  finally:self.active_chunk=0

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--lookahead',action='store_true');p.add_argument('--seeds',type=int,nargs='+',default=[745000,745001]);p.add_argument('--masses',type=float,nargs='+',default=[.75,1.25]);p.add_argument('--grasps',type=float,nargs='+',default=[.035,.07,.105]);p.add_argument('--modes',nargs='+',default=['baseline','refresh12']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 if a.lookahead:a.grasps=[None]
 task.TracePlanner=EffectPlanner;rows=[]
 for name in [Path(__file__).name,'program_stap_tool.py','program_multiskill_pull.py','program_multiskill_contact_probe.py','pull_tool_physics.py']:(a.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task='ARES-ToolLoad-v0',native_success='cube/base xy distance below0.6m',horizon=100,seeds=a.seeds,masses=a.masses,grasps=a.grasps,modes=a.modes,split='NEW_TRAIN_DESIGN',stability_max=10,acceleration=6.,native_path_stride=1,hardware_commands=False,optimizer_updates=0,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.py')}),indent=2))
 for mass in a.masses:
  env=ContactRecorder(gym.make('ARES-ToolLoad-v0',density_scale=mass,num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));env.allow_unstable=False;env.stability_max=10;env.acceleration=6.;env.target_y=None;env.target_feedback=False
  try:
   for seed in a.seeds:
    for gx in a.grasps:
     for mode in a.modes:
      random.seed(31131);np.random.seed(31131);torch.manual_seed(31131);env.effect_mode=mode;env.stability=None;error=None;start=time.monotonic()
      try:task.execute_chain(env,seed,grasp_x=gx,hook_y=None,transfer_stride=1,pull_policy='goal')
      except Exception as e:error=repr(e)
      q=np.array(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));env.phase='TERMINAL_HOLD'
      while len(env.records)<100:env.step(np.r_[q,-1.])
      r=dict(seed=seed,mass=mass,grasp=gx,mode=mode,success=env.records[-1]['native_success'],error=error,steps=100,stability=env.stability,initial_sha256=env.initial_sha,wall_s=time.monotonic()-start,decisions=len(env.decisions))
      path=a.out/f'{seed}_m{mass}_g{gx}_{mode}.json.gz'
      with gzip.open(path,'wt') as f:json.dump(dict(summary=r,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans,decisions=env.decisions,skill_results=env.skill_results),f)
      r.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());rows.append(r);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(seed,mass,gx,mode,r['success'],error,flush=True)
      if error and not error.startswith(('Horizon(', 'Failure(')):raise RuntimeError('Unexpected software error; recorded attempt is invalid: '+error)
  finally:env.close()
if __name__=='__main__':main()
