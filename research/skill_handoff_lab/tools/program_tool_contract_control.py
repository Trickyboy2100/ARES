"""Shared successor endpoint/arrival correction; native100 goal and dynamics unchanged."""
import sys,time,json
from pathlib import Path
import numpy as np
import program_tool_all_contacts as contacts
run=contacts.run
Parent=run.EffectPlanner
class ContractPlanner(Parent):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  def wrapped(original,name):
   def call(*a,**k):
    started=time.monotonic()
    try:result=original(*a,**k)
    except RuntimeError as exc:
     if str(exc)!='Fail to parameterize path':raise
     result={'status':'TOPP_PARAMETERIZATION_FAILED'}
    self.env.plans.append(dict(phase='PLANNER_QUERY',caller_phase=self.env.phase,step=len(self.env.records),solver=name,status=result.get('status'),native_points=len(result.get('position',[])),query_wall_s=time.monotonic()-started))
    return result
   return call
  for name in ['plan_screw','plan_qpos_to_pose']:setattr(self.planner,name,wrapped(getattr(self.planner,name),name))
 def error(self,q,pose):
  pin=self.planner.pinocchio_model;pin.compute_forward_kinematics(np.r_[q,run.serial(self.robot.get_qpos()[0])[-2:]])
  actual=np.asarray(pin.get_link_pose(self.planner.link_name_2_idx[self.planner.move_group]));goal=np.asarray(self.planner.transform_goal_to_wrt_base(np.r_[pose.p,pose.q]))
  return dict(position_m=float(np.linalg.norm(actual[:3]-goal[:3])),angle_rad=float(2*np.arccos(np.clip(abs(actual[3:]@goal[3:]),0,1))),planned_pose=actual.tolist(),goal=goal.tolist())
 def move_to_pose_with_screw(self,pose,**kwargs):
  if self.env.phase not in getattr(self.env,'corrected_phases',['TRANSFER','HOOK','PULL']) or self.env.effect_mode=='baseline':return super().move_to_pose_with_screw(pose,**kwargs)
  if not self.env.effect_mode.startswith('corrected'):return super().move_to_pose_with_screw(pose,**kwargs)
  start=time.monotonic();q=np.array(run.serial(self.robot.get_qpos()[0]));attempts=[]
  result=self.planner.plan_screw(np.r_[pose.p,pose.q],q,time_step=self.base_env.control_timestep,qpos_step=.01,use_point_cloud=False,wrt_world=True)
  err=None
  if result['status']=='Success' and len(result['position']):err=self.error(result['position'][-1],pose)
  attempts.append(dict(solver='screw_001',status=result['status'],endpoint=err))
  if err is None or err['position_m']>.002 or err['angle_rad']>.02:
   result=self.planner.plan_qpos_to_pose(np.r_[pose.p,pose.q],q,time_step=self.base_env.control_timestep,planning_time=.5,use_point_cloud=False,wrt_world=True)
   err=self.error(result['position'][-1],pose) if result['status']=='Success' and len(result['position']) else None
   attempts.append(dict(solver='qpos_to_pose',status=result['status'],endpoint=err))
  record=dict(phase=self.env.phase,step=len(self.env.records),target=run.rawpose(pose),attempts=attempts,planning_wall_s=time.monotonic()-start,status=result['status'],position=None if err is None else result['position'].tolist(),endpoint_fk=err)
  self.env.plans.append(record)
  if err is None:return -1
  ret=self.follow_path(result,refine_steps=0);holds=[]
  for j in range(getattr(self.env,'arrival_hold_max',4)+1):
   actual=run.sp(self.env.unwrapped.agent.tcp);dp=float(np.linalg.norm(actual.p-pose.p));da=float(2*np.arccos(np.clip(abs(actual.q@pose.q),0,1)));holds.append(dict(step=len(self.env.records),position_m=dp,angle_rad=da))
   if dp<=.004 and da<=.04 or j==getattr(self.env,'arrival_hold_max',4):break
   ret=self.env.step(np.r_[result['position'][-1],self.gripper_state])
  record['arrival']=holds
  if getattr(self.env,'arrival_replan',False) and self.env.phase=='TRANSFER' and (dp>.004 or da>.04):
   # One physical correction from the measured robot state; full native timing.
   q=np.array(run.serial(self.robot.get_qpos()[0]));fix=self.planner.plan_qpos_to_pose(np.r_[pose.p,pose.q],q,time_step=self.base_env.control_timestep,planning_time=.5,use_point_cloud=False,wrt_world=True)
   detail=dict(status=fix['status'],start_step=len(self.env.records),position=fix.get('position',np.empty((0,7))).tolist());record['measured_arrival_replan']=detail
   if fix['status']=='Success':
    detail['endpoint_fk']=self.error(fix['position'][-1],pose);ret=self.follow_path(fix,refine_steps=0)
    actual=run.sp(self.env.unwrapped.agent.tcp);detail['actual_position_m']=float(np.linalg.norm(actual.p-pose.p));detail['end_step']=len(self.env.records)
  return ret
original_execute=run.task.execute_chain
def execute(env,*args,**kwargs):
 mode=env.effect_mode.removesuffix('_flip');env.grasp_flip=env.effect_mode.endswith('_flip')
 env.arrival_replan=mode=='corrected_arrival';env.recompute_pull=mode in ['corrected_feedback','corrected_recompute','corrected_wide','corrected_arrival'];env.pull_limit=.36 if mode in ['corrected_feedback','corrected_expand','corrected_wide','corrected_arrival'] else .29
 env.expand_geometric_yaw=mode=='corrected_wide'
 return original_execute(env,*args,**kwargs)
run.task.execute_chain=execute
run.EffectPlanner=ContractPlanner
if __name__=='__main__':
 try:run.main()
 finally:
  out=Path(sys.argv[sys.argv.index('--out')+1]);(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(out/'program_tool_all_contacts.py').write_bytes(Path(contacts.__file__).read_bytes())
