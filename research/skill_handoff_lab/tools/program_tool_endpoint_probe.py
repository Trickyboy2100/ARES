"""Single-factor screw integration resolution, plus read-only planned endpoint FK.
No changed physics, horizon, goal or native velocity/acceleration limits.
"""
import sys,json
from pathlib import Path
import numpy as np
import program_tool_all_contacts as contacts
run=contacts.run
Parent=run.EffectPlanner
class EndpointPlanner(Parent):
 def follow_path(self,result,refine_steps=0):
  record=self.env.plans[-1]
  if record.get('target') and len(result['position']):
   pin=self.planner.pinocchio_model;q=np.r_[result['position'][-1],run.serial(self.robot.get_qpos()[0])[-2:]]
   pin.compute_forward_kinematics(q);actual=np.asarray(pin.get_link_pose(self.planner.link_name_2_idx[self.planner.move_group]));goal=self.planner.transform_goal_to_wrt_base(np.asarray(record['target']))
   record['endpoint_fk']=dict(frame='robot_base',planned_pose=actual.tolist(),goal_pose=np.asarray(goal).tolist(),position_error_mm=float(np.linalg.norm(actual[:3]-np.asarray(goal)[:3])*1000))
  return super().follow_path(result,refine_steps)
 def move_to_pose_with_screw(self,pose,**kwargs):
  if self.env.phase!='TRANSFER' or not self.env.effect_mode.startswith('dense'):return super().move_to_pose_with_screw(pose,**kwargs)
  original=self.planner.plan_screw;mode=self.env.effect_mode
  def dense(*a,**k):k['qpos_step']=float('.'+mode.replace('dense',''));return original(*a,**k)
  self.planner.plan_screw=dense;self.env.effect_mode='baseline'
  try:return super().move_to_pose_with_screw(pose,**kwargs)
  finally:self.planner.plan_screw=original;self.env.effect_mode=mode
run.EffectPlanner=EndpointPlanner
if __name__=='__main__':
 try:run.main()
 finally:
  out=Path(sys.argv[sys.argv.index('--out')+1]);(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(out/'program_tool_all_contacts.py').write_bytes(Path(contacts.__file__).read_bytes())
