"""Earlier measured-contact interventions within the unchanged strong skill chain.
No suspension/state restore: callback runs at actual contact and original chain resumes.
Zero action continues every original native joint waypoint; all methods native100.
"""
import argparse,json,time,random,shutil
from pathlib import Path
import numpy as np
from transforms3d.quaternions import qmult
import program_post_contact as old
r=old.r
class FirstContactPlanner(old.control.ContractPlanner):
 def follow_path(self,result,refine_steps=0):
  e=self.env
  if e.phase not in getattr(e,'event_phases',['TRANSFER','HOOK','PULL'])or e.contact_event is not None:return super().follow_path(result,refine_steps)
  target=next(z['target']for z in reversed(e.plans)if z.get('target'));n=len(result['position'])
  for i in range(n+refine_steps):
   before=e.measure();ret=e.step(np.r_[result['position'][min(i,n-1)],self.gripper_state]);self.elapsed_steps+=1;s=e.records[-1];motion=float(np.linalg.norm(np.array(s['cube'][:2])-before['cube'][:2]));force=max([z['force_norms_N']['held_tool']for z in s['physics_substep_contacts']]+[s['tool_cube_force_N']])
   if e.contact_event is None and force>.5 and motion>.0001:
    e.contact_event=dict(step=len(e.records),phase=e.phase,force_N=force,cube_motion_m=motion,motion_definition='one actual control interval, not cumulative from PICK_TOOL',measurement=e.measure(),remaining=100-len(e.records));a=e.first_action;phase=e.phase;yaw=next(d['parameters']['tool_yaw_rad']for d in reversed(e.decisions)if d['skill']=='USE_TOOL');e.contact_decision=dict(stage=2,skill='FIRST_CONTACT_CORRECTION',boundary_step=len(e.records),parameters=dict(normalized=a.tolist(),lateral_m=.015*a[0],relative_yaw_rad=.15*a[1]),observation=e.measure(),previous_skill_result=e.contact_event,selection='predeclared first moving contact intervention')
    if np.max(np.abs(a))<=.1:continue
    current=r.sp(e.unwrapped.l_shape_tool);attached=r.sp(e.unwrapped.agent.tcp).inv()*current;delta=np.array([-np.sin(yaw)*.015*a[0],np.cos(yaw)*.015*a[0],0.]);goal=r.sapien.Pose(current.p+delta,qmult(r.axangle2quat([0,0,1],.15*a[1]),current.q))
    if self.move_to_pose_with_screw(goal*attached.inv())==-1:raise r.Failure('FIRST_CONTACT_LOCAL_PLAN')
    e.plans.append(dict(phase='FIRST_CONTACT_LOCAL_EFFECT',step=len(e.records),measurement=e.measure()))
    # Return to the previously validated TCP waypoint; downstream strong chain
    # refreshes actual attachment and actual cube pose as before. No stale-q jump.
    ret=self.move_to_pose_with_screw(r.sapien.Pose(target[:3],target[3:]))
    if isinstance(ret,int)and ret==-1:raise r.Failure('FIRST_CONTACT_REJOIN_PLAN')
    return ret
  return ret

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['first','pull'],default='first');p.add_argument('--seeds',type=int,nargs='+',default=[748051,748052,748058,748059]);p.add_argument('--grasps',type=float,nargs='+',default=[.04,.07]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[];e=old.make_env();e.event_phases=['PULL']if a.phase=='pull'else ['TRANSFER','HOOK','PULL'];r.task.TracePlanner=FirstContactPlanner
 for name in ['program_first_contact.py','program_post_contact.py','program_tool_contract_control.py','program_stap_tool.py']:shutil.copy2(Path(__file__).with_name(name),a.out/name)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task='PullCubeTool-v1',horizon=100,event_phases=e.event_phases,first_contact='first TRANSFER/HOOK/PULL step with max100Hz tool/cube force>0.5N and same20Hz interval actual cubeXY motion>0.1mm',zero='exact original full chain',nonzero='local measured-relative yaw then replan to original current waypoint, resume original strong feedback chain',seeds=a.seeds,grasps=a.grasps,scope='selected design-scene timing diagnostic; not independent test; no new RL claim'),indent=2))
 try:
  for seed in a.seeds:
   for grasp in a.grasps:
    for action in [[0,0],[0,-1],[0,1]]:
     e.first_action=np.array(action,float);e.contact_event=None;e.contact_decision=None;e.pending=None;e.stability=None;random.seed(31131);np.random.seed(31131);r.torch.manual_seed(31131);t=time.monotonic();error=None
     try:old.control.execute(e,seed,grasp_x=grasp,hook_y=None,transfer_stride=1,pull_policy='goal')
     except(r.Failure,r.Horizon)as exc:error=repr(exc)
     old.finish(e)
     if e.contact_decision is not None:e.decisions.append(e.contact_decision);e.decisions.sort(key=lambda d:d['boundary_step'])
     row=old.save(e,a.out,seed,grasp,action,error,t,len(rows));rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(seed,grasp,action,row['contact_step'],row['success'],row['final_distance'],error,flush=True)
 finally:e.close()
if __name__=='__main__':main()
