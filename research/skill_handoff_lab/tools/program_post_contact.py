"""Native100 PickTool -> PULL contact-boundary parameter intervention. Continuous physics.
Zero action exactly continues the strong native-timed trajectory. No state restore.
"""
import argparse,gzip,json,hashlib,random,time,shutil
from pathlib import Path
import numpy as np
from transforms3d.quaternions import qmult
import program_tool_contract_control as control
r=control.run
class ContactBoundary(Exception):pass
class ContactPlanner(control.ContractPlanner):
 def follow_path(self,result,refine_steps=0):
  if self.env.phase!='PULL' or not getattr(self.env,'pause_contact',False):return super().follow_path(result,refine_steps)
  e=self.env;n=len(result['position']);target=next(z['target']for z in reversed(e.plans)if z.get('target'))
  attachment=r.sp(e.unwrapped.agent.tcp).inv()*r.sp(e.unwrapped.l_shape_tool);goal=r.sapien.Pose(target[:3],target[3:])*attachment
  start=np.array(e.decisions[-1]['observation']['cube'][:2])
  for i in range(n+refine_steps):
   ret=e.step(np.r_[result['position'][min(i,n-1)],self.gripper_state]);self.elapsed_steps+=1;s=e.records[-1]
   force=max([z['force_norms_N']['held_tool']for z in s['physics_substep_contacts']]+[s['tool_cube_force_N']]);motion=float(np.linalg.norm(np.array(s['cube'][:2])-start))
   if force>.5 and motion>.001:
    e.pending=dict(position=result['position'][i+1:].copy(),target_tcp=target,goal_tool=r.rawpose(goal),gripper=self.gripper_state);e.contact_event=dict(step=len(e.records),force_N=force,cube_motion_m=motion,measurement=e.measure(),remaining=100-len(e.records));e.pause_contact=False;raise ContactBoundary()
  return ret

def make_env():
 e=r.ContactRecorder(r.gym.make('PullCubeTool-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));e.allow_unstable=False;e.stability_max=10;e.acceleration=6.;e.target_y=None;e.target_feedback=False;e.hook_height=.026;e.effect_mode='corrected_feedback';r.task.TracePlanner=ContactPlanner;return e

def prefix(e,seed,grasp):
 e.pause_contact=True;e.pending=None;e.contact_event=None;e.stability=None;random.seed(31131);np.random.seed(31131);r.torch.manual_seed(31131)
 try:control.execute(e,seed,grasp_x=grasp,hook_y=None,transfer_stride=1,pull_policy='goal')
 except ContactBoundary:return None
 except (r.Failure,r.Horizon)as ex:return repr(ex)
 return 'NO_EFFECTIVE_CONTACT_BOUNDARY'

def suffix(e,action):
 action=np.clip(np.asarray(action,dtype=float),-1,1);selected_action=action.copy();action=np.zeros(2)if np.max(np.abs(action))<=.1 else action;e.pause_contact=False;p=ContactPlanner(e,vis=False,base_pose=e.unwrapped.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=1.8,joint_acc_limits=6.);p.manual=True;p.stride=1;p.gripper_state=-1.
 yaw=e.decisions[-1]['parameters']['tool_yaw_rad'];local=getattr(e,'correction','endpoint_yz')in ['local_yaw','relative_yaw','axial_pitch'];delta=(np.array([np.cos(yaw)*.025*action[0],np.sin(yaw)*.025*action[0],0.])if getattr(e,'correction','')=='axial_pitch'else np.array([-np.sin(yaw)*.015*action[0],np.cos(yaw)*.015*action[0],0. if local else .004*action[1]]));e.decisions.append(dict(stage=2,skill='CONTACT_CORRECTION',boundary_step=len(e.records),parameters=dict(normalized=action.tolist(),selected_normalized=selected_action.tolist(),deadzone=.1,world_offset_m=delta.tolist(),local_yaw_rad=.15*action[1]if local and getattr(e,'correction','')!='axial_pitch'else 0.,local_pitch_rad=.2*action[1]if getattr(e,'correction','')=='axial_pitch'else 0.,correction=getattr(e,'correction','endpoint_yz')),observation=e.measure(),previous_skill_result=e.contact_event,selection='post_effective_contact_same_authority'))
 e.phase='PULL'
 try:
  if np.max(np.abs(action))<=.1:
   for q in e.pending['position']:e.step(np.r_[q,-1.])
   pose=r.sapien.Pose(e.pending['target_tcp'][:3],e.pending['target_tcp'][3:]);last=e.pending['position'][-1]if len(e.pending['position'])else np.asarray(e.records[-1]['action'][:7])
   for j in range(5):
    actual=r.sp(e.unwrapped.agent.tcp);dp=np.linalg.norm(actual.p-pose.p);da=2*np.arccos(np.clip(abs(actual.q@pose.q),0,1))
    if dp<=.004 and da<=.04 or j==4:break
    e.step(np.r_[last,-1.])
  else:
   if local:
    current=r.sp(e.unwrapped.l_shape_tool);rot=qmult(r.axangle2quat([0,0,1],.15*action[1]),current.q)if getattr(e,'correction','endpoint_yz')=='relative_yaw'else r.axangle2quat([0,0,1],yaw+.15*action[1]);rot=qmult(current.q,r.axangle2quat([0,1,0],.2*action[1]))if getattr(e,'correction','')=='axial_pitch'else rot;goal=r.sapien.Pose(current.p+delta,rot);attached=r.sp(e.unwrapped.agent.tcp).inv()*current
    if p.move_to_pose_with_screw(goal*attached.inv())==-1:raise r.Failure('LOCAL_CONTACT_PLAN')
    e.plans.append(dict(phase='LOCAL_EFFECT',step=len(e.records),measurement=e.measure()))
   goal=r.sapien.Pose(np.array(e.pending['goal_tool'][:3])+delta,rot if local else e.pending['goal_tool'][3:]);attached=r.sp(e.unwrapped.agent.tcp).inv()*r.sp(e.unwrapped.l_shape_tool)
   if p.move_to_pose_with_screw(goal*attached.inv())==-1:raise r.Failure('CONTACT_CORRECTION_PLAN')
 except (r.Failure,r.Horizon)as ex:return repr(ex)
 finally:p.close()
 return None

def finish(e):
 e.phase='TERMINAL_HOLD';q=np.array(r.serial(e.unwrapped.agent.robot.get_qpos()[0,:7]))
 while len(e.records)<100:e.step(np.r_[q,-1.])

def save(e,out,seed,grasp,action,error,started,index):
 out.mkdir(parents=True,exist_ok=True);f=out/f'{index:05d}_{seed}.json.gz';s=dict(seed=seed,mass=1.,grasp=grasp,action=list(map(float,action)),success=e.records[-1]['native_success'],error=error,steps=len(e.records),initial_sha256=e.initial_sha,contact_step=None if e.contact_event is None else e.contact_event['step'],wall_s=time.monotonic()-started,planner_queries=sum(z['phase']=='PLANNER_QUERY'for z in e.plans),query_wall_s=sum(z.get('query_wall_s',0)for z in e.plans),final_distance=e.records[-1]['cube_base_distance_m'])
 with gzip.open(f,'wt')as h:json.dump(dict(summary=s,initial=e.initial,initial_measure=e.initial_measure,steps=e.records,plans=e.plans,decisions=e.decisions,skill_results=e.skill_results,contact_event=e.contact_event),h)
 return dict(s,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())

def main():
 p=argparse.ArgumentParser();p.add_argument('--correction',choices=['endpoint_yz','local_yaw','relative_yaw','axial_pitch'],default='endpoint_yz');p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=list(range(748000,748004)));p.add_argument('--grasps',type=float,nargs='+',default=[.07,.105]);p.add_argument('--actions',default='0,0;-1,0;1,0;0,-1;0,1');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);actions=[list(map(float,x.split(',')))for x in a.actions.split(';')];rows=[]
 for n in ['program_post_contact.py','program_tool_contract_control.py','program_stap_tool.py','program_tool_all_contacts.py','program_tool_effect_probe.py','program_multiskill_pull.py','program_multiskill_contact_probe.py','pull_tool_physics.py']:shutil.copy2(Path(__file__).with_name(n),a.out/n)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task='PullCubeTool-v1',native_horizon=100,pull_decision_boundary='PULL force>0.5N and cumulative cubeXY displacement>1mm since verified PICK_TOOL; not necessarily first physical contact',prefix='unmodified strong geometry/native timing',action={'endpoint_yz':'endpoint lateral +/-15mm and worldZ +/-4mm','local_yaw':'local lateral +/-15mm and absolute yaw +/-0.15rad','relative_yaw':'local lateral +/-15mm and measured-relative world yaw +/-0.15rad','axial_pitch':'local axial +/-25mm and measured-relative local pitch +/-0.2rad'}[a.correction]+'; infinity norm <=0.1 continues original native path',args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},no_state_restores=True),indent=2));e=make_env();e.correction=a.correction
 try:
  for seed in a.seeds:
   for grasp in a.grasps:
    for action in actions:
     start=time.monotonic();error=prefix(e,seed,grasp)
     if e.contact_event is not None:error=suffix(e,action)
     finish(e);row=save(e,a.out,seed,grasp,action,error,start,len(rows));rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print({k:v for k,v in row.items()if k not in ['path','sha256','initial_sha256']},flush=True)
 finally:e.close()
if __name__=='__main__':main()
