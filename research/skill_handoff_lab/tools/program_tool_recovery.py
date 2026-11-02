"""Shared native100 recovery primitives; no state restore, new task or success rule.
Decision at actual PULL entry (after the complete original HOOK execution).
All interventions retain the same planner/dynamics; zero executes original plan.
"""
import argparse,json,time,random,shutil,hashlib,subprocess
from pathlib import Path
import numpy as np
import program_post_contact as old
r=old.r

def geometry(e):
 u=e.unwrapped;t=r.sp(u.l_shape_tool);c=r.sp(u.cube);local=t.inv()*c
 return dict(cube_in_tool=r.rawpose(local),tilt_rad=float(np.arccos(np.clip(r.quat2mat(t.q)[2,2],-1,1))),remaining=100-len(e.records),measurement=e.measure())

class EarlyRecovery(Exception):pass
class RecoveryPlanner(old.control.ContractPlanner):
 def follow_path(self,result,refine_steps=0):
  e=self.env
  if not getattr(e,'early_recovery',False)or e.phase not in ['TRANSFER','HOOK','PULL']or e.early_event is not None:return super().follow_path(result,refine_steps)
  n=len(result['position'])
  for i in range(n+refine_steps):
   prev=e.measure();ret=e.step(np.r_[result['position'][min(i,n-1)],self.gripper_state]);self.elapsed_steps+=1;s=e.records[-1]
   force=max([z['force_norms_N']['held_tool']for z in s['physics_substep_contacts']]+[s['tool_cube_force_N']]);motion=np.linalg.norm(np.array(s['cube'][:2])-prev['cube'][:2])
   if e.early_event is None and force>.5 and motion>.0001:
    e.early_event=dict(step=len(e.records),phase=e.phase,force_N=force,motion_m=float(motion),measurement=e.measure())
    if e.recovery_mode!='none':raise EarlyRecovery()
  return ret

 def move_to_pose_with_screw(self,pose,**kwargs):
  e=self.env
  if e.phase!='PULL' or e.recovery_decision is not None:return super().move_to_pose_with_screw(pose,**kwargs)
  g=geometry(e);mode=e.recovery_mode
  if mode=='feedback':mode='rehook' if abs(g['cube_in_tool'][2])>.025 and g['remaining']>=24 else 'none'
  e.recovery_decision=dict(stage=2,skill='RECOVER_CONTACT',boundary_step=len(e.records),parameters=dict(mode=mode,requested=e.recovery_mode),observation=g['measurement'],geometry={k:v for k,v in g.items()if k!='measurement'},selection='shared actual HOOK completion; no native success access')
  e.contact_event=dict(step=len(e.records),phase='HOOK_COMPLETED',measurement=g['measurement'],remaining=g['remaining'])
  if mode=='none':return super().move_to_pose_with_screw(pose,**kwargs)
  u=e.unwrapped;yaw=next(d['parameters']['tool_yaw_rad']for d in reversed(e.decisions)if d['skill']=='USE_TOOL');quat=r.axangle2quat([0,0,1],yaw);R=r.quat2mat(quat)
  def move_tool(goal,label):
   att=r.sp(u.agent.tcp).inv()*r.sp(u.l_shape_tool);e.phase='PULL'
   e.plans.append(dict(phase='RECOVERY_TARGET',kind=label,step=len(e.records),tool_goal=r.rawpose(goal),actual_attachment=r.rawpose(att)))
   ret=super(RecoveryPlanner,self).move_to_pose_with_screw(goal*att.inv())
   if isinstance(ret,int)and ret==-1:raise r.Failure('RECOVERY_'+label+'_PLAN')
   return ret
  current=r.sp(u.l_shape_tool)
  if mode=='hold':
   q=np.asarray(r.serial(u.agent.robot.get_qpos()[0,:7]))
   for _ in range(4):e.step(np.r_[q,-1.])
   return super().move_to_pose_with_screw(pose,**kwargs)
  if mode=='press':
   pos=current.p.copy();pos[2]-=.008
   move_tool(r.sapien.Pose(pos,current.q),'PRESS8mm')
  else:
   if mode=='retract':move_tool(r.sapien.Pose(current.p+[0,0,.035],current.q),'RETRACT35mm')
   cube=r.sp(u.cube);cyaw=r.quat2euler(cube.q)[2];extent=.02*(abs(np.cos(cyaw-yaw))+abs(np.sin(cyaw-yaw)))
   hy=next(d['parameters']['hook_y_m']for d in reversed(e.decisions)if d['skill']=='USE_TOOL')
   pos=cube.p-R@np.array([.15-extent-.004,-hy,0]);pos[2]=.026
   move_tool(r.sapien.Pose(pos,quat),'LEVEL_REHOOK')
  # Recompute task-geometric pull with measured tool pose and same legal stroke.
  v=r.sp(u.cube).p[:2]-r.sp(u.agent.robot).p[:2];direction=R[:2,0];along=float(v@direction);perp=float(v@v)-along*along;dist=float(np.clip(along-np.sqrt(max(0.,.58**2-perp)),.25,.36))
  tool=r.sp(u.l_shape_tool);pos=tool.p+R@np.array([-dist,0,0]);return move_tool(r.sapien.Pose(pos,tool.q if mode=='press' else quat),'REPULL')

def run_episode(e,seed,grasp,mode,out,index):
 e.recovery_mode=mode;e.early_event=None;e.recovery_decision=None;e.contact_event=None;e.stability=None;e.pending=None
 random.seed(31131);np.random.seed(31131);r.torch.manual_seed(31131);t=time.monotonic();error=None
 try:old.control.execute(e,seed,grasp_x=grasp,hook_y=None,transfer_stride=1,pull_policy='goal')
 except EarlyRecovery:
  e.phase='PULL';p=RecoveryPlanner(e,vis=False,base_pose=e.unwrapped.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=1.8,joint_acc_limits=6.);p.manual=True;p.stride=1;p.gripper_state=-1.
  try:p.move_to_pose_with_screw(r.sp(e.unwrapped.agent.tcp))
  except(r.Failure,r.Horizon)as exc:error=repr(exc)
  finally:p.close()
 except(r.Failure,r.Horizon)as exc:error=repr(exc)
 old.finish(e)
 if e.recovery_decision is not None:e.decisions.append(e.recovery_decision)
 row=old.save(e,out,seed,grasp,[0,0],error,t,index);row.update(mode=mode,early_event=e.early_event,recovery=None if e.recovery_decision is None else e.recovery_decision['parameters'],geometry=None if e.recovery_decision is None else e.recovery_decision['geometry']);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--early',action='store_true');p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--grasps',type=float,nargs='+',default=[.0875]);p.add_argument('--modes',nargs='+',default=['none','rehook','retract','press','hold','feedback']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 for n in ['program_tool_recovery.py','program_post_contact.py','program_tool_contract_control.py','program_stap_tool.py','program_tool_effect_probe.py','program_tool_all_contacts.py','program_multiskill_pull.py','program_multiskill_contact_probe.py','pull_tool_physics.py']:shutil.copy2(Path(__file__).with_name(n),a.out/n)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task='PullCubeTool-v1',horizon=100,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in a.out.glob('*.py')},seeds=a.seeds,grasps=a.grasps,modes=a.modes,split='DEVELOPMENT_PHYSICAL_DIAGNOSIS',restore=False,nominal='inherited corrected strong planner acceleration6 velocity1.8',decision='first moving contact'if a.early else 'actual HOOK completion before PULL; prior actual contacts retained',zero='exact inherited full native trajectory',no_hardware=True),indent=2))
 e=old.make_env();e.early_recovery=a.early;r.task.TracePlanner=RecoveryPlanner;rows=[]
 try:
  for seed in a.seeds:
   for grasp in a.grasps:
    for mode in a.modes:
     row=run_episode(e,seed,grasp,mode,a.out,len(rows));rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(seed,grasp,mode,row['success'],round(row['final_distance'],4),row['contact_step'],row['error'],flush=True)
 finally:e.close()
if __name__=='__main__':main()
