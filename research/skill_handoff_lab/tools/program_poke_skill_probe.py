"""Native50 PokeCube scouting with real PickPeg -> PokeCube skills.
No state setters, unchanged native goal and robot-static success condition.
"""
import argparse,json,gzip,time,hashlib,shutil
from pathlib import Path
import numpy as np
import program_tool_contract_control as control
r=control.run
class PokeRecorder(r.ContactRecorder):
 def __init__(self,env):env.unwrapped.l_shape_tool=env.unwrapped.peg;super().__init__(env)
 def measure(self):
  self.unwrapped.l_shape_tool=self.unwrapped.peg;d=super().measure();d['goal']=r.rawpose(r.sp(self.unwrapped.goal_region));d['goal_error_m']=float(np.linalg.norm(np.asarray(d['cube'][:2])-d['goal'][:2]));return d

def execute(env,seed,gx):
 env.skill_results=[];env.reset(seed=seed);u=env.unwrapped;p=control.ContractPlanner(env,vis=False,base_pose=u.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=1.8,joint_acc_limits=6.);p.manual=True;p.stride=1
 def move(pose,phase):
  env.phase=phase
  if p.move_to_pose_with_screw(pose)==-1:raise r.Failure('PLAN_'+phase)
 try:
  tool=r.sp(u.peg);closing=np.array([0.,1.,0.]);closing*=1 if np.dot(closing,r.quat2mat(r.sp(u.agent.tcp).q)[:,1])>=0 else -1
  grasp=u.agent.build_grasp_pose(np.array([0,0,-1]),closing,tool.p+np.array([gx,0,0]));env.decisions.append(dict(stage=0,skill='PICK_PEG',boundary_step=0,parameters=dict(grasp_offset_m=gx),observation=env.measure()))
  
  if not getattr(env,'direct_approach',False):move(grasp*r.sapien.Pose([0,0,-.05]),'REACH')
  move(grasp,'DESCEND');p.close_gripper(t=4)
  if not bool(u.agent.is_grasping(u.peg)[0]):raise r.Failure('PEG_NOT_GRASPED')
  move(r.sapien.Pose(r.sp(u.agent.tcp).p+[0,0,.04],r.sp(u.agent.tcp).q),'LIFT');stable=control.run.task.check_stability(env,p)
  if not stable['pass']:raise r.Failure('UNSTABLE_PICK_CONTRACT')
  env.skill_results.append(dict(skill='PICK_PEG',boundary_step=len(env.records),physical_success=True,measurement=env.measure(),remaining_steps=50-len(env.records),stability=stable));env.decisions.append(dict(stage=1,skill='POKE_CUBE',boundary_step=len(env.records),parameters=dict(target_xy=r.sp(u.goal_region).p[:2].tolist(),clearance_m=.005),observation=env.measure()))
  delta=r.sp(u.goal_region).p[:2]-r.sp(u.cube).p[:2];yaw=float(np.arctan2(delta[1],delta[0]));quat=r.axangle2quat([0,0,1],yaw);R=r.quat2mat(quat);cube_yaw=r.quat2euler(r.sp(u.cube).q)[2];extent=.02*(abs(np.cos(cube_yaw-yaw))+abs(np.sin(cube_yaw-yaw)))
  def tool_goal(xyz,phase):move(r.sapien.Pose(xyz,quat)*(r.sp(u.agent.tcp).inv()*r.sp(u.peg)).inv(),phase)
  origin=r.sp(u.cube).p-R@np.array([.12+extent+.005,0,0]);origin[2]=.06;tool_goal(origin,'TRANSFER');origin[2]=.03;tool_goal(origin,'HOOK')
  origin=r.sp(u.goal_region).p-R@np.array([.12+extent,0,0]);origin[2]=.03;tool_goal(origin,'PULL')
 finally:p.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--direct-approach',action='store_true');p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[747000,747001]);p.add_argument('--grasps',type=float,nargs='+',default=[-.04,.04]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
 e=PokeRecorder(r.gym.make('PokeCube-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));e.effect_mode='corrected';e.corrected_phases=['REACH','DESCEND','LIFT','TRANSFER','HOOK','PULL'];e.arrival_hold_max=2;e.stability_max=6;e.direct_approach=a.direct_approach
 try:
  for seed in a.seeds:
   for g in a.grasps:
    error=None;t=time.monotonic()
    try:execute(e,seed,g)
    except (r.Horizon,r.Failure)as ex:error=repr(ex)
    q=np.asarray(r.serial(e.unwrapped.agent.robot.get_qpos()[0,:7]));e.phase='TERMINAL_HOLD'
    while len(e.records)<50:e.step(np.r_[q,-1.])
    row=dict(seed=seed,grasp=g,success=e.records[-1]['native_success'],error=error,steps=50,initial_sha256=e.initial_sha,wall_s=time.monotonic()-t,goal_error_m=e.records[-1]['goal_error_m'],decisions=len(e.decisions));f=a.out/f'{seed}_g{g}.json.gz'
    with gzip.open(f,'wt')as z:json.dump(dict(summary=row,initial=e.initial,initial_measure=e.initial_measure,steps=e.records,plans=e.plans,decisions=e.decisions,skill_results=e.skill_results),z)
    row.update(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest());rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
 finally:e.close()
 for name in [Path(__file__).name,'program_tool_contract_control.py','pull_tool_physics.py','program_tool_all_contacts.py','program_multiskill_contact_probe.py']:shutil.copy2(Path(__file__).with_name(name),a.out/name)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task='PokeCube-v1',native_horizon=50,native_success='cube XY within0.05m goal and robot static, unchanged',new_method=False,direct_approach=a.direct_approach,scope='initial strong-controller scout, not benchmark or RL gain'),indent=2))
if __name__=='__main__':main()
