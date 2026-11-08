"""Native PlugCharger200: physical grasp pitch -> insertion pace/angular authority.
No synthetic sensor bias, no altered physics, no Y/Z offset actions.
Inherited strong depth-preserving servo and observable dwell completion.
"""
import argparse,json,gzip,hashlib,random,time,shutil,subprocess
from pathlib import Path
import numpy as np, torch, gymnasium as gym, sapien
from transforms3d.quaternions import qmult,qinverse,quat2axangle,axangle2quat
from program_skill_baselines import make_episode
from program_adaptive_contact import AdaptiveMeasure
from pull_tool_physics import serial,sp,Failure,Horizon

class PaceMethods:
    def step(self, requested):
        obs=self.observation();start=len(self.env.records);requested=np.array(requested,dtype=float);action=np.clip(requested,-1,1);u=self.env.unwrapped;initial_local=(self.goal.inv()*sp(u.charger)).p.copy();reward=0.;self.last_action=np.r_[action,0.];authorities=[]
        if self.error is None:
            for _ in range(min(4,200-len(self.env.records))):
                if len(self.env.records)>=200:break
                self.env.phase='CONTACT_SERVO';actual=sp(u.charger);tcp=sp(u.agent.tcp)
                corrected=self.goal;local=corrected.inv()*actual
                orient=2*np.arccos(np.clip(abs(local.q[0]),0,1));lateral=np.linalg.norm(local.p[1:]);aligned=lateral<.0004 and orient<.03
                advance=float(.0005+.00175*(action[0]+1));x=min(0.,float(local.p[0])+advance) if aligned or advance<0 else float(local.p[0]);desired=corrected*sapien.Pose([x,0,0])
                self.bias_p=np.clip(self.bias_p+.2*(desired.p-actual.p),-.008,.008);attached=tcp.inv()*actual;target=desired*attached.inv();dp=target.p+self.bias_p-tcp.p;dp*=min(1.,.004/max(np.linalg.norm(dp),1e-9))
                dq=qmult(target.q,qinverse(tcp.q));axis,ang=quat2axangle(dq)
                if ang>np.pi:ang-=2*np.pi
                q=qmult(axangle2quat(axis,np.clip(ang,-(.01+.035*(action[1]+1)),(.01+.035*(action[1]+1)))),tcp.q);target=sapien.Pose(tcp.p+dp,q);qnow=np.asarray(serial(u.agent.robot.get_qpos()[0]))
                status,solutions=self.planner.planner.IK(self.planner.planner.transform_goal_to_wrt_base(np.r_[target.p,target.q]),qnow,n_init_qpos=1,threshold=1e-4)
                if status!='Success':self.error='SERVO_IK_FAILED';break
                raw_delta=solutions[0][:7]-qnow[:7];authorities.append([float(aligned and advance>0),float(np.any(np.abs(raw_delta)>.05))]);command=qnow[:7]+np.clip(raw_delta,-.05,.05);self.env.step(np.r_[command,-1.])
                self.env.records[-1]['servo']=dict(lateral_m=float(lateral),orientation_rad=float(orient),advance=bool(aligned),desired_local_x=x)
                self.env.records[-1]['skill_action']=dict(requested=requested.tolist(),clipped=action.tolist(),applied_goal_offset_m=[0,0,0],angular_step_limit_rad=float(.01+.035*(action[1]+1)),advance_m=advance,forward_allowed=bool(aligned and advance>0),joint_limit_active=bool(authorities[-1][1]))
                measured=self.goal.inv()*sp(u.charger)
                if self.completion_reached():break
        measured=self.goal.inv()*sp(u.charger);success=self.completion_reached();final_local=(self.goal.inv()*sp(u.charger)).p.copy();reward=float((final_local[0]-initial_local[0])/.02)-.005
        self.done=bool(success or self.error is not None or len(self.env.records)>=200)
        if self.done:
            q=np.array(serial(u.agent.robot.get_qpos()[0,:7]));self.env.phase='TERMINAL_HOLD'
            while len(self.env.records)<200:self.env.step(np.r_[q,-1.])
            reward+=5. if bool(u.evaluate()['success'][0]) else -1.
            self.planner.close()
        self.last_authority=np.mean(authorities,axis=0) if authorities else np.zeros(2)
        self.previous_delta=self.goal.to_transformation_matrix()[:3,:3].T@(sp(u.charger).p-self.previous_position);self.previous_position=sp(u.charger).p.copy();next_obs=self.observation();self.decisions+=1
        self.transitions.append(dict(step=start,obs=obs.tolist(),requested=requested.tolist(),action=action.tolist(),next_obs=next_obs.tolist(),reward=reward,done=self.done,physics_steps=len(self.env.records)-start))
        return next_obs,reward,self.done

def make_native(density=1.,friction=None):
 ep=make_episode(density,friction)
 if density==1. and friction is None:
  ep.env.close();ep.env=AdaptiveMeasure(gym.make('PlugCharger-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',sim_backend='physx_cuda',render_backend='gpu',render_mode='rgb_array',reconfiguration_freq=1))
 # Reuse inherited completion exactly, replace only the explicitly audited action map.
 import types
 ep.step=types.MethodType(PaceMethods.step,ep);ep.completion_mode='dwell';ep.authority='pace_rotation';ep.task_family='native_grasp_result'
 return ep

def prefix_failure(summary):
 """Retain predecessor failure without leaking a later servo failure."""
 error=summary.get('error')
 if summary['prefix_steps']>=200:return error
 if error in ('CHARGER_NOT_GRASPED','CHARGER_LOST_ON_LIFT'):return error
 if isinstance(error,str) and error.startswith('PLAN_'):return error
 return None

def replay_prefix(ep,path):
 """Replay actual prior joint commands in fresh native physics, never set state.
 Prefix observations must match; otherwise the episode is not a paired experiment.
 """
 from program_charger import Planner
 from program_completion_contract import CompletionGate
 d=json.load(gzip.open(path,'rt'));seed=d['summary']['seed'];e=ep.env;ep.seed=seed;ep.bias_mm=[0.,0.];ep.error=prefix_failure(d['summary']);ep.done=False;ep.transitions=[];ep.last_action=np.zeros(3);ep.last_authority=np.zeros(2);ep.bias_p=np.zeros(3);ep.started=time.monotonic();ep.gate=CompletionGate();ep.hold_entry=None;ep.last_probe=-100;ep.probe_steps=0
 e.reset(seed=seed);u=e.unwrapped;ep.goal=u.goal_pose.sp
 ep.planner=Planner(e,vis=False,base_pose=u.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=1.,joint_acc_limits=1.);ep.planner.manual=True;ep.planner.stride=1
 n=d['summary']['prefix_steps'];error=0.
 for old in d['steps'][:n]:
  e.phase=old['phase'];e.step(np.array(old['action']));new=e.records[-1]
  error=max(error,max(float(np.max(np.abs(np.array(new[k])-old[k])))for k in ['q','qvel','tcp','tool','cube']))
 ep.prefix_steps=n;ep.prefix_boundary=e.records[-1].copy();ep.previous_position=sp(u.charger).p.copy();ep.previous_delta=np.zeros(3);ep.prefix_replay=dict(source=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),max_physical_error=error,real_control_steps=n,planner_queries_added=0,state_setters_used=False)
 if error>1e-5:raise RuntimeError('Physical prefix failed equality: '+str(error))
 return ep.observation()

def save(ep,out,index,angle,policy):
 e=ep.env;row=dict(seed=ep.seed,angle=angle,policy=policy,success=e.records[-1]['native_success'],steps=len(e.records),error=ep.error,prefix_steps=ep.prefix_steps,initial_sha256=e.initial_sha,wall_s=time.monotonic()-ep.started,plan_calls=len(e.calls),planning_wall_s=sum(c['wall_s']for c in e.calls),decisions=len(ep.transitions),final=serial(e.unwrapped.evaluate()))
 u=e.unwrapped;row['task']=u.spec.id;row['actual_charger_mass_kg']=float(u.charger._bodies[0].mass);row['physical_variant']=u.adaptation_metadata()if hasattr(u,'adaptation_metadata')else None
 row['prefix_replay']=getattr(ep,'prefix_replay',None)
 path=out/f'{index:05d}_{ep.seed}.json.gz'
 with gzip.open(path,'wt')as f:json.dump(dict(summary=row,initial=e.initial,initial_measure=e.initial_measure,steps=e.records,plans=e.plans,calls=e.calls,transitions=ep.transitions,prefix_boundary=ep.prefix_boundary),f)
 return dict(row,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def action_for(name,obs):
 if name=='nominal':return np.array([-.142857142857,-.142857142857]) # 2mm, .04rad; exact inherited nominal
 if name=='fast':return np.array([1.,1.])
 if name=='advance_fast':return np.array([1.,-.142857142857])
 if name=='rotate_fast':return np.array([-.142857142857,1.])
 if name=='advance_soft':return np.array([-1.,-.142857142857])
 if name=='rotate_soft':return np.array([-.142857142857,-1.])
 if name=='soft':return np.array([-1.,-1.])
 if name=='force_guard':
  force=np.linalg.norm(obs[9:12])*10
  return np.array([-.142857142857,-1. if force>8 else -.142857142857])
 if name=='feedback':
  # Shared full-state geometric/contact law, no outcome label.
  force=np.linalg.norm(obs[9:12])*10;lat=np.linalg.norm(obs[1:3])*.025;angle=np.linalg.norm(obs[3:6])*.2
  return np.array([-1. if force>8 else 1., -1. if force>8 and lat<.001 and angle<.03 else 1.])
 raise ValueError(name)

def main():
 p=argparse.ArgumentParser();p.add_argument('--density',type=float,default=1.);p.add_argument('--friction',type=float);p.add_argument('--prefix-root',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--angles',type=float,nargs='+',default=[15,30,45]);p.add_argument('--policies',nargs='+',default=['nominal','fast','soft','feedback']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 files=['program_native_charger_pace.py','program_skill_baselines.py','program_adaptive_contact.py','program_adaptive_task.py','program_charger.py','program_completion_contract.py','pull_tool_physics.py']
 for n in files:shutil.copy2(Path(__file__).with_name(n),a.out/n)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task='PlugCharger-v1'if a.density==1. and a.friction is None else 'ARES-AdaptivePlug-v0',horizon=200,success='unchanged native criterion',bias=[0,0],physics='native'if a.density==1. and a.friction is None else 'named construction-time density/socket-friction variant; not official native score',args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},predecessor='actual grasp pitch; no state restore',successor='advance0.5..4mm, angular step0.01..0.08rad every4 physical steps; inherited depth preserving integration and completion',split='NEW_DESIGN_NOT_SEALED',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in a.out.glob('*.py')},optimizer_updates=0),indent=2));ep=make_native(a.density,a.friction);rows=[]
 try:
  for seed in a.seeds:
   for angle in a.angles:
    for policy in a.policies:
     random.seed(31031);np.random.seed(31031);torch.manual_seed(31031);source=next((z for z in json.loads((a.prefix_root/'RESULTS.json').read_text())if z['seed']==seed and z['angle']==angle and z['policy']=='nominal'),None)if a.prefix_root else None
     obs=replay_prefix(ep,source['path'])if source else ep.reset(seed,(0,0),angle=angle)
     while not ep.done:obs,reward,done=ep.step(action_for(policy,obs))
     row=save(ep,a.out,len(rows),angle,policy);rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(seed,angle,policy,row['success'],row['prefix_steps'],row['error'],flush=True)
 finally:ep.close()
if __name__=='__main__':main()
