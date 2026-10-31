"""EXP-002 full-task physical adaptation interface. No state restoration shortcuts."""
import argparse, copy, gzip, hashlib, json, random, time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import sapien
from transforms3d.quaternions import qinverse, qmult, quat2axangle, axangle2quat
from transforms3d.euler import euler2quat
from charger_probe import compute_grasp_info_by_obb, trimesh
from pull_tool_physics import Recorder, Failure, Horizon, serial, sp
from program_charger import Measure, Planner
import program_adaptive_task

class AdaptiveMeasure(Measure):
    def __init__(self, env):
        gym.Wrapper.__init__(self,env)
        self.horizon=200; self.video=None; self.phase='RESET'; self.records=[]; self.plans=[]; self.writer=None
    def step(self, action):
        result=super().step(action)
        self.records[-1]['tool_socket_force_vector_N']=serial(self.unwrapped.scene.get_pairwise_contact_forces(self.unwrapped.charger,self.unwrapped.receptacle)[0])
        return result

class Episode:
    """Legal skill authority: lateral goal offsets +/-1.5mm, advance -2..4mm.

    Controller sees measured tool and biased socket geometry, actual contact proxy,
    last applied skill action and remaining native budget. Hidden socket bias never
    enters observations; full truth remains only in evidence/native evaluation.
    """
    def __init__(self, density=1., friction=None):
        self.env=AdaptiveMeasure(gym.make('ARES-AdaptivePlug-v0',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',sim_backend='physx_cuda',render_backend='gpu',render_mode='rgb_array',reconfiguration_freq=1,charger_density_scale=density,socket_friction=friction))
        self.physics_steps=0;self.decisions=0
    def reset(self, seed, bias_mm=(0.,0.), angle=15.):
        self.seed=seed;self.bias_mm=list(bias_mm);self.error=None;self.done=False;self.transitions=[];self.last_action=np.zeros(3);self.last_authority=np.zeros(2);self.bias_p=np.zeros(3);self.env.reset(seed=seed);self.started=time.monotonic();u=self.env.unwrapped
        self.goal=u.goal_pose.sp*sapien.Pose([0,* (np.array(bias_mm)*.001)])
        p=Planner(self.env,vis=False,base_pose=u.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=1.,joint_acc_limits=1.)
        self.planner=p;p.manual=True;p.stride=1
        def move(target,label):
            self.env.phase=label;p.stride=2 if label in ['REACH','LIFT','ALIGN'] else 1
            if p.move_to_pose_with_screw(target,refine_steps=4)==-1:raise Failure('PLAN_'+label)
        try:
            box=trimesh.primitives.Box(extents=np.array(u._base_size)*2,transform=u.charger_base_pose.sp.to_transformation_matrix())
            g=compute_grasp_info_by_obb(box,approaching=np.array([0,0,-1]),target_closing=sp(u.agent.tcp).to_transformation_matrix()[:3,1],depth=.025)
            pose=u.agent.build_grasp_pose(np.array([0,0,-1]),g['closing'],g['center'])*sapien.Pose(q=euler2quat(0,np.deg2rad(angle),0))
            move(pose*sapien.Pose([0,0,-.05]),'REACH');move(pose,'DESCEND');p.close_gripper()
            if not bool(u.agent.is_grasping(u.charger)[0]):raise Failure('CHARGER_NOT_GRASPED')
            move(sapien.Pose(sp(u.agent.tcp).p+[0,0,.04],sp(u.agent.tcp).q),'LIFT')
            if not bool(u.agent.is_grasping(u.charger)[0]):raise Failure('CHARGER_LOST_ON_LIFT')
            attached=sp(u.agent.tcp).inv()*sp(u.charger)
            move(self.goal*sapien.Pose([-.025,0,0])*attached.inv(),'ALIGN')
        except (Failure,Horizon) as e:self.error=str(e)
        self.prefix_steps=len(self.env.records)
        self.prefix_boundary=copy.deepcopy(self.env.records[-1]) if self.env.records else None
        self.previous_position=sp(u.charger).p.copy();self.previous_delta=np.zeros(3)
        return self.observation()
    def observation(self):
        u=self.env.unwrapped;local=self.goal.inv()*sp(u.charger);force=np.array(serial(u.scene.get_pairwise_contact_forces(u.charger,u.receptacle)[0]));force=self.goal.to_transformation_matrix()[:3,:3].T@force
        # Physical state/effect features, no true socket pose, seed or hidden bias.
        velocity=self.goal.to_transformation_matrix()[:3,:3].T@np.array(serial(u.charger.linear_velocity[0]))
        axis,angle=quat2axangle(local.q)
        if angle>np.pi:angle-=2*np.pi
        return np.clip(np.r_[local.p/.025,axis*angle/.2,velocity/.1,force/10,self.last_action,self.previous_delta/.004,(200-len(self.env.records))/200,self.last_authority],-10,10).astype(np.float32)
    def step(self, requested):
        obs=self.observation();start=len(self.env.records);requested=np.array(requested,dtype=float);action=np.clip(requested,-1,1);u=self.env.unwrapped;initial_local=(self.goal.inv()*sp(u.charger)).p.copy();reward=0.;self.last_action=action.copy();authorities=[]
        if self.error is None:
            for _ in range(min(4,200-len(self.env.records))):
                self.env.phase='CONTACT_SERVO';actual=sp(u.charger);tcp=sp(u.agent.tcp)
                corrected=self.goal*sapien.Pose([0,action[0]*.0015,action[1]*.0015]);local=corrected.inv()*actual
                orient=2*np.arccos(np.clip(abs(local.q[0]),0,1));lateral=np.linalg.norm(local.p[1:]);aligned=lateral<.0004 and orient<.03
                advance=float(np.clip(.002+.004*action[2],-.002,.004));x=min(0.,float(local.p[0])+advance) if aligned or advance<0 else float(local.p[0]);desired=corrected*sapien.Pose([x,0,0])
                self.bias_p=np.clip(self.bias_p+.2*(desired.p-actual.p),-.008,.008);attached=tcp.inv()*actual;target=desired*attached.inv();dp=target.p+self.bias_p-tcp.p;dp*=min(1.,.004/max(np.linalg.norm(dp),1e-9))
                dq=qmult(target.q,qinverse(tcp.q));axis,ang=quat2axangle(dq)
                if ang>np.pi:ang-=2*np.pi
                q=qmult(axangle2quat(axis,np.clip(ang,-.04,.04)),tcp.q);target=sapien.Pose(tcp.p+dp,q);qnow=np.asarray(serial(u.agent.robot.get_qpos()[0]))
                status,solutions=self.planner.planner.IK(self.planner.planner.transform_goal_to_wrt_base(np.r_[target.p,target.q]),qnow,n_init_qpos=1,threshold=1e-4)
                if status!='Success':self.error='SERVO_IK_FAILED';break
                raw_delta=solutions[0][:7]-qnow[:7];authorities.append([float(aligned and advance>0),float(np.any(np.abs(raw_delta)>.05))]);command=qnow[:7]+np.clip(raw_delta,-.05,.05);self.env.step(np.r_[command,-1.])
                self.env.records[-1]['servo']=dict(lateral_m=float(lateral),orientation_rad=float(orient),advance=bool(aligned),desired_local_x=x)
                self.env.records[-1]['skill_action']=dict(requested=requested.tolist(),clipped=action.tolist(),applied_goal_offset_m=[0,action[0]*.0015,action[1]*.0015],advance_m=advance,forward_allowed=bool(aligned and advance>0),joint_limit_active=bool(authorities[-1][1]))
                measured=self.goal.inv()*sp(u.charger)
                if np.linalg.norm(measured.p)<=.004 and 2*np.arccos(np.clip(abs(measured.q[0]),0,1))<=.03:break
        measured=self.goal.inv()*sp(u.charger);success=np.linalg.norm(measured.p)<=.004 and 2*np.arccos(np.clip(abs(measured.q[0]),0,1))<=.03;final_local=(self.goal.inv()*sp(u.charger)).p.copy();reward=float((final_local[0]-initial_local[0])/.02)-.005
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
    def save(self,out,label):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);env=self.env;u=env.unwrapped;self.physics_steps+=len(env.records)
        row=dict(seed=self.seed,label=label,success=bool(u.evaluate()['success'][0]),error=self.error,steps=len(env.records),prefix_steps=self.prefix_steps,initial_sha256=env.initial_sha,bias_mm=self.bias_mm,physical_config=u.adaptation_metadata(),physical_readback=u.physical_readback(),wall_s=time.monotonic()-self.started,final=serial(u.evaluate()),decisions=len(self.transitions),planning_wall_s=sum(c['wall_s'] for c in env.calls))
        path=out/f'{label}_{self.seed}.json.gz'
        if path.exists():raise FileExistsError(path)
        payload=json.dumps(dict(summary=row,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans,calls=env.calls,transitions=self.transitions,prefix_boundary=self.prefix_boundary),separators=(',',':'))
        with gzip.open(path,'wt') as f:f.write(payload)
        row.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());return row
    def close(self):self.env.close()

class SearchControl:
    """Measured-progress pattern search, no hidden bias or outcome labels."""
    def __init__(self):self.index=0;self.age=0;self.backoff=0;self.best=-np.inf;self.stall=0;self.points=[(0,0),(.8,0),(-.8,0),(0,.8),(0,-.8),(.8,.8),(-.8,.8),(.8,-.8),(-.8,-.8)]
    def __call__(self,obs):
        x=obs[0]*.025
        if x>self.best+.0003:self.best=x;self.stall=0
        else:self.stall+=1
        if self.stall>=3 and x>-.019:self.index=(self.index+1)%len(self.points);self.stall=0;self.backoff=0
        retract=self.backoff>0
        if retract:self.backoff-=1
        return np.r_[self.points[self.index],-1. if retract else 0.]

class RLSControl:
    """Online measured actuator-to-object sensitivity; bounded inverse correction.
    Uses observed displacement and prior commanded lateral residual. This is a
    strong classical baseline, not a new learning method or oracle calibration.
    """
    def __init__(self):self.B=np.eye(2);self.P=np.eye(2)*100;self.prev=None;self.previous_action=np.zeros(2);self.search=SearchControl()
    def __call__(self,obs):
        current=obs[1:3]*.025
        if self.prev is not None:
            du=(self.previous_action-self.prev[1])/.001;dy=(current-self.prev[0])/.001
            if np.linalg.norm(du)>1e-5:
                k=self.P@du/(.98+du@self.P@du);self.B+=np.outer(dy-self.B@du,k);self.P=(self.P-np.outer(k,du)@self.P)/.98
        base=self.search(obs);correction=np.linalg.solve(self.B.T@self.B+.1*np.eye(2),self.B.T@(base[:2]*.0015-current))
        a=np.clip(self.previous_action+np.clip(correction,-.0004,.0004),-.0015,.0015)
        self.prev=(current.copy(),self.previous_action.copy());self.previous_action=a
        return np.r_[a/.0015,base[2]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',default='727000,727001');p.add_argument('--density',type=float,default=1.);p.add_argument('--friction',type=float);p.add_argument('--bias-y-mm',type=float,default=0.);p.add_argument('--bias-z-mm',type=float,default=0.);p.add_argument('--policies',default='geometry,search,rls');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);random.seed(31031);np.random.seed(31031);torch.manual_seed(31031);ep=Episode(a.density,a.friction);rows=[]
    manifest=dict(task='ARES-AdaptivePlug-v0',native_physics=a.density==1 and a.friction is None,observation_bias_mm=[a.bias_y_mm,a.bias_z_mm],hidden_bias_not_in_observation=True,original_success_horizon=True,sources={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['program_adaptive_contact.py','program_adaptive_task.py','program_charger.py','pull_tool_physics.py']},optimizer_steps=0)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));(a.out/'source.py').write_bytes(Path(__file__).read_bytes())
    for n in manifest['sources']:(a.out/n).write_bytes((Path(__file__).parent/n).read_bytes())
    try:
        for seed in map(int,a.seeds.split(',')):
            for name in a.policies.split(','):
                controller=SearchControl() if name=='search' else RLSControl() if name=='rls' else lambda o:np.array([{'geometry':0.,'fixed_plus':.55,'fixed_minus':-.55}[name],0,0])
                obs=ep.reset(seed,(a.bias_y_mm,a.bias_z_mm))
                while not ep.done:obs,reward,done=ep.step(controller(obs))
                rows.append(ep.save(a.out,name));(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
    finally:ep.close()
if __name__=='__main__':main()
