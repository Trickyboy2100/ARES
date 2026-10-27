"""State-based dynamic workpiece pushing; new ARES tasks, NOT public benchmarks.

Uses the deployed DirectRLEnv/InteractiveScene and the previously audited JAKA
Jacobian controller. A workpiece is free, not attached or repositioned by an
action. Only reset writes state. No field point clouds/configurations are used.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np


def run(a):
    import paired_physx_video_hooks as paired_video
    source_snapshot=Path(__file__).read_bytes();(a.output/'task_source.py').write_bytes(source_snapshot)
    method_source=Path(__file__).with_name('contact_credit_v2.py').read_bytes()
    (a.output/'boundary_credit_source.py').write_bytes(method_source)
    (a.output/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2))
    import torch,gymnasium as gym
    from scipy.spatial.transform import Rotation
    import isaaclab.sim as sim
    from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs import DirectRLEnv,DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor,ContactSensorCfg,Camera,CameraCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply,quat_mul,quat_conjugate,axis_angle_from_quat,skew_symmetric_matrix
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(2)
    meta=json.loads((a.asset/'asset_manifest.json').read_text())
    assert hashlib.sha256((a.asset/'jaka_paddle.usda').read_bytes()).hexdigest()==meta['asset_sha256']
    base=np.array(meta['body_bases']['right']);xyzw=Rotation.from_matrix(base[:3,:3]).as_quat()
    center=np.array(meta['paddle_center_body_m']);q0=meta['start_q'];names=[f'joint_{i+1}' for i in range(6)]
    family=[('slot' if i%2 else 'push') if a.family=='both' else a.family for i in range(a.num_envs)]
    scenario=json.loads(a.scenario_file.read_text()) if a.scenario_file else {'groups':[{'id':'development','gap':.052,'goal_x':.110,'slot_y':-.025}]}
    specs=[scenario['groups'][min(len(scenario['groups'])-1,i*len(scenario['groups'])//a.num_envs)] for i in range(a.num_envs)]

    simulation_config=sim.SimulationCfg(dt=1/120,render_interval=8,device=f'cuda:{a.gpu}')
    @configclass
    class Cfg(DirectRLEnvCfg):
        decimation=2;episode_length_s=a.episode_seconds;action_space=3;observation_space=56;state_space=0
        sim=simulation_config
        scene=InteractiveSceneCfg(num_envs=a.num_envs,env_spacing=2.5,replicate_physics=False,clone_in_fabric=False)
        seed=a.seed

    class Env(DirectRLEnv):
        def __init__(self):
            self.physics_steps=0;self.records=[];self.trace=[];self.evaluation=False;self.write_count=0
            super().__init__(Cfg())
            self.ids,_=self.robot.find_joints(names,preserve_order=True);self.link=self.robot.find_bodies('Link_6')[0][0]
            self.home=torch.tensor(q0,device=self.device);self.center=torch.tensor(center,device=self.device,dtype=torch.float32)
            self.offset=torch.tensor(meta['held_offset_link6_m'],device=self.device,dtype=torch.float32).repeat(self.num_envs,1)
            x,y,z,w=Rotation.from_matrix(meta['level_link6_rotation']).as_quat()
            self.orientation=torch.tensor([w,x,y,z],device=self.device,dtype=torch.float32).repeat(self.num_envs,1)
            self.low=torch.tensor(meta['joint_lower'],device=self.device);self.high=torch.tensor(meta['joint_upper'],device=self.device)
            self.command=self.center.repeat(self.num_envs,1);self.qcommand=self.home.repeat(self.num_envs,1)
            self.vcommand=torch.zeros_like(self.qcommand);self.actions=torch.zeros((self.num_envs,2),device=self.device)
            self.phase=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
            self.hold=torch.zeros_like(self.phase);self.previous=torch.zeros(self.num_envs,device=self.device)
            self.returns=torch.zeros_like(self.previous);self.cases=torch.full_like(self.phase,-1)
            self.goals=self.center.repeat(self.num_envs,2,1)
            self.goals[:,0,0]+=.075
            self.goals[:,:,2]=float(center[2]-.005)
            for i,f in enumerate(family):
                self.goals[i,1,0]+=specs[i]['goal_x']
                self.goals[i,:,1]+=specs[i]['slot_y'] if f=='slot' else 0.
            self.reward=torch.zeros_like(self.previous);self.success=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            self.max_force=torch.zeros_like(self.previous)
            self.applied_delta=torch.zeros((self.num_envs,2),device=self.device)
            self.raw_actions=torch.zeros((self.num_envs,3),device=self.device)
            self.switch_signal=torch.zeros(self.num_envs,device=self.device);self.dwell=torch.zeros_like(self.phase)
            self.handoffs=[None]*self.num_envs;self.contacts=[]
            self.min_final=torch.full((self.num_envs,),float('inf'),device=self.device)
            self.geometry=torch.tensor([[s.get(k,d) for k,d in [('gap',.052),('goal_x',.110),('slot_y',-.025),('mass',.05),('mu',.3)]]+
                [np.cos(s.get('direction',0.)),np.sin(s.get('direction',0.))] for s in specs],device=self.device,dtype=torch.float32)
            for i,f in enumerate(family):
                phi=specs[i].get('direction',0.);n=torch.tensor([np.cos(phi),np.sin(phi)],device=self.device)
                t=torch.tensor([-np.sin(phi),np.cos(phi)],device=self.device)
                self.goals[i,0,:2]=self.center[:2]+n*.075+t*(specs[i]['slot_y'] if f=='slot' else 0.)
                self.goals[i,1,:2]=self.center[:2]+n*specs[i]['goal_x']+t*(specs[i]['slot_y'] if f=='slot' else 0.)
        def _setup_scene(self):
            robotcfg=ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
                spawn=sim.UsdFileCfg(usd_path=str((a.asset/'jaka_paddle.usda').resolve()),activate_contact_sensors=True,
                    articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                        solver_position_iteration_count=32,solver_velocity_iteration_count=4)),
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(base[:3,3]),rot=(float(xyzw[3]),*map(float,xyzw[:3])),joint_pos=dict(zip(names,q0))),
                actuators={'arm':ImplicitActuatorCfg(joint_names_expr=['joint_.*'],stiffness=1500.,damping=80.,effort_limit_sim=100.,velocity_limit_sim=.1)})
            self.robot=Articulation(robotcfg)
            self.cube=RigidObject(RigidObjectCfg(prim_path='/World/envs/env_.*/Workpiece',
                spawn=sim.CuboidCfg(size=(.04,.04,.04),rigid_props=sim.RigidBodyPropertiesCfg(max_depenetration_velocity=.5),
                    collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),mass_props=sim.MassPropertiesCfg(mass=.05),
                    physics_material=sim.RigidBodyMaterialCfg(static_friction=.4,dynamic_friction=.3,restitution=0.),
                    visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.8,.12,.1))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(center+[.05,0,-.003]))))
            table=sim.CuboidCfg(size=(.26,.30,.03),collision_props=sim.CollisionPropertiesCfg(),
                physics_material=sim.RigidBodyMaterialCfg(static_friction=.4,dynamic_friction=.3,restitution=0.),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.25,.35,.45)))
            table.func('/World/envs/env_0/Table',table,translation=tuple(center+[.145,0,-.04]))
            self.scene.clone_environments(copy_from_source=False)
            # Heterogeneous static slot rails, same process and independent physics.
            for i,f in reversed(list(enumerate(family))):
                if f!='slot':continue
                for side in [-1,1]:
                    length=specs[i].get('rail_length',.05)
                    rail=sim.CuboidCfg(size=(length,.015,.04),collision_props=sim.CollisionPropertiesCfg(),
                        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.1,.55,.25)))
                    phi=specs[i].get('direction',0.);c,s=np.cos(phi),np.sin(phi)
                    x=specs[i].get('entry_x',specs[i]['goal_x']-.01)+length/2;y=specs[i]['slot_y']+side*(specs[i]['gap']/2+.0075)
                    rail.func(f'/World/envs/env_{i}/Rail_{"L" if side<0 else "R"}',rail,
                        translation=tuple(center+[c*x-s*y,s*x+c*y,-.005]),orientation=(float(np.cos(phi/2)),0.,0.,float(np.sin(phi/2))))
            self.contact=ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/Link_.*',history_length=1))
            self.scene.articulations['robot']=self.robot;self.scene.rigid_objects['workpiece']=self.cube
            self.scene.sensors['contact']=self.contact
            self.scene.filter_collisions(global_prim_paths=[])
            sim.DomeLightCfg(intensity=1500.).func('/World/Light',sim.DomeLightCfg(intensity=1500.))
            if not a.skip_capture:
                self.camera=Camera(CameraCfg(prim_path='/World/Preview',height=1080,width=1920,data_types=['rgb'],spawn=sim.PinholeCameraCfg(clipping_range=(.01,100))))
        def state(self):
            origin=self.scene.env_origins
            paddle=self.robot.data.body_link_pos_w[:,self.link]+quat_apply(self.robot.data.body_link_quat_w[:,self.link],self.offset)-origin
            obj=self.cube.data.root_pos_w-origin;vel=self.cube.data.root_lin_vel_w
            goal=self.goals[torch.arange(self.num_envs,device=self.device),self.phase]
            return paddle,obj,vel,goal
        def _get_observations(self):
            paddle,obj,vel,goal=self.state()
            old=torch.cat((self.robot.data.joint_pos[:,self.ids]-self.home,self.robot.data.joint_vel[:,self.ids],
                (paddle-self.center)*10,(obj-self.center)*10,vel,(goal-obj)*10,(self.goals[:,1]-obj)*10,
                (self.command-paddle)*10,torch.nn.functional.one_hot(self.phase,2).float(),
                (self.episode_length_buf/self.max_episode_length)[:,None]),-1)
            quat=self.cube.data.root_quat_w;quat=quat*torch.where(quat[:,:1]<0,-1.,1.)
            yaw=torch.atan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2))
            phi=torch.atan2(self.geometry[:,6],self.geometry[:,5]);err=yaw-phi
            extra=torch.cat((quat,self.cube.data.root_ang_vel_w,(obj-paddle)*10,
                self.contact.data.net_forces_w.sum(dim=1)/10,self.geometry,self.dwell[:,None]/30,
                torch.stack((torch.sin(4*err),torch.cos(4*err)),-1)),-1)
            if a.blind_next:
                old=old.clone();extra=extra.clone();mask=self.phase==0
                old[mask,24:27]=0;extra[mask,14:16]=0;extra[mask,18:20]=0
            if a.blind_rotation:extra=extra.clone();extra[:,:7]=0;extra[:,-2:]=0
            return {'policy':torch.cat((old,extra),-1)}
        def physical_snapshot(self,i):
            paddle,obj,vel,goal=self.state()
            return {'physics_step':self.physics_steps,'control_step':int(self.episode_length_buf[i]),
                'time_s':self.physics_steps*self.physics_dt,'case_id':int(self.cases[i]),'phase':int(self.phase[i]),
                'q':self.robot.data.joint_pos[i,self.ids].cpu().tolist(),'qd':self.robot.data.joint_vel[i,self.ids].cpu().tolist(),
                'object_position':obj[i].cpu().tolist(),'object_quaternion_wxyz':self.cube.data.root_quat_w[i].cpu().tolist(),
                'object_linear_velocity':vel[i].cpu().tolist(),'object_angular_velocity':self.cube.data.root_ang_vel_w[i].cpu().tolist(),
                'paddle_position':paddle[i].cpu().tolist(),'command':self.command[i].cpu().tolist(),
                'joint_velocity_command':self.vcommand[i].cpu().tolist(),'goal':goal[i].cpu().tolist(),
                'robot_contact_force':self.contact.data.net_forces_w[i].cpu().tolist()}
        def _pre_physics_step(self,actions):
            self.raw_actions=actions.clone();self.actions=actions[:,:2].clamp(-1,1)
            self.switch_signal=actions[:,2];paddle,obj,vel,goal=self.state()
            if a.force_switch_signal is not None:self.switch_signal=torch.full_like(self.switch_signal,a.force_switch_signal)
            if a.rule_recovery:self.switch_signal=torch.where((goal[:,:2]-obj[:,:2]).norm(dim=-1)<.007,-1.,1.)
            self.phase_before_action=self.phase.clone()
            direction=goal[:,:2]-obj[:,:2];direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(.001)
            # Ray intersection with the rectangle Minkowski sum, NOT its
            # support function (which leaves a spurious gap at diagonal push).
            envelope=torch.tensor([.02+meta['paddle_dims_m'][0]/2,.02+meta['paddle_dims_m'][1]/2],device=self.device)
            clearance=(envelope/direction.abs().clamp_min(1e-6)).amin(-1,keepdim=True)
            nominal=obj[:,:2]-direction*(clearance-.0025)
            nominal_delta=(nominal-self.command[:,:2]).clamp(-.0012,.0012)
            # Establish a real rear-face contact before pushing; do not drive
            # diagonally into a cube corner while still approaching from free space.
            align=(paddle[:,0]<obj[:,0]-.037)&(abs(paddle[:,1]-obj[:,1])>.003)
            nominal_delta[align,0]=0.;nominal_delta[align,1]=(obj[align,1]-self.command[align,1]).clamp(-.0012,.0012)
            nominal_delta=torch.where(((goal[:,:2]-obj[:,:2]).norm(dim=-1)<.006)[:,None],torch.zeros_like(nominal_delta),nominal_delta)
            forward=self.geometry[:,5:7]
            forward_clearance=(envelope/forward.abs().clamp_min(1e-6)).amin(-1,keepdim=True)
            endpoint=goal[:,:2]-forward*forward_clearance
            damped_delta=((endpoint-self.command[:,:2])*a.feedback_kp-vel[:,:2]*a.feedback_kd)*self.step_dt
            damped_delta=damped_delta.clamp(-.0012,.0012)
            damped_delta[align,0]=0.;damped_delta[align,1]=(obj[align,1]-self.command[align,1]).clamp(-.0012,.0012)
            delta=self.actions*.0012
            if a.controller=='nominal':delta=nominal_delta
            elif a.controller=='goal_feedback':
                # Strong non-learning position-feedback comparator: target the
                # final rear-face contact, rather than maintaining indentation
                # behind a moving cube. Never fitted to heldout outcomes.
                endpoint=goal[:,:2]-direction*clearance
                delta=(endpoint-self.command[:,:2]).clamp(-.0012,.0012)
                delta[align,0]=0.;delta[align,1]=(obj[align,1]-self.command[align,1]).clamp(-.0012,.0012)
            elif a.controller=='fixed_push':
                delta=torch.zeros_like(delta);delta[:,0]=.0012
            elif a.controller=='residual':delta=(damped_delta+self.actions*.0006).clamp(-.0012,.0012)
            elif a.controller=='random':delta=(torch.rand_like(delta)*2-1)*.0012
            if a.controller=='damped':
                # Endpoint contact controller with real workpiece velocity damping.
                delta=damped_delta
            if a.controller=='paired':delta=torch.where((self.phase==1)[:,None],damped_delta,delta)
            if a.switch_dwell>0:
                eligible=(self.phase==0)&((goal[:,:2]-obj[:,:2]).norm(dim=-1)<.009)&(vel.norm(dim=-1)<.12)
                delta=torch.where(eligible[:,None],torch.zeros_like(delta),delta)
            if a.switch_mode=='recover':
                wait=(self.phase==1)&(self.switch_signal<0)
                delta=torch.where(wait[:,None],torch.zeros_like(delta),delta)
            if a.action_intervention=='zero':delta=torch.zeros_like(delta)
            elif a.action_intervention in ['phase0_feedback','phase1_feedback']:
                selected=self.phase==(0 if a.action_intervention=='phase0_feedback' else 1)
                delta=torch.where(selected[:,None],nominal_delta,delta)
            previous_command=self.command[:,:2].clone()
            self.command[:,:2]=(self.command[:,:2]+delta).clamp(self.center[:2]+torch.tensor([-.035,-.08],device=self.device),
                self.center[:2]+torch.tensor([.17,.08],device=self.device))
            self.command[:,2]=self.center[2]
            if a.task_version!='v5_replay':
                # Uniform anti-windup guard for ALL v6 policies/controllers.
                # Never alter actual robot/workpiece state; bound reference lead.
                self.command[:,:2]=self.command[:,:2].clamp(paddle[:,:2]-.006,paddle[:,:2]+.006)
            self.applied_delta=self.command[:,:2]-previous_command
        def _apply_action(self):
            paddle,obj,vel,goal=self.state();quat=self.robot.data.body_link_quat_w[:,self.link]
            error=torch.cat(((self.command-paddle).clamp(-.008,.008),
                axis_angle_from_quat(quat_mul(self.orientation,quat_conjugate(quat))).clamp(-.03,.03)),-1)
            jac=self.robot.root_physx_view.get_jacobians()[:,self.link-1,:,self.ids].clone()
            com_to_paddle=paddle+self.scene.env_origins-self.robot.data.body_com_pos_w[:,self.link]
            jac[:,:3,:]-=skew_symmetric_matrix(com_to_paddle)@jac[:,3:,:]
            dq=(jac.transpose(1,2)@torch.linalg.solve(jac@jac.transpose(1,2)+torch.eye(6,device=self.device)*.0001,error[:,:,None])).squeeze(-1)
            desired=dq*8.;desired*=torch.minimum(torch.ones_like(desired[:,:1]),.1/desired.abs().amax(-1,keepdim=True).clamp_min(1e-8))
            dt=self.physics_dt;self.vcommand+= (desired-self.vcommand).clamp(-.2*dt,.2*dt)
            q=self.robot.data.joint_pos[:,self.ids]
            # Differential velocity servo, not a second position integrator:
            # integrating an actual-pose error into a separate q target caused
            # windup and physically pushed past the stage acceptance region.
            self.qcommand=(q+self.vcommand*dt).clamp(self.low,self.high)
            self.robot.set_joint_position_target(self.qcommand,joint_ids=self.ids)
            self.robot.set_joint_velocity_target(self.vcommand,joint_ids=self.ids)
            self.robot.set_joint_effort_target(self.robot.root_physx_view.get_gravity_compensation_forces()[:,self.ids],joint_ids=self.ids)
            if self.evaluation:
                goal=self.goals[torch.arange(self.num_envs,device=self.device),self.phase]
                self.trace.append(torch.cat((q,obj,paddle,vel,self.phase[:,None],self.cube.data.root_quat_w,
                    self.cases[:,None],self.command,goal,self.actions,self.applied_delta,self.cube.data.root_ang_vel_w,
                    self.robot.data.joint_vel[:,self.ids],torch.full((self.num_envs,1),self.physics_steps,device=self.device),
                    self.switch_signal[:,None],self.raw_actions),dim=-1).cpu().numpy())
            self.physics_steps+=1
        def _get_dones(self):
            paddle,obj,vel,goal=self.state();q=self.robot.data.joint_pos[:,self.ids]
            dist=(goal[:,:2]-obj[:,:2]).norm(dim=-1)
            self.min_final=torch.minimum(self.min_final,(self.goals[:,1,:2]-obj[:,:2]).norm(dim=-1))
            advance=(self.phase==0)&(dist<.009)&(vel.norm(dim=-1)<.12)
            self.dwell=torch.where(advance,self.dwell+1,torch.zeros_like(self.dwell))
            if a.switch_dwell:advance=advance&(self.dwell>=a.switch_dwell)
            if a.switch_mode=='learned':advance=advance&(self.switch_signal>0)
            for i in advance.nonzero().flatten().tolist():self.handoffs[i]=self.physical_snapshot(i)
            stable=(dist<.008)&(vel.norm(dim=-1)<.025)&(abs(obj[:,2]-(self.center[2]-.005))<.008)
            if a.task_version!='v5_replay':
                quat=self.cube.data.root_quat_w
                yaw=torch.atan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2))
                phi=torch.atan2(self.geometry[:,6],self.geometry[:,5])
                yaw_error=abs(torch.atan2(torch.sin(4*(yaw-phi)),torch.cos(4*(yaw-phi)))/4)
                upright=1-2*(quat[:,1]**2+quat[:,2]**2)
                stable=stable&(vel.norm(dim=-1)<.01)&(self.cube.data.root_ang_vel_w.norm(dim=-1)<.2)&(yaw_error<np.radians(8))&(upright>np.cos(np.radians(5)))
            self.hold=torch.where(stable&(self.phase==1),self.hold+1,torch.zeros_like(self.hold));self.success=self.hold>=(8 if a.task_version=='v5_replay' else 30)
            force=self.contact.data.net_forces_w.norm(dim=-1).amax(-1);self.max_force=torch.maximum(self.max_force,force)
            tilt=axis_angle_from_quat(quat_mul(self.orientation,quat_conjugate(self.robot.data.body_link_quat_w[:,self.link]))).norm(dim=-1)
            invalid=(q<self.low-.001).any(-1)|(q>self.high+.001).any(-1)|(~torch.isfinite(q)).any(-1)
            failed=invalid|(obj[:,2]<self.center[2]-.08)|(tilt>np.radians(3.5))|(force>80.)
            timeout=self.episode_length_buf>=self.max_episode_length-1
            charged=self.actions if a.action_cost=='raw' else self.applied_delta/.0012
            self.reward=60*(self.previous-dist)-self.step_dt-.0002*charged.square().sum(-1)
            self.reward+=3*advance.float()+10*(self.success&~failed).float()-5*failed.float()
            self.phase=torch.where(advance,torch.ones_like(self.phase),self.phase)
            self.previous=(self.goals[torch.arange(self.num_envs,device=self.device),self.phase,:2]-obj[:,:2]).norm(dim=-1)
            self.returns+=self.reward
            done=self.success|failed|timeout
            for i in done.nonzero().flatten().tolist():
                if self.evaluation and self.cases[i]<0:continue
                self.records.append({'evaluation':self.evaluation,'case_id':int(self.cases[i]),'env':i,'family':family[i],
                    'success':bool(self.success[i] and not failed[i]),'failed':bool(failed[i]),'timeout':bool(timeout[i]),
                    'phase':int(self.phase[i]),'return':float(self.returns[i]),'steps':int(self.episode_length_buf[i]),
                    'distance_m':float(dist[i]),'workpiece_position_m':obj[i].cpu().tolist(),'peak_force_N':float(self.max_force[i]),
                    'pose_error_rad':float(tilt[i]),'soft_limit_invalid':bool(invalid[i]),
                    'env_interactions':int(self.common_step_counter*self.num_envs),'scene_group':specs[i]['id']})
                self.records[-1].update(post_step_state=self.physical_snapshot(i),handoff_state=self.handoffs[i],
                    stable_hold_steps=int(self.hold[i]),minimum_final_distance_m=float(self.min_final[i]),
                    failure_reason='constraint' if failed[i] else ('success' if self.success[i] else 'timeout'))
            paired_video.post_step(self,done)
            self.terminal_obs=self._get_observations()['policy'].clone()
            return done,torch.zeros_like(done)
        def _get_rewards(self):return self.reward
        def _reset_idx(self,ids):
            if ids is None:ids=self.robot._ALL_INDICES
            super()._reset_idx(ids)
            if not hasattr(self,'home'):return
            n=len(ids);q=self.robot.data.default_joint_pos[ids].clone();v=torch.zeros_like(q);q[:,self.ids]=self.home
            self.robot.write_joint_state_to_sim(q,v,env_ids=ids)
            state=self.robot.data.default_root_state[ids].clone();state[:,:3]+=self.scene.env_origins[ids]
            self.robot.write_root_pose_to_sim(state[:,:7],env_ids=ids);self.robot.write_root_velocity_to_sim(state[:,7:],env_ids=ids)
            obj=self.cube.data.default_root_state[ids].clone();obj[:,:3]=self.center+torch.tensor([.05,0,-.003],device=self.device)+self.scene.env_origins[ids]
            for k,i in enumerate(ids.tolist()):
                if family[i]=='slot':obj[k,1]+=specs[i]['slot_y']
            noise=(torch.rand((n,2),device=self.device)-.5)*(.020 if self.evaluation else .012)
            if self.evaluation:
                for k,i in enumerate(ids.tolist()):
                    index=int(self.eval_cursors[i]);self.eval_cursors[i]+=self.num_envs
                    self.cases[i]=index if index<a.eval_episodes else -1
                    noise[k]=torch.tensor(np.random.default_rng(a.eval_seed+index).uniform(-.01,.01,2),device=self.device)
            obj[:,:2]+=noise;obj[:,7:]=0
            for k,i in enumerate(ids.tolist()):
                phi=specs[i].get('direction',0.);c,s=np.cos(phi),np.sin(phi)
                if phi:
                    xy=obj[k,:2]-self.center[:2]-self.scene.env_origins[i,:2]
                    obj[k,:2]=self.center[:2]+self.scene.env_origins[i,:2]+torch.stack((c*xy[0]-s*xy[1],s*xy[0]+c*xy[1]))
                yaw=phi+specs[i].get('initial_yaw',0.)
                obj[k,3:7]=torch.tensor([np.cos(yaw/2),0.,0.,np.sin(yaw/2)],device=self.device)
            # First waypoint is a measured straight extraction/push; the next
            # slot stage retains the fixed rail center and must handle offset.
            for k,i in enumerate(ids.tolist()):
                phi=specs[i].get('direction',0.);n=torch.tensor([np.cos(phi),np.sin(phi)],device=self.device)
                t=torch.tensor([-np.sin(phi),np.cos(phi)],device=self.device)
                lateral=((obj[k,:2]-self.scene.env_origins[i,:2]-self.center[:2])*t).sum()
                self.goals[i,0,:2]=self.center[:2]+n*.075+t*lateral
                if family[i]=='push':self.goals[i,1,:2]=self.center[:2]+n*specs[i]['goal_x']+t*lateral
            for k,i in enumerate(ids.tolist()):
                self.handoffs[i]=None
            self.cube.write_root_pose_to_sim(obj[:,:7],env_ids=ids);self.cube.write_root_velocity_to_sim(obj[:,7:],env_ids=ids)
            self.command[ids]=self.center;self.qcommand[ids]=self.home;self.vcommand[ids]=0
            self.phase[ids]=0;self.hold[ids]=0;self.returns[ids]=0;self.max_force[ids]=0
            self.dwell[ids]=0
            self.min_final[ids]=float('inf')
            self.previous[ids]=(self.goals[ids,0,:2]-(obj[:,:2]-self.scene.env_origins[ids,:2])).norm(dim=-1)
            self.write_count+=len(ids)

    env=Env()
    masses=env.cube.root_physx_view.get_masses();ratio=torch.tensor([s.get('mass',.05)/.05 for s in specs],device=masses.device)
    masses[:,0]=.05*ratio;indices=torch.arange(env.num_envs,dtype=torch.int32)
    env.cube.root_physx_view.set_masses(masses,indices)
    inertia=env.cube.root_physx_view.get_inertias();inertia*=ratio[:,None]
    env.cube.root_physx_view.set_inertias(inertia,indices)
    materials=env.cube.root_physx_view.get_material_properties()
    for i,s in enumerate(specs):materials[i,:,0]=s.get('mu',.3)+.1;materials[i,:,1]=s.get('mu',.3)
    env.cube.root_physx_view.set_material_properties(materials,indices)
    obs,_=env.reset(seed=a.seed);frames=[];frame_records=[];action_log=[]
    if not a.skip_capture:
        middle=env.scene.env_origins.mean(0);span=max(2.,float(env.scene.env_origins.max()-env.scene.env_origins.min())+1.)
        eye=middle+torch.tensor([span,-span,span],device=env.device)
        env.camera.set_world_poses_from_view(eye[None,:],(middle+torch.tensor([.4,-.4,1.2],device=env.device))[None,:])
    import paired_physx_video_hooks as paired_video
    paired_video.configure(env,a.output,[0, 1, 3, 9],'/home/andyee/robot-research/ARES-Skilllab/envs/video_encoder_runtime/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2','Historical_tuned_damped_feedback')
    class Adapter(VecEnv):
        def __init__(self):super().__init__(env.num_envs,gym.spaces.Box(-np.inf,np.inf,(56,),np.float32),gym.spaces.Box(-1,1,(3,),np.float32))
        def reset(self):return env.reset(seed=a.eval_seed if env.evaluation else a.seed)[0]['policy'].cpu().numpy()
        def step_async(self,actions):self.actions=actions
        def step_wait(self):
            obs,r,t,tr,info=env.step(torch.as_tensor(self.actions,device=env.device,dtype=torch.float32));done=(t|tr).cpu().numpy()
            infos=[{'terminal_observation':env.terminal_obs[i].cpu().numpy()} if done[i] else {} for i in range(env.num_envs)]
            return obs['policy'].cpu().numpy(),r.cpu().numpy(),done,infos
        def close(self):pass
        def get_attr(self,name,indices=None):return [getattr(env,name,None)]*env.num_envs
        def set_attr(self,name,value,indices=None):setattr(env,name,value)
        def env_method(self,*args,**kwargs):return [None]*env.num_envs
        def env_is_wrapped(self,*args,**kwargs):return [False]*env.num_envs
    from contact_credit_v2 import policy_class
    cls=PPO if a.credit_mode=='standard' else policy_class()
    class StudyPPO(cls):
        def train(self):
            super().train()
            if self.num_timesteps%a.checkpoint_every==0:
                self.save(a.output/'checkpoints'/f'policy_{self.num_timesteps}_after_update.zip')
    actor_class='MlpPolicy';policy_kwargs={'net_arch':[64,64],'log_std_init':a.initial_log_std}
    if a.policy_mode!='gaussian':
        from contact_hybrid_policy import ContractGatePolicy
        actor_class=ContractGatePolicy;policy_kwargs.update(gate_conditional=a.policy_mode=='hybrid_conditional',gate_prior=1.)
        (a.output/'hybrid_policy_source.py').write_bytes(Path(__file__).with_name('contact_hybrid_policy.py').read_bytes())
    vec=Adapter();policy=StudyPPO(actor_class,vec,seed=a.seed,device='cpu',n_steps=a.rollout_steps,
        batch_size=min(256,a.rollout_steps*env.num_envs),n_epochs=6,
        gamma=a.gamma,learning_rate=3e-4,ent_coef=0.,policy_kwargs=policy_kwargs,verbose=0)
    policy.credit_mode=a.credit_mode;policy.credit_mix=a.credit_mix
    if a.load_policy:policy=PPO.load(a.load_policy,env=vec,device='cpu')
    legacy=PPO.load(a.legacy_policy,device='cpu') if a.legacy_policy else None
    (a.output/'checkpoints').mkdir()
    updates_before=max((int(s['step']) for s in policy.policy.optimizer.state.values() if 'step' in s),default=0)
    before={k:v.clone() for k,v in policy.policy.state_dict().items()};start=time.perf_counter()
    policy.set_logger(configure(str(a.output/'ppo'),['csv']))
    def capture(label):
        return  # legacy GIF disabled; actual post-step camera recording is separate
    class Callback(BaseCallback):
        def _on_step(self):
            if hasattr(self.model,'outcome_flags'):self.model.outcome_flags.append(self.locals['dones'].copy())
            action_log.append({'mode':'train','physics_step':env.physics_steps,
                'raw':self.locals['actions'].tolist(),'clipped':self.locals['clipped_actions'].tolist(),
                'applied_xy':env.applied_delta.cpu().tolist(),'phase_before':env.phase_before_action.cpu().tolist()})
            if self.n_calls%64==0:capture('TRAINING')
            return True
    if a.controller in ['nominal','goal_feedback','damped','legacy','paired','random','fixed_push']:
        o=vec.reset()
        for _ in range(a.steps//a.num_envs):o,_,_,_=vec.step(np.zeros((a.num_envs,3),np.float32))
    elif a.steps:
        policy.learn(total_timesteps=a.steps,callback=Callback());policy.save(a.output/'policy.zip')
    train_steps=env.physics_steps;train_records=list(env.records);env.records=[];env.evaluation=True
    env.eval_cursors=torch.arange(env.num_envs,device=env.device)
    # Same evaluation seed/cases for every controller; no policy-based reset sampling.
    torch.manual_seed(a.eval_seed);o=vec.reset();iterations=0
    paired_video.initial(env)
    while len(env.records)<a.eval_episodes and iterations<env.max_episode_length*(int(np.ceil(a.eval_episodes/a.num_envs))+2):
        actor=legacy if a.controller in ['legacy','paired'] else policy
        if a.controller not in ['nominal','goal_feedback','damped','random','fixed_push']:
            with torch.no_grad():
                tensor=actor.policy.obs_to_tensor(o[:,:33] if a.controller in ['legacy','paired'] else o)[0]
                raw=actor.policy.get_distribution(tensor).get_actions(deterministic=not a.stochastic_eval).cpu().numpy()
            if raw.shape[1]==2:raw=np.column_stack((raw,np.ones(env.num_envs)))
        else:raw=np.zeros((env.num_envs,3),np.float32)
        actions=np.clip(raw,-1,1)
        o,_,_,_=vec.step(actions);iterations+=1
        action_log.append({'mode':'evaluation','physics_step':env.physics_steps,'raw':raw.tolist(),
            'clipped':actions.tolist(),'applied_xy':env.applied_delta.cpu().tolist(),'phase_before':env.phase_before_action.cpu().tolist()})
        if iterations%16==0:capture('EVALUATION')
    capture('FINAL_ACTUAL_STATE')
    result={'scope':'ARES SELF-BUILT CONTACT COMPOSITION; NOT public benchmark or field-aligned task',
        'isaac_sim':'5.1','isaac_lab':'2.3.0','controller':a.controller,'seed':a.seed,'num_envs':a.num_envs,'families':family,
        'physics_steps':env.physics_steps,'training_physics_steps':train_steps,'env_physics_steps':env.physics_steps*env.num_envs,
        'online_training_interactions':policy.num_timesteps if a.steps and a.controller in ['ppo','residual'] else 0,
        'optimizer_updates':max((int(s['step']) for s in policy.policy.optimizer.state.values() if 'step' in s),default=0)-updates_before,
        'parameters_changed':any(not torch.equal(before[k],v) for k,v in policy.policy.state_dict().items()),
        'credit_mode':a.credit_mode,'credit_mix':a.credit_mix,'gamma':a.gamma,'rollout_steps':a.rollout_steps,
        'action_intervention':a.action_intervention,
        'scene_specs':specs,'evaluation_case_seed':a.eval_seed,
        'evaluation_role':scenario.get('split','development'),
        'loaded_policy_sha256':hashlib.sha256(a.load_policy.read_bytes()).hexdigest() if a.load_policy else None,
        'training_episodes':len(train_records),'test_episodes':len(env.records),'test_successes':sum(r['success'] for r in env.records),
        'wall_s':time.perf_counter()-start,'reset_state_writes':env.write_count,'state_writes_during_actions':0,
        'asset_sha256':meta['asset_sha256'],'task_source_sha256':hashlib.sha256(source_snapshot).hexdigest(),
        'credit_source_sha256':hashlib.sha256(method_source).hexdigest(),
        'task_version':a.task_version,'switch_mode':a.switch_mode,'switch_dwell':a.switch_dwell,
        'forced_switch_signal':a.force_switch_signal,
        'rule_recovery':a.rule_recovery,
        'policy_mode':a.policy_mode,'action_cost':a.action_cost,
        'reference_lead_limit_m':None if a.task_version=='v5_replay' else .006,
        'stochastic_evaluation':a.stochastic_eval,'blind_next':a.blind_next,'blind_rotation':a.blind_rotation,'observation_dim':56,'action_dim':3,
        'actor_log_std':policy.policy.log_std.detach().cpu().tolist(),
        'legacy_log_std':legacy.policy.log_std.detach().cpu().tolist() if legacy else None,
        'success_contract':{'position_m':.008,'linear_speed_m_s':.025 if a.task_version=='v5_replay' else .01,
            'angular_speed_rad_s':None if a.task_version=='v5_replay' else .2,'yaw_error_rad':None if a.task_version=='v5_replay' else float(np.radians(8)),
            'hold_control_steps':8 if a.task_version=='v5_replay' else 30,'upright_rad':None if a.task_version=='v5_replay' else float(np.radians(5))},
        'limitations':['state-based actor assumes observed object pose/velocity','fixed research paddle; no contact grasp',
            'no field scene or measured tray CAD','prototype group split not a formal generalization claim']}
    def json_default(value):
        if torch.is_tensor(value):return value.detach().cpu().tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2,default=json_default))
    (a.output/'training_episodes.json').write_text(json.dumps(train_records,indent=2))
    (a.output/'credit_audits.json').write_text(json.dumps(getattr(policy,'credit_audits',[]),indent=2))
    (a.output/'frozen_evaluation.json').write_text(json.dumps(env.records,indent=2))
    (a.output/'action_log.json').write_text(json.dumps(action_log))
    np.savez_compressed(a.output/'actual_physics_trace.npz',states=np.array(env.trace),sample_dt_s=env.physics_dt,
        goals_body_m=env.goals.cpu().numpy(),columns=np.array([*[f'q{i}' for i in range(6)],'obj_x','obj_y','obj_z',
        'paddle_x','paddle_y','paddle_z','vx','vy','vz','stage','obj_qw','obj_qx','obj_qy','obj_qz','case_id',
            'cmd_x','cmd_y','cmd_z','goal_x','goal_y','goal_z','action_x','action_y','applied_dx','applied_dy',
            'wx','wy','wz',*[f'qd{i}' for i in range(6)],'physics_tick','switch_signal','raw_x','raw_y','raw_switch']))
    if frames:
        from PIL import Image
        Image.fromarray(frames[-1]).save(a.output/'physical_scene.png')
        images=[Image.fromarray(f).resize((640,480)).quantize(colors=128) for f in frames]
        durations=[max(10,round((frame_records[min(i+1,len(frame_records)-1)]['physics_steps']-r['physics_steps'])*env.physics_dt*1000))
                   if i<len(frame_records)-1 else 267 for i,r in enumerate(frame_records)]
        images[0].save(a.output/'actual_physics.gif',save_all=True,append_images=images[1:],duration=durations,loop=0)
        (a.output/'video_frame_records.json').write_text(json.dumps(frame_records))
    paired_video.finalize(env,a.output)
    print(json.dumps(result,default=json_default),flush=True);env.close()
