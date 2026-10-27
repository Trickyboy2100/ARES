"""State-based dynamic workpiece pushing; new ARES tasks, NOT public benchmarks.

Uses the deployed DirectRLEnv/InteractiveScene and the previously audited JAKA
Jacobian controller. A workpiece is free, not attached or repositioned by an
action. Only reset writes state. No field point clouds/configurations are used.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np


def run(a):
    source_snapshot=Path(__file__).read_bytes();(a.output/'task_source.py').write_bytes(source_snapshot)
    method_source=Path(__file__).with_name('contact_credit_v2.py').read_bytes()
    (a.output/'boundary_credit_source.py').write_bytes(method_source)
    frame_source=Path(__file__).with_name('contact_task_frame_v7.py').read_bytes()
    (a.output/'frame_source.py').write_bytes(frame_source)
    geometry_source=Path(__file__).with_name('contact_geometry_v8.py').read_bytes()
    (a.output/'geometry_source.py').write_bytes(geometry_source)
    if a.recontact_enabled:
        for name in ['contact_masked_policy_v8.py','recontact_waypoint_v8.py','recontact_minkowski_v8.py','audit_contact_actor_gradient_v8.py']:
            (a.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (a.output/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2))
    if a.scenario_file:(a.output/'scene.json').write_bytes(a.scenario_file.read_bytes())
    import torch,gymnasium as gym
    from contact_task_frame_v7 import to_task_xy,from_task_xy,rear_alignment,task_observation
    from contact_geometry_v8 import contact_ray_clearance,projected_halfwidth
    from recontact_waypoint_v8 import recontact_waypoint
    from recontact_minkowski_v8 import recontact_waypoint_exact
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
    action_dim=5 if a.recontact_enabled else (4 if a.yaw_enabled else 3)
    observation_dim=87 if a.recontact_enabled else (66 if a.yaw_enabled else 64)

    simulation_config=sim.SimulationCfg(dt=1/120,render_interval=8,device=f'cuda:{a.gpu}')
    @configclass
    class Cfg(DirectRLEnvCfg):
        decimation=2;episode_length_s=a.episode_seconds;action_space=action_dim;observation_space=observation_dim;state_space=0
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
            self.effect=torch.zeros((self.num_envs,8),device=self.device)
            self.prior_obj=torch.zeros((self.num_envs,3),device=self.device)
            self.parameter=torch.zeros(self.num_envs,device=self.device)
            self.yaw_command=torch.zeros_like(self.parameter)
            self.held_yaw=torch.zeros_like(self.parameter);self.held_angle=torch.zeros_like(self.parameter)
            self.remaining=torch.zeros_like(self.phase);self.recontact_mode=torch.full_like(self.phase,-1)
            self.recontact_attempts=torch.zeros_like(self.phase);self.recontact_cooldown=torch.zeros_like(self.phase)
            self.push_direction=torch.zeros((self.num_envs,2),device=self.device);self.push_direction[:,0]=1
            self.prior_wz=torch.zeros_like(self.parameter);self.prior_vt=torch.zeros_like(self.parameter)
            self.prior_contact=torch.zeros_like(self.parameter);self.refresh_event=torch.ones_like(self.phase,dtype=torch.bool)
            self.execution_mask=torch.ones((self.num_envs,action_dim),device=self.device)
            self.min_final=torch.full((self.num_envs,),float('inf'),device=self.device)
            self.withdraw_left=torch.zeros_like(self.phase);self.withdraw_attempts=torch.zeros_like(self.phase);self.cooldown=torch.zeros_like(self.phase)
            self.start_object_xy=torch.zeros((self.num_envs,2),device=self.device);self.object_displacement=torch.zeros(self.num_envs,device=self.device)
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
                self.camera=Camera(CameraCfg(prim_path='/World/Preview',height=720,width=960,data_types=['rgb'],spawn=sim.PinholeCameraCfg(clipping_range=(.01,100))))
        def state(self):
            origin=self.scene.env_origins
            paddle=self.robot.data.body_link_pos_w[:,self.link]+quat_apply(self.robot.data.body_link_quat_w[:,self.link],self.offset)-origin
            obj=self.cube.data.root_pos_w-origin;vel=self.cube.data.root_lin_vel_w
            goal=self.goals[torch.arange(self.num_envs,device=self.device),self.phase]
            return paddle,obj,vel,goal
        def action_mask(self):
            mask=torch.ones((self.num_envs,action_dim),device=self.device)
            if a.recontact_enabled:
                normal=self.recontact_mode<0
                refresh=(self.remaining<=0)|self.refresh_event
                mask[:,:2]=normal[:,None]
                mask[:,2]=(normal&refresh&(self.phase==1))
                mask[:,3]=(normal&refresh)
                mask[:,4]=(normal&refresh&(self.phase==0)&(self.recontact_attempts>0))
            return mask
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
            # B1 BODY state vs B2/B3 task frame; same information and 64D capacity.
            if a.frame=='task':
                old,extra=task_observation(old,extra,self.geometry[:,5:7])
                extra[:,18:20]=self.geometry[:,5:7] # retain actual WORLD direction; fixed BODY tool/workspace matters
            effect=self.effect.clone()
            if getattr(a,'mask_measured_effect',False):effect[:,[0,1,4,5,6]]=0
            if a.blind_outcome:effect.zero_()
            if a.blind_next:
                old=old.clone();extra=extra.clone();mask=self.phase==0
                old[mask,24:27]=0;extra[mask,13:16]=0;extra[mask,18:20]=0
            observation=torch.cat((old,extra,effect),-1)
            if a.yaw_enabled:
                qrel=quat_mul(self.robot.data.body_link_quat_w[:,self.link],quat_conjugate(self.orientation))
                tool_yaw=torch.atan2(2*(qrel[:,0]*qrel[:,3]+qrel[:,1]*qrel[:,2]),1-2*(qrel[:,2]**2+qrel[:,3]**2))
                observation=torch.cat((observation,torch.stack((tool_yaw,self.yaw_command),-1)),-1)
            if a.recontact_enabled:
                controller_state=torch.cat((torch.stack((self.parameter/a.contact_limit,self.held_yaw/a.yaw_command_limit,
                    self.held_angle/a.normal_angle_limit,self.remaining/a.parameter_hold,self.recontact_mode/3,
                    self.recontact_attempts/3,self.recontact_cooldown/120,self.prior_wz,self.prior_vt,
                    self.prior_contact,self.refresh_event.float()),-1),self.push_direction,
                    (self.start_object_xy-self.center[:2])*10,self.object_displacement[:,None]*10),-1)
                observation=torch.cat((observation,controller_state,self.action_mask()),-1)
            return {'policy':observation}
        def physical_snapshot(self,i):
            paddle,obj,vel,goal=self.state()
            return {'physics_step':self.physics_steps,'control_step':int(self.episode_length_buf[i]),
                'time_s':self.physics_steps*self.physics_dt,'case_id':int(self.cases[i]),'phase':int(self.phase[i]),
                'episode_time_s':int(self.episode_length_buf[i])*self.step_dt,
                'q':self.robot.data.joint_pos[i,self.ids].cpu().tolist(),'qd':self.robot.data.joint_vel[i,self.ids].cpu().tolist(),
                'object_position':obj[i].cpu().tolist(),'object_quaternion_wxyz':self.cube.data.root_quat_w[i].cpu().tolist(),
                'object_linear_velocity':vel[i].cpu().tolist(),'object_angular_velocity':self.cube.data.root_ang_vel_w[i].cpu().tolist(),
                'paddle_position':paddle[i].cpu().tolist(),'command':self.command[i].cpu().tolist(),
                'joint_velocity_command':self.vcommand[i].cpu().tolist(),'goal':goal[i].cpu().tolist(),
                'robot_contact_force':self.contact.data.net_forces_w[i].cpu().tolist(),
                'actual_link6_quaternion_wxyz':self.robot.data.body_link_quat_w[i,self.link].cpu().tolist(),'tool_yaw_command_rad':float(self.yaw_command[i]),
                'recontact_mode':int(self.recontact_mode[i]),'recontact_attempts':int(self.recontact_attempts[i]),
                'held_contact_m':float(self.parameter[i]),'held_yaw_rad':float(self.held_yaw[i]),'held_normal_angle_rad':float(self.held_angle[i]),
                'parameter_hold_remaining':int(self.remaining[i]),'push_direction':self.push_direction[i].cpu().tolist()}
        def _pre_physics_step(self,actions):
            if actions.shape[1]==2:actions=torch.cat((actions,torch.ones_like(actions[:,:1])),-1)
            self.raw_actions=actions.clone();self.actions=actions[:,:2].clamp(-1,1)
            self.switch_signal=actions[:,2];paddle,obj,vel,goal=self.state()
            if a.force_switch_signal is not None:self.switch_signal=torch.full_like(self.switch_signal,a.force_switch_signal)
            if a.rule_recovery:self.switch_signal=torch.where((goal[:,:2]-obj[:,:2]).norm(dim=-1)<.007,-1.,1.)
            self.phase_before_action=self.phase.clone()
            self.case_before_action=self.cases.clone()
            self.execution_mask=self.action_mask().clone()
            self.hold_before_action=self.remaining.clone();self.mode_before_action=self.recontact_mode.clone()
            self.held_before_action=torch.stack((self.parameter,self.held_yaw,self.held_angle),-1).clone()
            if a.recontact_enabled:
                refresh=self.execution_mask[:,3]>0
                self.remaining=torch.where(refresh,torch.full_like(self.remaining,a.parameter_hold),self.remaining)
                self.held_yaw=torch.where(refresh,actions[:,3].clamp(-1,1)*a.yaw_command_limit*a.yaw_residual_scale,self.held_yaw)
                self.held_angle=torch.where(self.execution_mask[:,4]>0,actions[:,4].clamp(-1,1)*a.normal_angle_limit,self.held_angle)
                self.refresh_event.zero_()
            desired_yaw=torch.zeros_like(self.yaw_command)
            if a.yaw_enabled:
                phi=torch.atan2(self.geometry[:,6],self.geometry[:,5])
                cube_quat=self.cube.data.root_quat_w
                cube_yaw=torch.atan2(2*(cube_quat[:,0]*cube_quat[:,3]+cube_quat[:,1]*cube_quat[:,2]),1-2*(cube_quat[:,2]**2+cube_quat[:,3]**2))
                err=torch.atan2(torch.sin(4*(cube_yaw-phi)),torch.cos(4*(cube_yaw-phi)))/4
                reference=phi if a.yaw_reference=='slot' else (phi+.5*err if a.yaw_reference=='midface' else torch.zeros_like(phi))
                desired_yaw=reference+actions[:,3].clamp(-1,1)*a.yaw_command_limit*a.yaw_residual_scale
                if a.recontact_enabled:desired_yaw=reference+self.held_yaw
                if a.fixed_tool_yaw is not None:desired_yaw=torch.full_like(desired_yaw,a.fixed_tool_yaw)
                desired_yaw=desired_yaw.clamp(-a.yaw_command_limit,a.yaw_command_limit)
            self.yaw_command+=(desired_yaw-self.yaw_command).clamp(-.08*self.step_dt,.08*self.step_dt)
            direction=goal[:,:2]-obj[:,:2];direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(.001)
            if a.recontact_enabled:
                c,s=self.held_angle.cos(),self.held_angle.sin()
                rotated=torch.stack((c*direction[:,0]-s*direction[:,1],s*direction[:,0]+c*direction[:,1]),-1)
                recovery_control=(self.phase==0)&(self.recontact_attempts>0)
                normal_direction=torch.where(recovery_control[:,None],rotated,self.geometry[:,5:7])
                self.push_direction=torch.where((self.recontact_mode<0)[:,None],normal_direction,self.push_direction)
                direction=self.push_direction
            # Ray intersection with the rectangle Minkowski sum, NOT its
            # support function (which leaves a spurious gap at diagonal push).
            envelope=torch.tensor([.02+meta['paddle_dims_m'][0]/2,.02+meta['paddle_dims_m'][1]/2],device=self.device)
            clearance=(envelope/direction.abs().clamp_min(1e-6)).amin(-1,keepdim=True)
            cube_quat=self.cube.data.root_quat_w
            cube_yaw_geometry=torch.atan2(2*(cube_quat[:,0]*cube_quat[:,3]+cube_quat[:,1]*cube_quat[:,2]),1-2*(cube_quat[:,2]**2+cube_quat[:,3]**2))
            tool_rel=quat_mul(self.robot.data.body_link_quat_w[:,self.link],quat_conjugate(self.orientation))
            tool_yaw_geometry=torch.atan2(2*(tool_rel[:,0]*tool_rel[:,3]+tool_rel[:,1]*tool_rel[:,2]),1-2*(tool_rel[:,2]**2+tool_rel[:,3]**2))
            if a.exact_footprint:clearance=contact_ray_clearance(direction,cube_yaw_geometry,tool_yaw_geometry,[.02,.02],np.array(meta['paddle_dims_m'][:2])/2)
            nominal=obj[:,:2]-direction*(clearance-.0025)
            nominal_delta=(nominal-self.command[:,:2]).clamp(-.0012,.0012)
            # Establish a real rear-face contact before pushing; do not drive
            # diagonally into a cube corner while still approaching from free space.
            forward=self.geometry[:,5:7];lateral=torch.stack((-forward[:,1],forward[:,0]),-1)
            # Rear-face alignment in task coordinates for EVERY baseline.
            align,align_delta=rear_alignment(paddle[:,:2],obj[:,:2],self.command[:,:2],forward)
            if a.recontact_enabled:
                # Preserve the validated initial rear-face approach. Only a
                # genuinely different push normal may bypass that alignment.
                same_face=(self.push_direction*self.geometry[:,5:7]).sum(-1)>np.cos(np.radians(35))
                align=align&((self.phase==1)|same_face)
            nominal_delta=torch.where(align[:,None],align_delta,nominal_delta)
            nominal_delta=torch.where(((goal[:,:2]-obj[:,:2]).norm(dim=-1)<.006)[:,None],torch.zeros_like(nominal_delta),nominal_delta)
            forward=self.geometry[:,5:7]
            if a.recontact_enabled:forward=torch.where((self.phase==0)[:,None],self.push_direction,forward)
            lateral=torch.stack((-forward[:,1],forward[:,0]),-1)
            forward_clearance=(envelope/forward.abs().clamp_min(1e-6)).amin(-1,keepdim=True)
            if a.exact_footprint:forward_clearance=contact_ray_clearance(forward,cube_yaw_geometry,tool_yaw_geometry,[.02,.02],np.array(meta['paddle_dims_m'][:2])/2)
            endpoint=goal[:,:2]-forward*forward_clearance
            damped_delta=((endpoint-self.command[:,:2])*a.feedback_kp-vel[:,:2]*a.feedback_kd)*self.step_dt
            damped_delta=damped_delta.clamp(-.0012,.0012)
            # B3 third action selects bounded lateral contact parameter; actual
            # object/robot state is never rewritten. The subsequent stage
            # begins from continuous real physics, not a predicted terminal.
            active=(self.phase==1) if a.contact_phase=='phase1' else ((self.phase==0) if a.contact_phase=='phase0' else torch.ones_like(self.phase,dtype=torch.bool))
            quat=self.cube.data.root_quat_w
            cube_yaw=torch.atan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2))
            phi=torch.atan2(self.geometry[:,6],self.geometry[:,5])
            yaw_error=torch.atan2(torch.sin(4*(cube_yaw-phi)),torch.cos(4*(cube_yaw-phi)))/4
            nominal_parameter=(a.contact_steering*yaw_error).clamp(-a.contact_limit,a.contact_limit)
            target_parameter=torch.where(active,(nominal_parameter+self.switch_signal.clamp(-1,1)*a.contact_limit).clamp(-a.contact_limit,a.contact_limit),torch.zeros_like(self.parameter))
            if a.fixed_contact is not None:target_parameter=torch.where(active,torch.full_like(target_parameter,a.fixed_contact),torch.zeros_like(target_parameter))
            if a.disable_contact:target_parameter=torch.where(active,nominal_parameter,torch.zeros_like(target_parameter))
            # All learning baselines have identical legal parameter capacity.
            # B2 differs only by withholding measured execution-effect history.
            if a.recontact_enabled:
                self.parameter=torch.where(self.execution_mask[:,2]>0,target_parameter,self.parameter)
                self.parameter=torch.where(self.phase==0,torch.zeros_like(self.parameter),self.parameter)
            else:self.parameter=target_parameter
            if a.contact_following:
                lateral_goal=((goal[:,:2]-obj[:,:2])*lateral).sum(-1,keepdim=True)
                endpoint=endpoint+lateral*(-lateral_goal+lateral_goal.clamp(-.006,.006))
            parameter_target=endpoint+lateral*self.parameter[:,None]
            if a.exact_footprint:
                lateral_offset=((parameter_target-obj[:,:2])*lateral).sum(-1)
                distance=contact_ray_clearance(forward,cube_yaw_geometry,tool_yaw_geometry,[.02,.02],np.array(meta['paddle_dims_m'][:2])/2,lateral_offset)
                forward_goal=((goal[:,:2]-obj[:,:2])*forward).sum(-1,keepdim=True)
                if a.guarded_alignment:
                    width=2*projected_halfwidth(cube_yaw_geometry,lateral,[.02,.02])
                    blocked=(self.phase==1)&((width>self.geometry[:,0]-.002)|(yaw_error.abs()>np.radians(8)))
                    forward_goal=torch.where(blocked[:,None],torch.full_like(forward_goal,.002),forward_goal)
                parameter_target=obj[:,:2]+forward*(forward_goal-distance)+lateral*lateral_offset[:,None]
            damped_delta=((parameter_target-self.command[:,:2])*a.feedback_kp-vel[:,:2]*a.feedback_kd)*self.step_dt
            damped_delta=damped_delta.clamp(-.0012,.0012)
            damped_delta=torch.where(align[:,None],align_delta,damped_delta)
            delta=self.actions*.0012
            if a.controller=='nominal':delta=nominal_delta
            elif a.controller=='goal_feedback':
                # Strong non-learning position-feedback comparator: target the
                # final rear-face contact, rather than maintaining indentation
                # behind a moving cube. Never fitted to heldout outcomes.
                endpoint=goal[:,:2]-direction*clearance
                delta=(endpoint-self.command[:,:2]).clamp(-.0012,.0012)
                delta=torch.where(align[:,None],align_delta,delta)
            elif a.controller=='fixed_push':
                delta=torch.zeros_like(delta);delta[:,0]=.0012
            elif a.controller=='residual':
                residual=self.actions
                if a.frame=='task':residual=from_task_xy(residual,forward)
                delta=(damped_delta+residual*.0006).clamp(-.0012,.0012)
            elif a.controller=='random':delta=(torch.rand_like(delta)*2-1)*.0012
            if a.controller=='damped':
                # Endpoint contact controller with real workpiece velocity damping.
                delta=damped_delta
            if a.controller=='paired':delta=torch.where((self.phase==1)[:,None],damped_delta,delta)
            self.next_recontact_mode=self.recontact_mode.clone()
            if a.recontact_enabled:
                contact_distance=contact_ray_clearance(self.push_direction,cube_yaw_geometry,tool_yaw_geometry,[.02,.02],np.array(meta['paddle_dims_m'][:2])/2).squeeze(-1)+.001
                if a.exact_recontact:
                    target,next_mode,ready=recontact_waypoint_exact(paddle[:,:2],obj[:,:2],self.push_direction,
                        self.recontact_mode.clamp_min(0),contact_distance,cube_yaw_geometry,tool_yaw_geometry,
                        tool_half=np.array(meta['paddle_dims_m'][:2])/2)
                else:
                    target,next_mode,ready=recontact_waypoint(paddle[:,:2],obj[:,:2],self.push_direction,
                        self.recontact_mode.clamp_min(0),contact_distance,tool_half=np.array(meta['paddle_dims_m'][:2])/2)
                primitive=self.recontact_mode>=0
                delta=torch.where(primitive[:,None],((target-self.command[:,:2])*a.feedback_kp*self.step_dt).clamp(-.0012,.0012),delta)
                self.next_recontact_mode=torch.where(primitive,torch.where(ready,torch.full_like(next_mode,-1),next_mode),self.recontact_mode)
            if a.switch_dwell>0:
                eligible=(self.phase==0)&((goal[:,:2]-obj[:,:2]).norm(dim=-1)<.009)&(vel.norm(dim=-1)<.12)
                delta=torch.where(eligible[:,None],torch.zeros_like(delta),delta)
            if a.switch_mode=='recover':
                wait=(self.phase==1)&(self.switch_signal<0)
                delta=torch.where(wait[:,None],torch.zeros_like(delta),delta)
            if a.withdraw_rule:
                force=self.contact.data.net_forces_w.norm(dim=-1).amax(-1)
                trigger=(self.phase==1)&((force>20)|(self.cube.data.root_ang_vel_w.norm(dim=-1)>.5))&(self.cooldown==0)&(self.withdraw_attempts<3)
                self.withdraw_left=torch.where(trigger,torch.full_like(self.phase,16),self.withdraw_left)
                self.withdraw_attempts+=trigger.long();self.cooldown=torch.where(trigger,torch.full_like(self.phase,60),self.cooldown)
                delta=torch.where((self.withdraw_left>0)[:,None],-forward*.0006,delta)
                self.withdraw_left=(self.withdraw_left-1).clamp_min(0);self.cooldown=(self.cooldown-1).clamp_min(0)
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
            self.applied_held_before_physics=torch.stack((self.parameter,self.held_yaw,self.held_angle),-1).clone()
        def _apply_action(self):
            paddle,obj,vel,goal=self.state();quat=self.robot.data.body_link_quat_w[:,self.link]
            error=torch.cat(((self.command-paddle).clamp(-.008,.008),
                axis_angle_from_quat(quat_mul(quat_mul(torch.stack((torch.cos(self.yaw_command/2),torch.zeros_like(self.yaw_command),torch.zeros_like(self.yaw_command),torch.sin(self.yaw_command/2)),-1),self.orientation),quat_conjugate(quat))).clamp(-.03,.03)),-1)
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
                    self.switch_signal[:,None],self.raw_actions[:,:3],self.yaw_command[:,None],self.robot.data.body_link_quat_w[:,self.link],
                    self.recontact_mode[:,None],self.remaining[:,None],self.parameter[:,None],self.held_yaw[:,None],self.held_angle[:,None],self.push_direction),dim=-1).cpu().numpy())
            self.physics_steps+=1
        def _get_dones(self):
            paddle,obj,vel,goal=self.state();q=self.robot.data.joint_pos[:,self.ids]
            dist=(goal[:,:2]-obj[:,:2]).norm(dim=-1)
            self.min_final=torch.minimum(self.min_final,(self.goals[:,1,:2]-obj[:,:2]).norm(dim=-1))
            self.object_displacement=torch.maximum(self.object_displacement,(obj[:,:2]-self.start_object_xy).norm(dim=-1))
            displacement=(obj-self.prior_obj)/self.step_dt
            forward=self.geometry[:,5:7];lateral=torch.stack((-forward[:,1],forward[:,0]),-1)
            effect_xy=torch.stack(((displacement[:,:2]*forward).sum(-1),(displacement[:,:2]*lateral).sum(-1)),-1)
            command_xy=torch.stack(((self.applied_delta*forward).sum(-1),(self.applied_delta*lateral).sum(-1)),-1)/self.step_dt
            self.effect=torch.cat((effect_xy,command_xy,effect_xy-command_xy,
                self.cube.data.root_ang_vel_w[:,2:3],self.parameter[:,None]*250),-1).clamp(-5,5)
            self.prior_obj=obj.clone()
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
            if a.log_contact_history and self.evaluation:
                self.contacts.append(torch.cat((torch.full((self.num_envs,1),self.physics_steps,device=self.device),
                    self.cases[:,None],self.phase[:,None],force[:,None],self.contact.data.net_forces_w.sum(dim=1),
                    vel,self.cube.data.root_ang_vel_w,obj),dim=-1).cpu().numpy())
            tilt=axis_angle_from_quat(quat_mul(self.orientation,quat_conjugate(self.robot.data.body_link_quat_w[:,self.link]))).norm(dim=-1)
            invalid=(q<self.low-.001).any(-1)|(q>self.high+.001).any(-1)|(~torch.isfinite(q)).any(-1)
            qrel=quat_mul(self.robot.data.body_link_quat_w[:,self.link],quat_conjugate(self.orientation))
            actual_yaw=torch.atan2(2*(qrel[:,0]*qrel[:,3]+qrel[:,1]*qrel[:,2]),1-2*(qrel[:,2]**2+qrel[:,3]**2))
            horizontal_tilt=torch.acos((1-2*(qrel[:,1]**2+qrel[:,2]**2)).clamp(-1,1))
            tool_invalid=((horizontal_tilt>np.radians(3.5))|(actual_yaw.abs()>a.yaw_limit+.001)) if a.yaw_enabled else (tilt>np.radians(3.5))
            failed=invalid|(obj[:,2]<self.center[2]-.08)|tool_invalid|(force>80.)
            timeout=self.episode_length_buf>=self.max_episode_length-1
            charged=self.actions if a.action_cost=='raw' else self.applied_delta/.0012
            self.reward=60*(self.previous-dist)-self.step_dt-.0002*charged.square().sum(-1)
            self.reward+=3*advance.float()+10*(self.success&~failed).float()-5*failed.float()
            self.phase=torch.where(advance,torch.ones_like(self.phase),self.phase)
            if a.recontact_enabled:
                # All mask events are derived from already observed physics and
                # become the NEXT observation; no action-dependent likelihood mask.
                self.remaining=(self.remaining-1).clamp_min(0)
                self.recontact_cooldown=(self.recontact_cooldown-1).clamp_min(0)
                was_primitive=self.recontact_mode>=0
                self.recontact_mode=self.next_recontact_mode
                # A physical phase handoff always returns to the slot's rear
                # controller; never carry a phase0 orbit into the rail stage.
                self.recontact_mode=torch.where(advance,torch.full_like(self.phase,-1),self.recontact_mode)
                finished=was_primitive&(self.recontact_mode<0)
                self.recontact_cooldown=torch.where(finished,torch.full_like(self.phase,120),self.recontact_cooldown)
                offset=paddle[:,:2]-obj[:,:2]
                recovery_direction=goal[:,:2]-obj[:,:2]
                recovery_direction=recovery_direction/recovery_direction.norm(dim=-1,keepdim=True).clamp_min(.001)
                c,s=self.held_angle.cos(),self.held_angle.sin()
                recovery_direction=torch.stack((c*recovery_direction[:,0]-s*recovery_direction[:,1],s*recovery_direction[:,0]+c*recovery_direction[:,1]),-1)
                face_cos=(offset/offset.norm(dim=-1,keepdim=True).clamp_min(.001)*(-recovery_direction)).sum(-1)
                trigger=(self.phase==0)&(dist>.010)&(self.object_displacement>.020)&(face_cos<np.cos(np.radians(35)))&\
                    (self.episode_length_buf>20)&(self.recontact_mode<0)&(self.recontact_cooldown==0)&(self.recontact_attempts<2)
                self.recontact_mode=torch.where(trigger,torch.zeros_like(self.phase),self.recontact_mode)
                self.push_direction=torch.where(trigger[:,None],recovery_direction,self.push_direction)
                self.recontact_attempts+=trigger.long()
                self.remaining=torch.where(advance|finished,torch.zeros_like(self.remaining),self.remaining)
                vt=(vel[:,:2]*lateral).sum(-1);wz=self.cube.data.root_ang_vel_w[:,2]
                contact=(force>1).float()
                event=((contact!=self.prior_contact)|((wz-self.prior_wz).abs()>.25)|((vt-self.prior_vt).abs()>.015))&(self.remaining<=a.parameter_hold-8)
                self.refresh_event=(event if a.event_refresh else torch.zeros_like(event))|advance|finished
                self.prior_wz=wz.clone();self.prior_vt=vt.clone();self.prior_contact=contact.clone()
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
                    withdraw_attempts=int(self.withdraw_attempts[i]),
                    recontact_attempts=int(self.recontact_attempts[i]),
                    maximum_object_xy_displacement_m=float(self.object_displacement[i]),
                    failure_reason='constraint' if failed[i] else ('success' if self.success[i] else 'timeout'))
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
            noise=(torch.rand((n,2),device=self.device)-.5)*a.initial_noise
            if self.evaluation:
                for k,i in enumerate(ids.tolist()):
                    index=int(self.eval_cursors[i]);self.eval_cursors[i]+=self.num_envs
                    self.cases[i]=index if index<a.eval_episodes else -1
                    noise[k]=torch.tensor(np.random.default_rng(a.eval_seed+index).uniform(-a.initial_noise/2,a.initial_noise/2,2),device=self.device)
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
            self.start_object_xy[ids]=obj[:,:2]-self.scene.env_origins[ids,:2];self.object_displacement[ids]=0
            self.prior_obj[ids]=obj[:,:3]-self.scene.env_origins[ids];self.effect[ids]=0;self.parameter[ids]=0
            self.command[ids]=self.center;self.qcommand[ids]=self.home;self.vcommand[ids]=0
            self.phase[ids]=0;self.hold[ids]=0;self.returns[ids]=0;self.max_force[ids]=0
            self.dwell[ids]=0;self.yaw_command[ids]=0
            self.held_yaw[ids]=0;self.held_angle[ids]=0;self.remaining[ids]=0;self.recontact_mode[ids]=-1
            self.recontact_attempts[ids]=0;self.recontact_cooldown[ids]=0;self.push_direction[ids]=self.geometry[ids,5:7]
            self.prior_wz[ids]=0;self.prior_vt[ids]=0;self.prior_contact[ids]=0;self.refresh_event[ids]=True
            self.min_final[ids]=float('inf')
            self.withdraw_left[ids]=0;self.withdraw_attempts[ids]=0;self.cooldown[ids]=0
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
    class Adapter(VecEnv):
        def __init__(self):super().__init__(env.num_envs,gym.spaces.Box(-np.inf,np.inf,(observation_dim,),np.float32),gym.spaces.Box(-1,1,(action_dim,),np.float32))
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
    actor_gradient_audits=[]
    class StudyPPO(cls):
        def train(self):
            if a.recontact_enabled:
                from audit_contact_actor_gradient_v8 import audit_first_rollout
                audit_first_rollout(self,a.output)
            batch={};rows=[];std_rows=[]
            original=self.policy.evaluate_actions
            def evaluated(obs,actions):
                if a.recontact_enabled:batch['mask']=obs[:,-action_dim:].detach()
                return original(obs,actions)
            def weight_hook(gradient):
                inactive=(batch['mask'].sum(0)==0).cpu().tolist() if 'mask' in batch else [False]*action_dim
                rows.append({'all_inactive_in_minibatch':inactive,'row_l2':gradient.detach().norm(dim=1).cpu().tolist()})
            def std_hook(gradient):std_rows.append(gradient.detach().cpu().tolist())
            self.policy.evaluate_actions=evaluated
            h1=self.policy.action_net.weight.register_hook(weight_hook);h2=self.policy.log_std.register_hook(std_hook)
            try:super().train()
            finally:h1.remove();h2.remove();self.policy.evaluate_actions=original
            masks=self.rollout_buffer.observations[...,-action_dim:] if a.recontact_enabled else None
            actor_gradient_audits.append({'online_interactions':self.num_timesteps,'minibatch_actor_gradients':rows,
                'minibatch_log_std_gradients':std_rows,
                'pre_action_active_fraction':None if masks is None else masks.reshape(-1,action_dim).mean(0).tolist(),
                'rollout_samples':int(self.rollout_buffer.buffer_size*self.rollout_buffer.n_envs)})
            if self.num_timesteps%a.checkpoint_every==0:
                self.save(a.output/'checkpoints'/f'policy_{self.num_timesteps}_after_update.zip')
    actor_class='MlpPolicy';policy_kwargs={'net_arch':[64,64],'log_std_init':a.initial_log_std}
    if a.policy_mode=='qualified_gaussian':
        from contact_masked_policy_v8 import ExecutionActivePolicy
        actor_class=ExecutionActivePolicy
    if a.policy_mode.startswith('hybrid'):
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
        if a.skip_capture:return
        env.sim.render();env.camera.update(env.physics_dt,force_recompute=True)
        frames.append(env.camera.data.output['rgb'][0].cpu().numpy().copy())
        frame_records.append({'phase':label,'physics_steps':env.physics_steps,'q':env.robot.data.joint_pos[:,env.ids].cpu().tolist(),
            'object_positions':env.cube.data.root_pos_w.cpu().tolist()})
    class Callback(BaseCallback):
        def _on_step(self):
            if hasattr(self.model,'outcome_flags'):self.model.outcome_flags.append(self.locals['dones'].copy())
            action_log.append({'mode':'train','contact_parameter_m':env.parameter.cpu().tolist(),'effect_history':env.effect.cpu().tolist(),'physics_step':env.physics_steps,
                'raw':self.locals['actions'].tolist(),'clipped':self.locals['clipped_actions'].tolist(),
                'applied_xy':env.applied_delta.cpu().tolist(),'phase_before':env.phase_before_action.cpu().tolist(),
                'case_before_action':env.case_before_action.cpu().tolist(),
                'pre_action_execution_mask':env.execution_mask.cpu().tolist(),'pre_action_hold_remaining':env.hold_before_action.cpu().tolist(),
                'pre_action_mode':env.mode_before_action.cpu().tolist(),'pre_action_held':env.held_before_action.cpu().tolist(),
                'executed_held':env.applied_held_before_physics.cpu().tolist()})
            action_log[-1]['pre_action_observation']=self.model._last_obs.tolist()
            if self.n_calls%64==0:capture('TRAINING')
            return True
    if a.controller in ['nominal','goal_feedback','damped','legacy','paired','random','fixed_push']:
        o=vec.reset()
        for _ in range(a.steps//a.num_envs):o,_,_,_=vec.step(np.zeros((a.num_envs,action_dim),np.float32))
    elif a.steps:
        policy.learn(total_timesteps=a.steps,callback=Callback());policy.save(a.output/'policy.zip')
    train_steps=env.physics_steps;train_records=list(env.records);env.records=[];env.evaluation=True
    env.eval_cursors=torch.arange(env.num_envs,device=env.device)
    # Same evaluation seed/cases for every controller; no policy-based reset sampling.
    torch.manual_seed(a.eval_seed);o=vec.reset();iterations=0
    while len(env.records)<a.eval_episodes and iterations<env.max_episode_length*(int(np.ceil(a.eval_episodes/a.num_envs))+2):
        actor=legacy if a.controller in ['legacy','paired'] else policy
        if a.controller not in ['nominal','goal_feedback','damped','random','fixed_push']:
            with torch.no_grad():
                tensor=actor.policy.obs_to_tensor(o[:,:33] if a.controller in ['legacy','paired'] else o)[0]
                raw=actor.policy.get_distribution(tensor).get_actions(deterministic=not a.stochastic_eval).cpu().numpy()
            if raw.shape[1]==2 and action_dim==3:raw=np.column_stack((raw,np.ones(env.num_envs)))
        else:raw=np.zeros((env.num_envs,action_dim),np.float32)
        actions=np.clip(raw,-1,1)
        o,_,_,_=vec.step(actions);iterations+=1
        action_log.append({'mode':'evaluation','contact_parameter_m':env.parameter.cpu().tolist(),'effect_history':env.effect.cpu().tolist(),'physics_step':env.physics_steps,'raw':raw.tolist(),
            'clipped':actions.tolist(),'applied_xy':env.applied_delta.cpu().tolist(),'phase_before':env.phase_before_action.cpu().tolist(),
            'case_before_action':env.case_before_action.cpu().tolist(),
            'pre_action_execution_mask':env.execution_mask.cpu().tolist(),'pre_action_hold_remaining':env.hold_before_action.cpu().tolist(),
            'pre_action_mode':env.mode_before_action.cpu().tolist(),'pre_action_held':env.held_before_action.cpu().tolist(),
            'executed_held':env.applied_held_before_physics.cpu().tolist()})
        if iterations%16==0:capture('EVALUATION')
    capture('FINAL_ACTUAL_STATE')
    result={'engine_version':'v8_control_authority','physics_dt_s':env.physics_dt,'control_dt_s':env.step_dt,'scope':'ARES SELF-BUILT CONTACT COMPOSITION; NOT public benchmark or field-aligned task',
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
        'frame_source_sha256':hashlib.sha256(frame_source).hexdigest(),
        'geometry_source_sha256':hashlib.sha256(geometry_source).hexdigest(),
        'recontact_enabled':a.recontact_enabled,'exact_recontact':a.exact_recontact,'parameter_hold_steps':a.parameter_hold,'event_refresh':a.event_refresh,
        'recontact_contract':{'max_attempts':2,'phase':'phase0 only','face_error_trigger_deg':35,'minimum_goal_error_m':.010,
            'cooldown_control_steps':120,'normal_angle_limit_rad':a.normal_angle_limit,'phase1_normal':'fixed slot direction',
            'load_event_threshold_N':1.,'event_angular_velocity_change_rad_s':.25,'event_lateral_velocity_change_m_s':.015,
            'minimum_parameter_hold_steps_before_event':8,'load_event_is_complete_contact_detector':False,
            'minimum_real_object_displacement_before_recontact_m':.020,
            'default_phase0_control':'preserved slot-forward nominal until actual recovery',
            'normal_parameter_active':'phase0 only after a physical recovery attempt'},
        'dependency_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
            (['contact_masked_policy_v8.py','recontact_waypoint_v8.py','recontact_minkowski_v8.py','audit_contact_actor_gradient_v8.py'] if a.recontact_enabled else [])},
        'method':a.method,'frame':a.frame,'blind_outcome':a.blind_outcome,'mask_measured_effect':getattr(a,'mask_measured_effect',False),'contact_phase':a.contact_phase,'contact_limit_m':a.contact_limit,'contact_following':a.contact_following,'contact_steering':a.contact_steering,'disable_contact':a.disable_contact,'task_version':a.task_version,'switch_mode':a.switch_mode,'switch_dwell':a.switch_dwell,
        'forced_switch_signal':a.force_switch_signal,
        'rule_recovery':a.rule_recovery,
        'policy_mode':a.policy_mode,'action_cost':a.action_cost,
        'withdraw_rule':a.withdraw_rule,
        'reference_lead_limit_m':None if a.task_version=='v5_replay' else .006,
        'stochastic_evaluation':a.stochastic_eval,'blind_next':a.blind_next,'blind_rotation':a.blind_rotation,'observation_dim':observation_dim,'action_dim':action_dim,
        'actor_log_std':policy.policy.log_std.detach().cpu().tolist(),
        'legacy_log_std':legacy.policy.log_std.detach().cpu().tolist() if legacy else None,
        'tool_contract':{'max_tilt_rad':float(np.radians(3.5)),'max_yaw_rad':a.yaw_limit+.001 if a.yaw_enabled else float(np.radians(3.5)),
            'physical_yaw_limit_rad':a.yaw_limit if a.yaw_enabled else float(np.radians(3.5)),
            'command_yaw_limit_rad':a.yaw_command_limit if a.yaw_enabled else 0.,
            'command_to_physical_allowance_rad':a.yaw_limit-a.yaw_command_limit+.001 if a.yaw_enabled else 0.,
            'numeric_threshold_margin_rad':.001 if a.yaw_enabled else 0.,
            'reference_level_rotation':meta['level_link6_rotation'],**({} if a.yaw_enabled else {'max_full_rotation_rad':float(np.radians(3.5))})},
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
    (a.output/'actor_gradient_audits.json').write_text(json.dumps(actor_gradient_audits,indent=2))
    (a.output/'frozen_evaluation.json').write_text(json.dumps(env.records,indent=2))
    (a.output/'action_log.json').write_text(json.dumps(action_log))
    if env.contacts:np.savez_compressed(a.output/'post_step_contact_history.npz',states=np.array(env.contacts),
        columns=['physics_tick','case_id','phase','max_robot_net_contact_N','sum_fx','sum_fy','sum_fz','vx','vy','vz','wx','wy','wz','obj_x','obj_y','obj_z'],sample_dt_s=env.step_dt)
    np.savez_compressed(a.output/'actual_physics_trace.npz',states=np.array(env.trace),sample_dt_s=env.physics_dt,
        goals_body_m=env.goals.cpu().numpy(),columns=np.array([*[f'q{i}' for i in range(6)],'obj_x','obj_y','obj_z',
        'paddle_x','paddle_y','paddle_z','vx','vy','vz','stage','obj_qw','obj_qx','obj_qy','obj_qz','case_id',
            'cmd_x','cmd_y','cmd_z','goal_x','goal_y','goal_z','action_x','action_y','applied_dx','applied_dy',
            'wx','wy','wz',*[f'qd{i}' for i in range(6)],'physics_tick','switch_signal','submitted_x','submitted_y','submitted_switch','tool_yaw_command_rad','actual_link6_qw','actual_link6_qx','actual_link6_qy','actual_link6_qz',
            'recontact_mode','parameter_hold_remaining','held_contact_m','held_yaw_rad','held_normal_angle_rad','push_direction_x','push_direction_y']))
    if frames:
        from PIL import Image
        Image.fromarray(frames[-1]).save(a.output/'physical_scene.png')
        images=[Image.fromarray(f).resize((640,480)).quantize(colors=128) for f in frames]
        durations=[max(10,round((frame_records[min(i+1,len(frame_records)-1)]['physics_steps']-r['physics_steps'])*env.physics_dt*1000))
                   if i<len(frame_records)-1 else 267 for i,r in enumerate(frame_records)]
        images[0].save(a.output/'actual_physics.gif',save_all=True,append_images=images[1:],duration=durations,loop=0)
        (a.output/'video_frame_records.json').write_text(json.dumps(frame_records))
    print(json.dumps(result,default=json_default),flush=True);env.close()
