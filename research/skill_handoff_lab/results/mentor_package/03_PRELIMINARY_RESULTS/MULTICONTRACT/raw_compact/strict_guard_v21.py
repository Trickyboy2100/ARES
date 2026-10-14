"""Actual DirectRLEnv physics and standard PPO smoke, not paper method evidence.

Isolated fixed-base JAKA, every source arm/gripper collision retained; reliable
welded tray proxy. Reset-only state writes; actions are PhysX drive targets.
"""
import argparse,json,hashlib,time,sys,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=1)
p.add_argument('--steps',type=int,default=512);p.add_argument('--gpu',type=int,default=1)
p.add_argument('--seed',type=int,default=31);p.add_argument('--replay',type=Path)
p.add_argument('--skip-capture',action='store_true')
p.add_argument('--hold-correction',action='store_true',help='bounded deterministic joint feedback baseline, not RL or a planner trajectory')
p.add_argument('--industrial',action='store_true')
p.add_argument('--controller',choices=['ppo','nominal','residual','contract','contract_blind'],default='ppo')
p.add_argument('--eval-episodes',type=int,default=0)
p.add_argument('--opening-height',type=float,default=.18)
p.add_argument('--opening-width',type=float,default=.26)
p.add_argument('--load-policy',type=Path)
p.add_argument('--preplace-result',type=Path,help='validated raw cuRobo knots before local learned skill, no fallback')
p.add_argument('--eval-case-file',type=Path)
p.add_argument('--eval-case-ids',default='')
p.add_argument('--residual-mode',choices=['learned','zero','random'],default='learned')
p.add_argument('--action-audit',action='store_true')
p.add_argument('--task-suite',type=Path);p.add_argument('--split',choices=['train','validation','test'],default='train')
p.add_argument('--reward-revision',choices=['legacy','v2'],default='legacy');p.add_argument('--process-audit',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
suite=json.loads(a.task_suite.read_text()) if a.task_suite else None
active_tasks=[suite[a.split][i%len(suite[a.split])] for i in range(a.num_envs)] if suite else None
conditioned=a.controller in ['contract','contract_blind']
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device=f'cuda:{a.gpu}',enable_cameras=True,
    kit_args=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false')
app=launcher.app
print('R1_APP_LAUNCHED',flush=True)
try:
    import numpy as np
    import torch
    import gymnasium as gym
    from scipy.spatial.transform import Rotation
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation,ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs import DirectRLEnv,DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor,ContactSensorCfg,Camera,CameraCfg
    from isaaclab.utils import configclass
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.callbacks import BaseCallback
    from isaaclab.utils.math import quat_apply,quat_mul,quat_conjugate,axis_angle_from_quat,skew_symmetric_matrix,matrix_from_quat
    torch.set_num_threads(2)
    provenance=json.loads((a.asset/'asset_manifest.json').read_text())
    t=np.array(provenance['body_bases']['right']);quat=Rotation.from_matrix(t[:3,:3]).as_quat()
    q0=provenance['start_q'];qnames=[f'joint_{i+1}' for i in range(6)]
    industrial_center=[.454750000,-.480000000,1.450000000]

    @configclass
    class Config(DirectRLEnvCfg):
        decimation=2;episode_length_s=(5. if suite else 4.) if a.industrial else 1.;action_space=3 if a.industrial else 6;observation_space=(40 if conditioned else 25) if a.industrial else 18;state_space=0
        sim=sim_utils.SimulationCfg(dt=1/120,render_interval=2,device=f'cuda:{a.gpu}')
        scene=InteractiveSceneCfg(num_envs=a.num_envs,env_spacing=2.5,replicate_physics=not bool(suite),clone_in_fabric=False)
        seed=a.seed
        robot_cfg=ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
            spawn=sim_utils.UsdFileCfg(usd_path=str((a.asset/'jaka_held.usda').resolve()),
                activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=32,solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(t[:3,3]),rot=(float(quat[3]),*map(float,quat[:3])),
                joint_pos=dict(zip(qnames,q0))),
            actuators={'arm':ImplicitActuatorCfg(joint_names_expr=['joint_.*'],stiffness=1500.,damping=80.,
                effort_limit_sim=100.,velocity_limit_sim=1.57)})

    class Env(DirectRLEnv):
        def __init__(self,cfg):
            self.total_physics_steps=0;self.episodes=0;self.rows=[];self.last_snapshot={};self.action_rows=[];self.process_rows=[];self.scene_specs=[]
            super().__init__(cfg)
            self.ids,_=self.robot.find_joints(qnames,preserve_order=True)
            self.base=torch.tensor(q0,device=self.device)
            self.joint_command=self.base.repeat(self.num_envs,1)
            self.targets=self.base.repeat(self.num_envs,1)
            self.initial=self.targets.clone();self.actions=torch.zeros_like(self.targets)
            self.low=torch.tensor(provenance['joint_lower'],device=self.device)
            self.high=torch.tensor(provenance['joint_upper'],device=self.device)
            if a.industrial:
                self.link_id=self.robot.find_bodies('Link_6')[0][0]
                request=json.loads((a.asset/'industrial_request.json').read_text())
                self.held_offset=torch.tensor(request['attached_object_collision']['T_link6_object'],device=self.device,dtype=torch.float32)[:3,3]
                level_r=np.array(request['orientation_lock']['target_R_body_tcp'])@np.array(provenance['tool'])[:3,:3].T
                level_xyzw=Rotation.from_matrix(level_r).as_quat()
                self.level_quat=torch.tensor([level_xyzw[3],*level_xyzw[:3]],device=self.device,dtype=torch.float32).repeat(self.num_envs,1)
                self.center=torch.tensor([g.get('center_body_m',industrial_center) for g in active_tasks] if suite else [industrial_center]*self.num_envs,device=self.device)
                self.group_start_q=torch.tensor([g.get('start_q',q0) for g in active_tasks] if suite else [q0]*self.num_envs,device=self.device)
                if suite:
                    quats=[]
                    for g in active_tasks:
                        rotation=Rotation.from_euler('z',g.get('tool_yaw_deg',0),degrees=True).as_matrix()@level_r
                        x,y,z,w=Rotation.from_matrix(rotation).as_quat();quats.append([w,x,y,z])
                    self.level_quat=torch.tensor(quats,device=self.device,dtype=torch.float32)
                self.command=self.center.clone()
                self.normals=torch.tensor([[np.cos(np.radians(g['yaw_deg'])),np.sin(np.radians(g['yaw_deg'])),0] for g in active_tasks] if suite else [[1,0,0]]*self.num_envs,device=self.device,dtype=torch.float32)
                self.previous_distance=torch.full((self.num_envs,),float('nan'),device=self.device)
                # Explicit observable Skill Contract: workspace, action axes,
                # success depth/alignment, pose/step/force limits and duration.
                self.contract=torch.tensor([-.03,-.035,-.03,.16,.035,.03,1,1,1,.055,.007,.0523599,.001,100,4],
                    device=self.device,dtype=torch.float32).repeat(self.num_envs,1)
                self.mouth=self.center[:,0]+.05;self.goal=self.center.clone();self.goal[:,0]+=.11
                self.success_hold=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
                self.peak_force=torch.zeros(self.num_envs,device=self.device)
                self.task_records=[];self.episode_return=torch.zeros(self.num_envs,device=self.device)
                self.evaluation=False;self.case_ids=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
                self._task_reset(self.robot._ALL_INDICES)
        def _setup_scene(self):
            self.robot=Articulation(self.cfg.robot_cfg)
            self.contact=ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/Link_.*',history_length=1))
            if a.industrial and not suite:
                # Parameterized research cabinet: source tray dimensions, not claimed as measured cabinet CAD.
                cx,cy,cz=industrial_center;mouth=cx+.05
                hz=a.opening_height/2+.015;hy=a.opening_width/2+.015
                walls=[('bottom',(mouth+.08,cy,cz-hz),(.16,a.opening_width+.06,.03)),
                       ('top',(mouth+.08,cy,cz+hz),(.16,a.opening_width+.06,.03)),
                       ('left',(mouth+.08,cy-hy,cz),(.16,.03,a.opening_height)),
                       ('right',(mouth+.08,cy+hy,cz),(.16,.03,a.opening_height))]
                for name,pos,size in walls:
                    spawn=sim_utils.CuboidCfg(size=size,collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.15,.35,.6)))
                    spawn.func('/World/envs/env_0/Cabinet_'+name,spawn,translation=pos)
            sim_utils.GroundPlaneCfg().func('/World/Ground',sim_utils.GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            if a.industrial and suite:
                # Author heterogeneous static geometry before PhysX starts.
                # Reverse order avoids newly inherited env_0 wall definitions.
                for i in reversed(range(a.num_envs)):
                    g=active_tasks[i];angle=np.radians(g['yaw_deg']);c,s=np.cos(angle),np.sin(angle)
                    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    origin=self.scene.env_origins[i].cpu().numpy();mouth=np.array(g.get('center_body_m',industrial_center))+rotation@np.array([g['mouth_offset'],0,0])
                    dz=g['height']/2+.015;dy=g['width']/2+.015;d=g['depth']
                    walls=[('bottom',[d/2,0,-dz],[d,g['width']+.06,.03]),
                        ('left',[d/2,-dy,0],[d,.03,g['height']]),('right',[d/2,dy,0],[d,.03,g['height']])]
                    walls.append(('top',[d/2,0,dz],[d,g['width']+.06,.03]) if g['family']=='T1' else
                        ('top',[d+.015,0,0],[.03,g['width'],g['height']]))
                    for name,offset,size in walls:
                        pos=mouth+rotation@offset
                        spawn=sim_utils.CuboidCfg(size=tuple(size),collision_props=sim_utils.CollisionPropertiesCfg(),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.15,.35,.6) if g['family']=='T1' else (.45,.3,.12)))
                        # env_i already has the cloned origin transform. Author
                        # BODY-local translation here, never add origin twice.
                        spawn.func(f'/World/envs/env_{i}/Cabinet_'+name,spawn,translation=tuple(pos),
                            orientation=(float(np.cos(angle/2)),0.,0.,float(np.sin(angle/2))))
                        self.scene_specs.append({'env':i,'group':g['id'],'family':g['family'],'wall':name,
                            'center_body_m':pos.tolist(),'size_m':size,'rotation_body':rotation.tolist()})
            self.scene.articulations['robot']=self.robot;self.scene.sensors['contacts']=self.contact
            if not a.skip_capture:
                self.preview=Camera(CameraCfg(prim_path='/World/PreviewCamera',update_period=0.,height=720,width=960,
                    data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,clipping_range=(.01,100.))))
                # One global overview camera is not an N-env sensor. Do not let
                # InteractiveScene.reset(env_ids) index its single camera buffer.
            self.scene.filter_collisions(global_prim_paths=['/World/Ground'])
            sim_utils.DomeLightCfg(intensity=1500).func('/World/Light',sim_utils.DomeLightCfg(intensity=1500))
        def _pre_physics_step(self,actions):
            self.actions=actions.clamp(-1,1)
            if a.industrial:
                observation_before_action=self._get_observations()['policy'].clone() if a.action_audit else None
                applied=self.actions
                if a.residual_mode=='zero':applied=torch.zeros_like(applied)
                if a.residual_mode=='random':applied=torch.rand_like(applied)*2-1
                residual=applied*self.contract[:,12:13]*.5
                delta=applied*self.contract[:,12:13]
                nominal=(self.goal-self.command).clamp(-self.contract[:,12:13],self.contract[:,12:13])
                if a.controller=='nominal':delta=nominal
                elif a.controller in ['residual','contract','contract_blind']:delta=nominal+residual
                before=delta.clone();delta=delta*self.contract[:,6:9]
                delta=delta.clamp(-self.contract[:,12:13],self.contract[:,12:13])
                previous=self.command.clone()
                self.command=(self.command+delta).clamp(self.center+self.contract[:,:3],self.center+self.contract[:,3:6])
                if a.action_audit:
                    self.action_rows.append({'control_step':self.common_step_counter,'phase':'evaluation' if self.evaluation else 'diagnostic',
                        'case_ids':self.case_ids.tolist(),'raw_policy_action':actions.tolist(),'clipped_policy_action':self.actions.tolist(),
                        'applied_policy_action':applied.tolist(),'nominal_delta_m':nominal.tolist(),'residual_before_clip_m':residual.tolist(),
                        'combined_before_mask_clip_m':before.tolist(),'after_mask_clip_m':delta.tolist(),
                        'actual_command_delta_m':(self.command-previous).tolist(),'q_before':self.robot.data.joint_pos[:,self.ids].tolist(),
                        'policy_observation':observation_before_action.tolist(),
                        'post_command_policy_observation':self._get_observations()['policy'].tolist(),'executed_contract':self.contract.tolist()})
        def _apply_action(self):
            if a.industrial:
                if a.process_audit and self.evaluation:self._record_process()
                actual=self.robot.data.body_pos_w[:,self.link_id]
                quat=self.robot.data.body_quat_w[:,self.link_id]
                offset=quat_apply(quat,self.held_offset.repeat(self.num_envs,1))
                center=actual+offset-self.scene.env_origins
                # Integrated bounded task-space command allows the physical
                # servo to compensate gravity instead of chasing moving offsets.
                delta=(self.command-center).clamp(-.01,.01)
                workspace_low=self.center+self.contract[:,:3]
                workspace_high=self.center+self.contract[:,3:6]
                delta=(center+delta).clamp(workspace_low,workspace_high)-center
                self.peak_force=torch.maximum(self.peak_force,self.contact.data.net_forces_w.norm(dim=-1).max(dim=-1).values)
                # Actor selects translation only. Angular target remains the contract's level orientation.
                angular=axis_angle_from_quat(quat_mul(self.level_quat,quat_conjugate(quat)))
                error=torch.cat((delta,(angular*(2. if suite else 1.)).clamp(-.06 if suite else -.03,.06 if suite else .03)),-1)
                jac=self.robot.root_physx_view.get_jacobians()[:,self.link_id-1,:,self.ids].clone()
                # Numerically audited on this rotated JAKA: Jacobian is WORLD
                # and referenced to body COM, not the Link6 origin.
                com_to_held=actual+offset-self.robot.data.body_com_pos_w[:,self.link_id]
                jac[:,:3,:]=jac[:,:3,:]-skew_symmetric_matrix(com_to_held)@jac[:,3:,:]
                if self.total_physics_steps==0:
                    (a.output/'controller_jacobian_audit.json').write_text(json.dumps({'q':self.robot.data.joint_pos[0,self.ids].tolist(),
                        'J_body_held':jac[0].tolist(),'root_quaternion':self.robot.data.root_quat_w[0].tolist()},indent=2))
                jt=jac.transpose(1,2)
                dq=(jt@torch.linalg.solve(jac@jt+torch.eye(6,device=self.device)*.0001,error.unsqueeze(-1))).squeeze(-1)
                # Scale the complete IK direction; per-joint clipping destroys
                # translation/rotation cancellation and violates level keeping.
                dq=dq*torch.minimum(torch.ones_like(dq[:,:1]),.012/dq.abs().amax(-1,keepdim=True).clamp_min(1e-8))
                q=self.robot.data.joint_pos[:,self.ids]
                if suite:
                    # Bounded integral pose servo compensates gravity/loading;
                    # common to EVERY baseline, never a learned privileged action.
                    self.joint_command=(self.joint_command+.25*dq).clamp(q-.04,q+.04).clamp(self.low,self.high)
                    target=self.joint_command
                else:target=(q+dq).clamp(self.low,self.high)
                self.robot.set_joint_position_target(target,joint_ids=self.ids)
            else:self.robot.set_joint_position_target(self.base+.04*self.actions,joint_ids=self.ids)
            self.total_physics_steps+=1
        def _get_observations(self):
            q=self.robot.data.joint_pos[:,self.ids];v=self.robot.data.joint_vel[:,self.ids]
            if a.industrial:
                center,error,depth,tilt,force=self._task_state()
                geometry=torch.tensor([[g['width'],g['height'],.0995,.1795,.03,np.radians(g['yaw_deg'])] for g in active_tasks] if suite else
                    [[a.opening_width,a.opening_height,.0995,.1795,.03,.0523599]]*self.num_envs,device=self.device)
                state=torch.cat((q-self.base,v,error*10,depth[:,None]*10,geometry[:,:6],(self.command-center)*10),-1)
                if conditioned:
                    scales=torch.tensor([10,10,10,10,10,10,1,1,1,10,100,20,1000,.01,.25],device=self.device)
                    state=torch.cat((state,torch.zeros_like(self.contract) if a.controller=='contract_blind' else self.contract*scales),-1)
                return {'policy':state}
            return {'policy':torch.cat(((q-self.base)/.04,v,self.targets.sub(q)/.04),-1)}
        def _task_state(self):
            quat=self.robot.data.body_quat_w[:,self.link_id]
            center=self.robot.data.body_pos_w[:,self.link_id]+quat_apply(quat,self.held_offset.repeat(self.num_envs,1))-self.scene.env_origins
            error=self.goal-center
            depth=((center-self.center)*self.normals).sum(-1)-torch.tensor([g['mouth_offset'] for g in active_tasks],device=self.device) if suite else center[:,0]-self.mouth
            tilt=axis_angle_from_quat(quat_mul(self.level_quat,quat_conjugate(quat))).norm(dim=-1)
            force=self.contact.data.net_forces_w.norm(dim=-1).max(dim=-1).values
            return center,error,depth,tilt,force
        def _alignment(self,error):return (error-(error*self.normals).sum(-1,keepdim=True)*self.normals).norm(dim=-1)
        def _record_process(self):
            self.process_rows.append({'physics_step':self.total_physics_steps,'phase':getattr(self,'process_phase','RL_LOCAL'),'case_ids':self.case_ids.tolist(),
                'q':self.robot.data.joint_pos[:,self.ids].tolist(),
                'body_pos_body':(self.robot.data.body_pos_w-self.scene.env_origins[:,None,:]).tolist(),
                'body_quat_w':self.robot.data.body_quat_w.tolist(),
                'forces_per_link_w':self.contact.data.net_forces_w.tolist()})
        def _task_reset(self,env_ids):
            self.goal[env_ids]=self.center[env_ids];self.goal[env_ids,0]+=.11
            self.goal[env_ids,1:]+=torch.rand((len(env_ids),2),device=self.device)*.012-.006
            self.success_hold[env_ids]=0;self.episode_return[env_ids]=0
            self.peak_force[env_ids]=0
            self.command[env_ids]=self.center[env_ids]
            self.previous_distance[env_ids]=float('nan')
            if suite:
                rng=suite['contract_episode_randomization']
                for i in env_ids.tolist():
                    g=active_tasks[i];normal=self.normals[i];lateral=torch.tensor([-float(normal[1]),float(normal[0]),0],device=self.device)
                    mask=torch.tensor(g['mask'],device=self.device,dtype=torch.float32)
                    margin=float(np.random.uniform(*rng['workspace_margin_m']))
                    low=torch.full((3,),-margin,device=self.device);high=-low
                    forward=normal*g['travel'];low=torch.minimum(low,forward-margin);high=torch.maximum(high,forward+margin)
                    self.contract[i,:3]=low;self.contract[i,3:6]=high;self.contract[i,6:9]=mask
                    self.contract[i,9]=g['success_depth'];self.contract[i,10]=float(np.random.uniform(*rng['alignment_m']))
                    self.contract[i,11]=float(np.radians(np.random.uniform(*rng['pose_tolerance_deg'])))
                    self.contract[i,12]=float(np.random.uniform(*rng['step_m']))
                    self.contract[i,14]=max(float(np.random.uniform(*rng['horizon_s'])),g['travel']/float(self.contract[i,12])*self.step_dt+.6)
                    lat=float(np.random.uniform(-g['goal_yz_jitter'],g['goal_yz_jitter']))
                    if not bool((lateral.abs()*mask).sum()):lat=0.
                    self.goal[i]=self.center[i]+forward+lateral*lat
                    if g['mask'][2]:self.goal[i,2]+=float(np.random.uniform(-g['goal_yz_jitter'],g['goal_yz_jitter']))
        def _get_rewards(self):
            if a.industrial:
                center,error,depth,tilt,force=self._task_state()
                distance=error.norm(dim=-1)
                if a.reward_revision=='v2':
                    progress=torch.where(torch.isfinite(self.previous_distance),self.previous_distance-distance,torch.zeros_like(distance))
                    rewards=20*progress-self.step_dt-.01*force.clamp(max=100)*self.step_dt-.0001*self.actions.square().sum(-1)
                    rewards+=20*(self.task_success&~self.task_failed&~self.task_timeout).float()-10*(self.task_failed|self.task_timeout).float()
                    self.previous_distance=distance.clone()
                else:rewards=2*torch.exp(-distance*25)+depth*5-self._alignment(error)*30-tilt*2-force.clamp(max=100)*.002-.01+10*self.task_success.float()
                self.episode_return+=rewards
                for record in reversed(self.task_records):
                    if record['control_step']!=self.common_step_counter:break
                    record['episode_return']=float(self.episode_return[record['env']])
                return rewards
            q=self.robot.data.joint_pos[:,self.ids];error=(q-self.targets).square().sum(-1)
            rewards=-error/.0004-.001*self.actions.square().sum(-1)
            if self.common_step_counter%16==0:
                forces=self.contact.data.net_forces_w.norm(dim=-1).max(dim=-1).values
                self.rows.append({'control_step':self.common_step_counter,'physics_steps':self.total_physics_steps,
                    'mean_reward':float(rewards.mean()),'mean_joint_error_rad':float(error.sqrt().mean()),
                    'per_env_joint_error_rad':error.sqrt().tolist(),'max_contact_force_N':float(forces.max()),
                    'per_env_q':q.tolist(),'per_env_target':self.targets.tolist(),'episodes':self.episodes})
            return rewards
        def _get_dones(self):
            q=self.robot.data.joint_pos[:,self.ids]
            self.terminal_observation=self._get_observations()['policy'].clone()
            invalid=(q<self.low).any(-1)|(q>self.high).any(-1)|(~torch.isfinite(q)).any(-1)
            if a.industrial:
                center,error,depth,tilt,force=self._task_state()
                aligned=(self._alignment(error)<self.contract[:,10])&(tilt<self.contract[:,11])
                # Trailing tray edge must enter, with 5 mm depth margin.
                success=(depth>self.contract[:,9])&aligned&(force<30)
                self.success_hold=torch.where(success,self.success_hold+1,torch.zeros_like(self.success_hold))
                self.task_success=self.success_hold>=6
                self.peak_force=torch.maximum(self.peak_force,force)
                relative=center-self.center
                # Hard measured-state guard as well as command projection.
                # 1mm is an explicit servo tracking tolerance, not geometry slack.
                workspace_failed=(relative<self.contract[:,:3]-.001).any(-1)|(relative>self.contract[:,3:6]+.001).any(-1)
                failed=invalid|(tilt>self.contract[:,11])|(self.peak_force>self.contract[:,13])|workspace_failed
                timeout=self.episode_length_buf>=torch.ceil(self.contract[:,14]/self.step_dt).long()-(0 if a.reward_revision=='v2' else 1)
                if a.reward_revision=='v2':timeout=timeout&~self.task_success
                self.task_failed=failed;self.task_timeout=timeout
                if a.process_audit and self.evaluation:self._record_process()
                ended=self.task_success|failed|timeout
                for i in ended.nonzero().flatten().tolist():
                    if self.evaluation and self.case_ids[i]<0:continue
                    self.task_records.append({'env':i,'control_step':self.common_step_counter,'success':bool(self.task_success[i] and not failed[i]),
                        'case_id':int(self.case_ids[i]) if self.evaluation else None,
                        'failed':bool(failed[i]),'timeout':bool(timeout[i]),'depth_m':float(depth[i]),'alignment_m':float(self._alignment(error)[i]),
                        'failure_reasons':{'soft_limit_or_invalid_state':bool(invalid[i]),'pose':bool(tilt[i]>self.contract[i,11]),
                            'excess_peak_net_force':bool(self.peak_force[i]>self.contract[i,13]),'workspace':bool(workspace_failed[i])},
                        'task_group':active_tasks[i]['id'] if suite else 'legacy_T1','contract':self.contract[i].tolist(),
                        'orientation_error_rad':float(tilt[i]),'contact_force_N':float(force[i]),'return_before_terminal_reward':float(self.episode_return[i]),
                        'peak_net_contact_force_N':float(self.peak_force[i]),
                        'goal_body_m':self.goal[i].tolist(),'terminal_q':q[i].tolist(),'episode_control_steps':int(self.episode_length_buf[i])})
                if a.reward_revision=='v2':return self.task_success|failed|timeout,torch.zeros_like(timeout)
                return self.task_success|failed,timeout
            return invalid,self.episode_length_buf>=self.max_episode_length-1
        def _reset_idx(self,env_ids):
            if env_ids is None:env_ids=self.robot._ALL_INDICES
            super()._reset_idx(env_ids)
            # Genuine reset, never applied in an action or trajectory step.
            q=self.robot.data.default_joint_pos[env_ids].clone();v=torch.zeros_like(q)
            if hasattr(self,'targets'):
                self.targets[env_ids]=self.base+torch.rand((len(env_ids),6),device=self.device)*.02-.01
                q[:,self.ids]=self.base if a.replay else self.base+torch.rand((len(env_ids),6),device=self.device)*.008-.004
                if suite and hasattr(self,'group_start_q'):q[:,self.ids]=self.group_start_q[env_ids]+torch.rand((len(env_ids),6),device=self.device)*.008-.004
                self.initial[env_ids]=q[:,self.ids];self.episodes+=len(env_ids)
                if a.industrial and hasattr(self,'goal'):self._task_reset(env_ids)
                if a.industrial and getattr(self,'evaluation',False):
                    for i in env_ids.tolist():
                        index=int(self.eval_cursor[i]);self.eval_cursor[i]+=self.num_envs
                        case=self.eval_cases[index%a.eval_episodes]
                        self.case_ids[i]=case['case_id'] if index<a.eval_episodes else -1
                        q[env_ids.tolist().index(i),self.ids]=(self.group_start_q[i] if suite else self.base)+torch.tensor(case['joint_noise'],device=self.device)
                        self.goal[i]=torch.tensor(case['goal_body_m'],device=self.device)
                        if 'contract' in case:self.contract[i]=torch.tensor(case['contract'],device=self.device)
            state=self.robot.data.default_root_state[env_ids].clone();state[:,:3]+=self.scene.env_origins[env_ids]
            self.robot.write_root_pose_to_sim(state[:,:7],env_ids=env_ids)
            self.robot.write_root_velocity_to_sim(state[:,7:],env_ids=env_ids)
            self.robot.write_joint_state_to_sim(q,v,env_ids=env_ids)
            if hasattr(self,'joint_command'):self.joint_command[env_ids]=q[:,self.ids]

    print('R1_BUILD_ENV',flush=True)
    env=Env(Config());observations,_=env.reset(seed=a.seed)
    print('R1_ENV_READY',flush=True)
    if suite:(a.output/'scene_specs.json').write_text(json.dumps(env.scene_specs,indent=2))
    if suite:
        import omni.usd
        from pxr import UsdGeom,Usd
        stage=omni.usd.get_context().get_stage();bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
        actual=[]
        for spec in env.scene_specs:
            path=f"/World/envs/env_{spec['env']}/Cabinet_{spec['wall']}"
            box=bbox.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            origin=env.scene.env_origins[spec['env']].cpu().numpy()
            actual.append({**spec,'actual_center_body_m':(np.array(box.GetMidpoint())-origin).tolist(),
                'actual_aabb_size_m':np.array(box.GetSize()).tolist()})
        (a.output/'actual_scene_bounds.json').write_text(json.dumps(actual,indent=2))
        error=max(np.linalg.norm(np.array(x['actual_center_body_m'])-x['center_body_m']) for x in actual)
        if error>1e-4:raise ValueError(f'Authored/actual wall coordinates disagree: {error}m')
    if not a.skip_capture:
        origins=env.scene.env_origins.cpu().numpy();center=origins.mean(0)
        span=max(3.,float(np.max(np.ptp(origins,axis=0)))+2.)
        env.preview.set_world_poses_from_view(eyes=torch.tensor(np.array([center+[span,-span,span]]),device=env.device,dtype=torch.float32),
            targets=torch.tensor(np.array([center+[0,0,1]]),device=env.device,dtype=torch.float32))
    frames=[];frame_records=[]
    if a.replay:
        data=json.loads(a.replay.read_text());points=np.array(data['trajectory_points_rad'],dtype=np.float32)
        env.base=torch.tensor(q0,device=env.device)
        rows=[]
        link_id=env.robot.find_bodies('Link_6')[0][0]
        # Time-scale raw planner path, retaining raw knots, through drive targets.
        for index,q in enumerate(points):
            for dwell in range(4):
                env.robot.set_joint_position_target(torch.tensor(q,device=env.device).repeat(a.num_envs,1),joint_ids=env.ids)
                env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.sim.get_physics_dt())
            actual=env.robot.data.joint_pos[:,env.ids]
            rows.append({'raw_knot':index,'q_target':q.tolist(),'q_actual':actual.tolist(),
                'link6_pos_w':env.robot.data.body_pos_w[:,link_id].tolist(),
                'link6_quat_w':env.robot.data.body_quat_w[:,link_id].tolist(),
                'max_contact_force_N':float(env.contact.data.net_forces_w.norm(dim=-1).max()),
                'tracking_error_rad':float((actual-torch.tensor(q,device=env.device)).norm(dim=-1).max())})
        for _ in range(120):
            env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.sim.get_physics_dt())
        correction_steps=0
        if a.hold_correction:
            # A diagnostic baseline: compensate measured static tracking error.
            # This is explicitly not a new cuRobo trajectory or learned recovery.
            command=torch.tensor(points[-1],device=env.device).repeat(a.num_envs,1)
            desired=command.clone()
            for iteration in range(4):
                command=(command+(desired-env.robot.data.joint_pos[:,env.ids]).clamp(-.01,.01)).clamp(env.low,env.high)
                env.robot.set_joint_position_target(command,joint_ids=env.ids)
                for _ in range(60):
                    env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.sim.get_physics_dt())
                    correction_steps+=1
                rows.append({'raw_knot':f'feedback_hold_{iteration}','q_target':points[-1].tolist(),
                    'q_drive_command':command.tolist(),'q_actual':env.robot.data.joint_pos[:,env.ids].tolist(),
                    'link6_pos_w':env.robot.data.body_pos_w[:,link_id].tolist(),
                    'link6_quat_w':env.robot.data.body_quat_w[:,link_id].tolist(),
                    'max_contact_force_N':float(env.contact.data.net_forces_w.norm(dim=-1).max()),
                    'tracking_error_rad':float((env.robot.data.joint_pos[:,env.ids]-desired).norm(dim=-1).max())})
        rows.append({'raw_knot':'settled','q_target':points[-1].tolist(),'q_actual':env.robot.data.joint_pos[:,env.ids].tolist(),
            'link6_pos_w':env.robot.data.body_pos_w[:,link_id].tolist(),
            'link6_quat_w':env.robot.data.body_quat_w[:,link_id].tolist(),
            'max_contact_force_N':float(env.contact.data.net_forces_w.norm(dim=-1).max()),
            'tracking_error_rad':float((env.robot.data.joint_pos[:,env.ids]-torch.tensor(points[-1],device=env.device)).norm(dim=-1).max())})
        (a.output/'replay.json').write_text(json.dumps(rows,indent=2))
        updates=0;changed=False;steps=len(points)*4+120+correction_steps;episodes=0
    else:
        class Adapter(VecEnv):
            def __init__(self):
                super().__init__(env.num_envs,gym.spaces.Box(-np.inf,np.inf,((40 if conditioned else 25) if a.industrial else 18,),np.float32),
                    gym.spaces.Box(-1,1,(3 if a.industrial else 6,),np.float32))
            def reset(self):return env.reset(seed=a.seed)[0]['policy'].cpu().numpy()
            def step_async(self,actions):self.actions=actions
            def step_wait(self):
                o,r,terminated,truncated,info=env.step(torch.tensor(self.actions,device=env.device))
                done=(terminated|truncated).cpu().numpy()
                infos=[{'TimeLimit.truncated':bool(truncated[i] and not terminated[i])} for i in range(env.num_envs)]
                for i in range(env.num_envs):
                    if done[i]:infos[i]['terminal_observation']=env.terminal_observation[i].cpu().numpy()
                return o['policy'].cpu().numpy(),r.cpu().numpy(),done,infos
            def close(self):pass
            def get_attr(self,name,indices=None):return [getattr(env,name,None)]*env.num_envs
            def set_attr(self,name,value,indices=None):setattr(env,name,value)
            def env_method(self,*args,**kwargs):return [None]*env.num_envs
            def env_is_wrapped(self,*args,**kwargs):return [False]*env.num_envs
        adapter=Adapter();policy=PPO('MlpPolicy',adapter,device='cpu',seed=a.seed,n_steps=64 if a.industrial else 32,batch_size=64 if a.industrial else 32,
            n_epochs=4,policy_kwargs={'net_arch':[64,64]},verbose=0)
        if a.load_policy:policy=PPO.load(a.load_policy,env=adapter,device='cpu')
        updates_before=max((int(state['step']) for state in policy.policy.optimizer.state.values() if 'step' in state),default=0)
        before={k:v.clone() for k,v in policy.policy.state_dict().items()}
        policy.set_logger(configure(str(a.output/'ppo'),['csv']))
        class Frames(BaseCallback):
            def _on_step(self):
                if not a.skip_capture and self.n_calls%(64 if a.industrial else 32)==0:
                    env.sim.render();env.preview.update(env.sim.get_physics_dt(),force_recompute=True)
                    frames.append(env.preview.data.output['rgb'][0].cpu().numpy().copy())
                    frame_records.append({'control_step':env.common_step_counter,'env_interactions':self.num_timesteps,
                        'per_env_q':env.robot.data.joint_pos[:,env.ids].tolist()})
                return True
        if a.industrial and a.controller=='nominal':
            for _ in range(max(1,a.steps//env.num_envs)):adapter.step(np.zeros((env.num_envs,3),dtype=np.float32))
        else:
            policy.learn(total_timesteps=a.steps,callback=Frames());policy.save(a.output/'ppo_physics')
        changed=any(not torch.equal(v,before[k]) for k,v in policy.policy.state_dict().items())
        updates=max((int(state['step']) for state in policy.policy.optimizer.state.values() if 'step' in state),default=0)-updates_before
        training_epochs=policy._n_updates;steps=env.total_physics_steps;episodes=env.episodes
        (a.output/'physics_interactions.json').write_text(json.dumps(env.rows,indent=2))
        if a.industrial:
            (a.output/'industrial_episodes.json').write_text(json.dumps(env.task_records,indent=2))
            if a.eval_episodes:
                generator=np.random.default_rng(1731)
                env.eval_cases=[{'case_id':i,'joint_noise':generator.uniform(-.004,.004,6).tolist(),
                    'goal_body_m':[industrial_center[0]+.11,industrial_center[1]+generator.uniform(-.006,.006),
                        industrial_center[2]+generator.uniform(-.006,.006)]} for i in range(a.eval_episodes)]
                if suite:
                    env.eval_cases=[];limits=suite['contract_episode_randomization']
                    for i in range(a.eval_episodes):
                        g=active_tasks[i%env.num_envs];theta=np.radians(g['yaw_deg']);normal=np.array([np.cos(theta),np.sin(theta),0])
                        lateral=np.array([-normal[1],normal[0],0]);margin=generator.uniform(*limits['workspace_margin_m'])
                        forward=normal*g['travel'];low=np.minimum(-np.ones(3)*margin,forward-margin);high=np.maximum(np.ones(3)*margin,forward+margin)
                        c=[*low,*high,*g['mask'],g['success_depth'],generator.uniform(*limits['alignment_m']),
                            np.radians(generator.uniform(*limits['pose_tolerance_deg'])),generator.uniform(*limits['step_m']),100,generator.uniform(*limits['horizon_s'])]
                        c[14]=max(c[14],g['travel']/c[12]*env.step_dt+.6)
                        goal=np.array(g.get('center_body_m',industrial_center))+forward
                        if abs(lateral*np.array(g['mask'])).sum()>0:goal+=lateral*generator.uniform(-g['goal_yz_jitter'],g['goal_yz_jitter'])
                        if g['mask'][2]:goal[2]+=generator.uniform(-g['goal_yz_jitter'],g['goal_yz_jitter'])
                        env.eval_cases.append({'case_id':i,'task_group':g['id'],'joint_noise':generator.uniform(-.004,.004,6).tolist(),
                            'goal_body_m':goal.tolist(),'contract':c})
                if a.eval_case_file:
                    source_cases=json.loads(a.eval_case_file.read_text())
                    ids=[int(x) for x in a.eval_case_ids.split(',') if x]
                    env.eval_cases=[source_cases[i] for i in ids] if ids else source_cases
                    a.eval_episodes=len(env.eval_cases)
                (a.output/'frozen_cases.json').write_text(json.dumps(env.eval_cases,indent=2))
                env.evaluation=True;env.eval_cursor=torch.arange(env.num_envs,device=env.device)
                env.task_records=[];obs=env.reset(seed=1731)[0]['policy'].cpu().numpy()
                if a.preplace_result:
                    if env.num_envs!=1 or not suite or a.eval_episodes!=1:
                        raise ValueError('First full-chain probe requires one suite environment and one frozen episode')
                    planned=json.loads(a.preplace_result.read_text())
                    qualification=json.loads((a.preplace_result.parent/'prefix_mesh_qualification.json').read_text())
                    if qualification.get('dense_raw_planner_path_sha256')!=hashlib.sha256(a.preplace_result.read_bytes()).hexdigest():
                        raise ValueError('Prefix qualification does not match this raw planner path')
                    if qualification.get('asset_sha256')!=hashlib.sha256((a.asset/'jaka_held.usda').read_bytes()).hexdigest():
                        raise ValueError('Prefix qualification collision geometry mismatch')
                    if not qualification.get('nonadjacent_right_arm_self_checked') or any(
                        r.get('collision_samples') or r.get('self_collision_samples') or r.get('soft_limit_violating_samples')
                        or r.get('pose_violating_samples') or r.get('max_workspace_excess_m',0)>.001 for r in qualification['cases']):
                        raise ValueError('Prefix independent mesh/pose/limit qualification failed')
                    points=np.asarray(planned['trajectory_points_rad'],dtype=np.float32)
                    if not (planned.get('success') or planned.get('observed_result')=='SUCCESS') or points.ndim!=2 or points.shape[1]!=6:
                        raise ValueError('A successful raw planner result is required; no fallback')
                    initial=torch.as_tensor(points[0],device=env.device).reshape(1,6)
                    if bool(((initial<env.low)|(initial>env.high)).any()):raise ValueError('Illegal planner initial state')
                    # Initial-state assignment only, before the first motion.
                    env.robot.write_joint_state_to_sim(initial,torch.zeros_like(initial),joint_ids=env.ids)
                    env.robot.set_joint_position_target(initial,joint_ids=env.ids)
                    env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.sim.get_physics_dt())
                    prefix=[];env.process_phase='CUROBO_PREPLACE';prefix_peak_force=0.
                    for knot,q in enumerate(points):
                        command=torch.as_tensor(q,device=env.device).reshape(1,6)
                        if bool(((command<env.low)|(command>env.high)).any()):raise ValueError('Illegal raw planner knot')
                        env.robot.set_joint_position_target(command,joint_ids=env.ids)
                        for _ in range(4):
                            env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.sim.get_physics_dt())
                            env.total_physics_steps+=1
                            prefix_peak_force=max(prefix_peak_force,float(env.contact.data.net_forces_w.norm(dim=-1).max()))
                            actual_q=env.robot.data.joint_pos[:,env.ids]
                            if bool(((actual_q<env.low)|(actual_q>env.high)).any()):raise ValueError('Physical prefix violates soft limits')
                            if a.process_audit:env._record_process()
                        prefix.append({'raw_knot':knot,'q_target':q.tolist(),'q_actual':env.robot.data.joint_pos[:,env.ids].tolist()})
                        if not a.skip_capture and knot%8==0:
                            env.sim.render();env.preview.update(env.sim.get_physics_dt(),force_recompute=True)
                            frames.append(env.preview.data.output['rgb'][0].cpu().numpy().copy())
                            frame_records.append({'phase':'CUROBO_PREPLACE','raw_knot':knot,'physics_step':env.total_physics_steps,
                                'per_env_q':env.robot.data.joint_pos[:,env.ids].tolist()})
                    for _ in range(120):
                        env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.sim.get_physics_dt())
                        env.total_physics_steps+=1
                        if a.process_audit:env._record_process()
                    actual=env._task_state()[0]
                    env.peak_force[:]=prefix_peak_force;env.process_phase='RL_LOCAL'
                    env.command=actual.clone();env.joint_command=env.robot.data.joint_pos[:,env.ids].clone()
                    env.previous_distance[:]=float('nan')
                    obs=env._get_observations()['policy'].cpu().numpy()
                    (a.output/'preplace_execution.json').write_text(json.dumps({'planner_result_sha256':hashlib.sha256(a.preplace_result.read_bytes()).hexdigest(),
                        'qualification_sha256':hashlib.sha256((a.preplace_result.parent/'prefix_mesh_qualification.json').read_bytes()).hexdigest(),
                        'raw_knots':prefix,'dwell_physics_steps_per_knot':4,'settle_physics_steps':120,
                        'held_center_at_skill_boundary_body_m':actual.tolist(),'prefix_peak_net_force_N':prefix_peak_force,
                        'no_state_write_after_initialization':True},indent=2))
                for _ in range(300*(a.eval_episodes//env.num_envs+2)):
                    if a.controller=='nominal':actions=np.zeros((env.num_envs,3),dtype=np.float32)
                    else:
                        with torch.no_grad():actions=policy.policy.get_distribution(torch.as_tensor(obs,device='cpu')).get_actions(deterministic=True).cpu().numpy()
                    obs,_,_,_=adapter.step(actions)
                    if a.load_policy and not a.skip_capture and env.common_step_counter%8==0:
                        env.sim.render();env.preview.update(env.sim.get_physics_dt(),force_recompute=True)
                        frames.append(env.preview.data.output['rgb'][0].cpu().numpy().copy())
                        frame_records.append({'phase':'frozen_policy_evaluation','control_step':env.common_step_counter,
                            'per_env_q':env.robot.data.joint_pos[:,env.ids].tolist()})
                    if len(env.task_records)>=a.eval_episodes:break
                (a.output/'frozen_evaluation.json').write_text(json.dumps({'seed':1731,'requested_episodes':a.eval_episodes,
                    'episodes':sorted(env.task_records,key=lambda r:r['case_id']),
                    'scope':f'frozen geometry-group {a.split}; task families T1/T2 shared with training' if suite else 'same legacy geometry distribution'},indent=2))
        if a.action_audit:(a.output/'action_audit.json').write_text(json.dumps(env.action_rows))
        if a.process_audit:
            import gzip
            with gzip.open(a.output/'process_states.json.gz','wt') as stream:json.dump(env.process_rows,stream)
    result={'scope':('ISOLATED_HELD_JAKA_MULTICONTRACT_T1_T2' if suite else 'ISOLATED_HELD_JAKA_PARAMETRIC_T1_CABINET') if a.industrial else provenance['scope'],'num_envs':env.num_envs,'seed':a.seed,'physics_dt':1/120,
        'physics_steps':steps,'env_physics_steps':steps*env.num_envs,'episodes':episodes,'gradient_updates':updates,
        'policy_parameters_changed':changed,'real_physics':True,'industrial_task_qualified':False,
        'episode_counter_semantics':'reset events including initialization; not successful tasks',
        'replay_result_sha256':hashlib.sha256(a.replay.read_bytes()).hexdigest() if a.replay else None,
        'deterministic_hold_correction':a.hold_correction,
        'industrial_research_task':a.industrial,'controller':a.controller,
        'policy_controls_robot':not(a.industrial and a.controller=='nominal'),
        'contract_conditioning':a.controller=='contract','actor_contract_dim':15 if conditioned else 0,
        'contract_input_blinded':a.controller=='contract_blind',
        'residual_mode':a.residual_mode,
        'reward_revision':a.reward_revision,'task_suite_sha256':hashlib.sha256(a.task_suite.read_bytes()).hexdigest() if suite else None,
        'task_split':a.split,'task_groups':active_tasks,'geometry_replication':not bool(suite),
        'training_physics_steps':steps,'evaluation_physics_steps':env.total_physics_steps-steps if a.industrial else 0,
        'industrial_geometry':{'source':'PARAMETERIZED_RESEARCH_NOT_MEASURED_CABINET','opening_width_m':a.opening_width,
            'opening_height_m':a.opening_height,'cabinet_depth_m':.16,'wall_thickness_m':.03,'entry_direction_body':[1,0,0],
            'initial_held_center_body_m':industrial_center,'mouth_offset_m':.05,'success_depth_m':.055,
            'alignment_tolerance_m':.007,'orientation_tolerance_rad':.0523599,'success_hold_control_steps':6,
            'goal_yz_randomization_m':[-.006,.006],'reset_joint_noise_rad':[-.004,.004],
            'training_seed':a.seed,'evaluation_seed':1731},
        'q_final':env.robot.data.joint_pos[:,env.ids].tolist(),'origins':env.scene.env_origins.cpu().tolist(),
        'body_names':env.robot.body_names,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'asset':provenance,
        'isaac_lab_git':'3c6e67bb5c7ada942a6d1884ab69338f57596f77','isaac_sim':'5.1.0',
        'limitations':['isolated arm, not dual-arm/full-chassis','fixed fingers and welded tray research geometry',
            ('parameterized T1/T2 geometries; no measured contact-guide or method-superiority claim' if suite else
            'single wide research cabinet, not cross-task/contract generalization or superiority') if a.industrial else
            'smoke joint tracking, not mechanism A/B or scientific method superiority']}
    (a.output/'physics_manifest.json').write_text(json.dumps(result,indent=2))
    if a.skip_capture:
        (a.output/'manifest.json').write_text(json.dumps(result,indent=2));env.close();app.close();raise SystemExit(0)
    # Capture current physical state, not a copied static scene or motion loop.
    from isaaclab.sensors import Camera,CameraCfg
    from PIL import Image
    origins=env.scene.env_origins.cpu().numpy();center=origins.mean(0)
    span=max(3.,float(np.max(np.ptp(origins,axis=0)))+2.)
    camera=env.preview
    camera.set_world_poses_from_view(eyes=torch.tensor([center+np.array([span,-span,span])],device=env.device,dtype=torch.float32),
        targets=torch.tensor([center+[0,0,1]],device=env.device,dtype=torch.float32))
    for _ in range(12):env.sim.render();camera.update(env.sim.get_physics_dt(),force_recompute=True)
    image=camera.data.output['rgb'][0].cpu().numpy();Image.fromarray(image).save(a.output/'physical_scene.png')
    if frames:
        images=[Image.fromarray(frame).resize((640,480)) for frame in frames]
        images[0].save(a.output/('policy_evaluation.gif' if a.load_policy else 'online_ppo_physics.gif'),save_all=True,append_images=images[1:],duration=180,loop=0)
        (a.output/'video_frame_records.json').write_text(json.dumps(frame_records,indent=2))
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    env.close()
except BaseException:
    (a.output/'exception.txt').write_text(traceback.format_exc())
    traceback.print_exc();raise
finally:app.close()
