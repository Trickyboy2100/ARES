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
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
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
    torch.set_num_threads(2)
    provenance=json.loads((a.asset/'asset_manifest.json').read_text())
    t=np.array(provenance['body_bases']['right']);quat=Rotation.from_matrix(t[:3,:3]).as_quat()
    q0=provenance['start_q'];qnames=[f'joint_{i+1}' for i in range(6)]

    @configclass
    class Config(DirectRLEnvCfg):
        decimation=2;episode_length_s=1.;action_space=6;observation_space=18;state_space=0
        sim=sim_utils.SimulationCfg(dt=1/120,render_interval=2,device=f'cuda:{a.gpu}')
        scene=InteractiveSceneCfg(num_envs=a.num_envs,env_spacing=2.5,replicate_physics=True,clone_in_fabric=False)
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
            self.total_physics_steps=0;self.episodes=0;self.rows=[];self.last_snapshot={}
            super().__init__(cfg)
            self.ids,_=self.robot.find_joints(qnames,preserve_order=True)
            self.base=torch.tensor(q0,device=self.device)
            self.targets=self.base.repeat(self.num_envs,1)
            self.initial=self.targets.clone();self.actions=torch.zeros_like(self.targets)
            self.low=torch.tensor(provenance['joint_lower'],device=self.device)
            self.high=torch.tensor(provenance['joint_upper'],device=self.device)
        def _setup_scene(self):
            self.robot=Articulation(self.cfg.robot_cfg)
            self.contact=ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/Link_.*',history_length=1))
            sim_utils.GroundPlaneCfg().func('/World/Ground',sim_utils.GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            self.scene.articulations['robot']=self.robot;self.scene.sensors['contacts']=self.contact
            if not a.skip_capture:
                self.preview=Camera(CameraCfg(prim_path='/World/PreviewCamera',update_period=0.,height=720,width=960,
                    data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,clipping_range=(.01,100.))))
                # One global overview camera is not an N-env sensor. Do not let
                # InteractiveScene.reset(env_ids) index its single camera buffer.
            self.scene.filter_collisions(global_prim_paths=['/World/Ground'])
            sim_utils.DomeLightCfg(intensity=1500).func('/World/Light',sim_utils.DomeLightCfg(intensity=1500))
        def _pre_physics_step(self,actions):self.actions=actions.clamp(-1,1)
        def _apply_action(self):
            self.robot.set_joint_position_target(self.base+.04*self.actions,joint_ids=self.ids)
            self.total_physics_steps+=1
        def _get_observations(self):
            q=self.robot.data.joint_pos[:,self.ids];v=self.robot.data.joint_vel[:,self.ids]
            return {'policy':torch.cat(((q-self.base)/.04,v,self.targets.sub(q)/.04),-1)}
        def _get_rewards(self):
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
            return invalid,self.episode_length_buf>=self.max_episode_length-1
        def _reset_idx(self,env_ids):
            if env_ids is None:env_ids=self.robot._ALL_INDICES
            super()._reset_idx(env_ids)
            # Genuine reset, never applied in an action or trajectory step.
            q=self.robot.data.default_joint_pos[env_ids].clone();v=torch.zeros_like(q)
            if hasattr(self,'targets'):
                self.targets[env_ids]=self.base+torch.rand((len(env_ids),6),device=self.device)*.02-.01
                q[:,self.ids]=self.base+torch.rand((len(env_ids),6),device=self.device)*.008-.004
                self.initial[env_ids]=q[:,self.ids];self.episodes+=len(env_ids)
            state=self.robot.data.default_root_state[env_ids].clone();state[:,:3]+=self.scene.env_origins[env_ids]
            self.robot.write_root_pose_to_sim(state[:,:7],env_ids=env_ids)
            self.robot.write_root_velocity_to_sim(state[:,7:],env_ids=env_ids)
            self.robot.write_joint_state_to_sim(q,v,env_ids=env_ids)

    print('R1_BUILD_ENV',flush=True)
    env=Env(Config());observations,_=env.reset(seed=a.seed)
    print('R1_ENV_READY',flush=True)
    if not a.skip_capture:
        origins=env.scene.env_origins.cpu().numpy();center=origins.mean(0)
        span=max(3.,float(np.max(np.ptp(origins,axis=0)))+2.)
        env.preview.set_world_poses_from_view(eyes=torch.tensor(np.array([center+[span,-span,span]]),device=env.device,dtype=torch.float32),
            targets=torch.tensor(np.array([center+[0,0,1]]),device=env.device,dtype=torch.float32))
    frames=[];frame_records=[]
    if a.replay:
        data=json.loads(a.replay.read_text());points=np.array(data['trajectory_points_rad'])
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
        rows.append({'raw_knot':'settled','q_target':points[-1].tolist(),'q_actual':env.robot.data.joint_pos[:,env.ids].tolist(),
            'link6_pos_w':env.robot.data.body_pos_w[:,link_id].tolist(),
            'link6_quat_w':env.robot.data.body_quat_w[:,link_id].tolist(),
            'max_contact_force_N':float(env.contact.data.net_forces_w.norm(dim=-1).max()),
            'tracking_error_rad':float((env.robot.data.joint_pos[:,env.ids]-torch.tensor(points[-1],device=env.device)).norm(dim=-1).max())})
        (a.output/'replay.json').write_text(json.dumps(rows,indent=2))
        updates=0;changed=False;steps=len(points)*4+120;episodes=0
    else:
        class Adapter(VecEnv):
            def __init__(self):
                super().__init__(env.num_envs,gym.spaces.Box(-np.inf,np.inf,(18,),np.float32),
                    gym.spaces.Box(-1,1,(6,),np.float32))
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
        adapter=Adapter();policy=PPO('MlpPolicy',adapter,device='cpu',seed=a.seed,n_steps=32,batch_size=32,
            n_epochs=4,policy_kwargs={'net_arch':[64,64]},verbose=0)
        before={k:v.clone() for k,v in policy.policy.state_dict().items()}
        policy.set_logger(configure(str(a.output/'ppo'),['csv']))
        class Frames(BaseCallback):
            def _on_step(self):
                if not a.skip_capture and self.n_calls%32==0:
                    env.sim.render();env.preview.update(env.sim.get_physics_dt(),force_recompute=True)
                    frames.append(env.preview.data.output['rgb'][0].cpu().numpy().copy())
                    frame_records.append({'control_step':env.common_step_counter,'env_interactions':self.num_timesteps,
                        'per_env_q':env.robot.data.joint_pos[:,env.ids].tolist()})
                return True
        policy.learn(total_timesteps=a.steps,callback=Frames());policy.save(a.output/'ppo_physics')
        changed=any(not torch.equal(v,before[k]) for k,v in policy.policy.state_dict().items())
        updates=max(int(state['step']) for state in policy.policy.optimizer.state.values() if 'step' in state)
        training_epochs=policy._n_updates;steps=env.total_physics_steps;episodes=env.episodes
        (a.output/'physics_interactions.json').write_text(json.dumps(env.rows,indent=2))
    result={'scope':provenance['scope'],'num_envs':env.num_envs,'seed':a.seed,'physics_dt':1/120,
        'physics_steps':steps,'env_physics_steps':steps*env.num_envs,'episodes':episodes,'gradient_updates':updates,
        'policy_parameters_changed':changed,'real_physics':True,'industrial_task_qualified':False,
        'q_final':env.robot.data.joint_pos[:,env.ids].tolist(),'origins':env.scene.env_origins.cpu().tolist(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'asset':provenance,
        'isaac_lab_git':'3c6e67bb5c7ada942a6d1884ab69338f57596f77','isaac_sim':'5.1.0',
        'limitations':['isolated arm, not dual-arm/full-chassis','fixed fingers and welded tray research geometry',
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
        images[0].save(a.output/'online_ppo_physics.gif',save_all=True,append_images=images[1:],duration=180,loop=0)
        (a.output/'video_frame_records.json').write_text(json.dumps(frame_records,indent=2))
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    env.close()
except BaseException:
    (a.output/'exception.txt').write_text(traceback.format_exc())
    traceback.print_exc();raise
finally:app.close()
