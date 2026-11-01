"""EXP-003 component baseline: shared completion and task-conditioned authority.

This is ordinary SAC/known critic LayerNorm, not the multi-skill paper method.
The context reveals the declared calibration axes, never the hidden bias values.
"""
import argparse, hashlib, json, random, subprocess, time
from pathlib import Path
import numpy as np
import gymnasium as gym
from program_adaptive_rl import SkillEnv
from program_completion_contract import CompletionGate

CONDITIONS = [('Y',[-1.2,0.]),('Y',[0.,0.]),('Y',[1.2,0.]),
              ('YZ',[-1.2,-1.2]),('YZ',[0.,1.2]),('YZ',[1.2,-1.2])]

def make_episode(density, friction):
    from program_adaptive_contact import Episode
    from pull_tool_physics import sp, serial
    class SharedEpisode(Episode):
        def reset(self,*args,**kwargs):
            self.gate=CompletionGate();self.hold_entry=None;self.last_probe=-100;self.probe_steps=0
            return super().reset(*args,**kwargs)
        def completion_measure(self):
            u=self.env.unwrapped;pose=self.goal.inv()*sp(u.charger)
            d=float(np.linalg.norm(pose.p));angle=float(2*np.arccos(np.clip(abs(pose.q[0]),0,1)))
            speed=float(np.linalg.norm(serial(u.charger.linear_velocity[0])))
            angular=float(np.linalg.norm(serial(u.charger.angular_velocity[0])))
            # Frame-to-frame observed stability, rather than the last PhysX
            # substep's oscillatory instantaneous velocity under finger contact.
            raw_speed,raw_angular=speed,angular
            if len(self.env.records)>=2:
                prev=np.asarray(self.env.records[-2]['tool']).reshape(-1)
                now=np.asarray(self.env.records[-1]['tool']).reshape(-1)
                dt=float(u.control_timestep)
                speed=float(np.linalg.norm(now[:3]-prev[:3])/dt)
                qa=prev[3:]/np.linalg.norm(prev[3:]);qb=now[3:]/np.linalg.norm(now[3:])
                angular=float(2*np.arccos(np.clip(abs(qa@qb),0,1))/dt)
            k=len(self.env.records);done=self.gate.update(k,d,angle,speed,angular)
            if self.env.records:
                self.env.records[-1]['observable_completion']=dict(distance_m=d,angle_rad=angle,speed_m_s=speed,measured_angular_speed_rad_s=angular,instantaneous_speed_m_s=raw_speed,instantaneous_angular_speed_rad_s=raw_angular,stability_source="successive_measured_pose_at_control_frequency",**self.gate.snapshot())
            return done,d,angle
        def completion_reached(self):
            if self.completion_mode=='legacy':return super().completion_reached()
            done,d,angle=self.completion_measure();k=len(self.env.records)
            # Stop commanding motion before asking whether a candidate is stable.
            # Every verification step consumes the unchanged native budget.
            if not done and d<=self.gate.distance_enter_m and angle<=self.gate.angle_enter_rad and k-self.last_probe>=8:
                self.last_probe=k
                q=np.asarray(serial(self.env.unwrapped.agent.robot.get_qpos()[0,:7]))
                for _ in range(min(6,200-k)):
                    self.env.phase='OBSERVABLE_SETTLING_PROBE'
                    self.env.step(np.r_[q,-1.]);self.probe_steps+=1
                    done,d,angle=self.completion_measure()
                    if done or d>self.gate.distance_exit_m or angle>self.gate.angle_exit_rad:break
            if done and self.hold_entry is None:self.hold_entry=len(self.env.records)
            return done
        def step(self, action):
            result=super().step(action)
            self.transitions[-1]['authority']=dict(mode=self.authority,family=self.task_family,requested=self.policy_requested.tolist(),applied=np.asarray(action).tolist())
            return result
        def save(self,out,label):
            self.extra_summary=dict(authority=self.authority,task_family=self.task_family,completion_mode=self.completion_mode,completion_gate=self.gate.snapshot(),hold_entry=self.hold_entry,settling_probe_steps=self.probe_steps,
                       ever_success=any(bool(np.asarray(v.get('native_success',False)).reshape(-1)[0]) for v in self.env.records))
            return super().save(out,label)
    return SharedEpisode(density,friction)

class BaselineEnv(SkillEnv):
    def __init__(self,*args,authority='full',completion='dwell',**kwargs):
        super().__init__(*args,episode_factory=make_episode,method='ln_sac',**kwargs)
        self.authority=authority;self.completion=completion;self.family='Y'
        self.observation_space=gym.spaces.Box(-10,10,(92,),dtype=np.float32)
    def context(self,obs):return np.r_[obs,[self.family=='Y',self.family=='YZ']].astype(np.float32)
    def reset(self,*,seed=None,options=None):
        opts=dict(options or {})
        if 'family' not in opts:
            self.family='YZ' if self.rng.random()<.5 else 'Y'
            opts['bias_mm']=[float(self.rng.choice([-1.2,0.,1.2])),0. if self.family=='Y' else float(self.rng.choice([-1.2,1.2]))]
        else:self.family=opts['family']
        self.episode.completion_mode=self.completion;self.episode.authority=self.authority;self.episode.task_family=self.family
        obs,info=super().reset(seed=seed,options=opts)
        return self.context(obs),info
    def step(self,action):
        requested=np.asarray(action,dtype=np.float32);applied=requested.copy()
        if self.authority=='y_only' or (self.authority=='task' and self.family=='Y'):applied[1]=0.
        self.episode.policy_requested=requested.copy()
        obs,reward,done,trunc,info=super().step(applied)
        return self.context(obs),reward,done,trunc,info


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--sb3-root',type=Path,required=True)
    p.add_argument('--mode',choices=['train','eval','controls'],default='train');p.add_argument('--authority',choices=['full','y_only','task'],default='full');p.add_argument('--completion',choices=['legacy','dwell'],default='dwell')
    p.add_argument('--steps',type=int,default=4096);p.add_argument('--seed',type=int,default=31101);p.add_argument('--seeds');p.add_argument('--model',type=Path);p.add_argument('--policies',default='geometry,search,rls');p.add_argument('--conditions',default='0,1,2,3,4,5');p.add_argument('--legacy-model',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
    import sys;sys.path.insert(0,str(a.sb3_root))
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from program_sac_policy import CriticLayerNormPolicy
    from program_adaptive_contact import SearchControl,RLSControl
    from program_authority_search import AuthoritySearchControl
    a.out.mkdir(parents=True,exist_ok=False);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    seeds=list(map(int,a.seeds.split(','))) if a.seeds else list(range(731000,731064)) if a.mode=='train' else list(range(732000,732006))
    files=['program_skill_baselines.py','program_completion_contract.py','program_authority_search.py','program_adaptive_rl.py','program_adaptive_contact.py','program_adaptive_task.py','program_contract_effect.py','program_sac_policy.py','program_charger.py','pull_tool_physics.py']
    source={}
    for n in files:
        b=(Path(__file__).parent/n).read_bytes();(a.out/n).write_bytes(b);source[n]=hashlib.sha256(b).hexdigest()
    (a.out/'MANIFEST.json').write_text(json.dumps(dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},conditions=CONDITIONS,seeds=seeds,sources=source,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),native_horizon=200,native_success_unchanged=True,claim='COMPONENT_STRONG_RL_BASELINE_NOT_MULTISKILL_METHOD',observation='same84 history +6zeros +known two-axis-family context; no actual bias labels',legal_actions='Y and Z offsets ±1.5mm; -2..4mm advance; permissions projected by declared mode',optimizer='SAC critic-LN alpha.02 lr3e-4 batch128 warmup256 gamma.99 tau.005'),indent=2))
    env=BaselineEnv(a.out,seeds,rng_seed=a.seed,evaluation=a.mode!='train',authority=a.authority,completion=a.completion)
    class Checkpoint(BaseCallback):
        def _on_step(self):
            if self.num_timesteps%2048==0:
                self.model.save(a.out/f'CHECKPOINT_{self.num_timesteps}.zip')
                (a.out/f'BUDGET_{self.num_timesteps}.json').write_text(json.dumps(env.counters(),indent=2))
            return True
    model=None;start=time.monotonic();error=None;previous_updates=0
    try:
        if a.mode=='train':
            if a.resume:
                model=SAC.load(a.resume,env=env,device='cpu');previous_updates=model._n_updates
                model.load_replay_buffer(a.resume.parent/'REPLAY_BUFFER.pkl')
            else:
                model=SAC(CriticLayerNormPolicy,env,policy_kwargs=dict(net_arch=[128,128]),learning_rate=3e-4,buffer_size=50000,learning_starts=256,batch_size=128,train_freq=1,gradient_steps=1,ent_coef=.02,gamma=.99,tau=.005,seed=a.seed,device='cpu',verbose=0)
            model.learn(total_timesteps=a.steps,callback=Checkpoint(),reset_num_timesteps=not bool(a.resume));env.finish_tail();model.save(a.out/'MODEL.zip');model.save_replay_buffer(a.out/'REPLAY_BUFFER.pkl')
        else:
            if a.mode=='eval':model=SAC.load(a.model,device='cpu')
            for seed in seeds:
                for ci in map(int,a.conditions.split(',')):
                    family,bias=CONDITIONS[ci]
                    for policy in (['model'] if a.mode=='eval' else a.policies.split(',')):
                        env.active_policy=policy
                        obs,_=env.reset(options=dict(native_seed=seed,family=family,bias_mm=bias))
                        control=AuthoritySearchControl(a.authority,family) if policy in {'search_unique','search_unique_fast'} else SearchControl() if policy=='search' else RLSControl() if policy=='rls' else lambda o:np.zeros(3)
                        done=False
                        while not done:
                            action=model.predict(obs[:90] if a.legacy_model else obs,deterministic=True)[0] if model is not None else control(obs[63:84])
                            # A deterministic stronger comparator uses the same
                            # positive-advance bound available to the learned actor.
                            # Retain any controller-requested retraction.
                            if policy=='search_unique_fast' and action[2]>=0:
                                action=np.asarray(action).copy();action[2]=1.
                            obs,_,done,_,_=env.step(action)
                        env.rows[-1]['policy']=policy;env.rows[-1]['condition_index']=ci
                        (a.out/'EPISODES.json').write_text(json.dumps(env.rows,indent=2))
    except BaseException as exc:error=repr(exc);raise
    finally:
        env.close();updates=int(model._n_updates)-previous_updates if model is not None and a.mode=='train' else 0
        (a.out/'RESULTS.json').write_text(json.dumps(dict(**env.counters(),optimizer_updates=updates,wall_s=time.monotonic()-start,error=error),indent=2))
if __name__=='__main__':main()
