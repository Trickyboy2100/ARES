"""First honest planning-cache Option-RL pilot, NOT PhysX-online training."""
import argparse,dataclasses,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from research.skill_handoff_lab.cash.option_backend import PlanningOptionBackend
from research.skill_handoff_lab.cash.gym_env import GymSkillHandoffEnv
from controlled_paths import require_private_location


def make_env(dataset,split,seed,permute=True):
    backend=PlanningOptionBackend(dataset,split,permute=permute)
    sample=backend.reset(seed)
    dim=len(sample.robot_state+sample.remaining_goal+sample.successor_constraints)
    return GymSkillHandoffEnv(backend,dim,len(sample.actions[0].features),4,allow_planning=True,cost_weight=.005)


def evaluate(dataset,method,policy=None,seed=2027,episodes=100):
    env=make_env(dataset,'test',seed);rows=[]
    for index in range(episodes):
        obs,_=env.reset(seed=seed+index);backend=env.core.backend;begin=len(backend.logs);terminated=truncated=False
        while not (terminated or truncated):
            current=env.core.current;actions=current.actions;mask=env.action_masks()
            if method=='ppo':action=int(policy.predict(obs,action_masks=mask,deterministic=True)[0])
            elif method=='random':action=int(np.random.default_rng(seed+index+len(backend.logs)).choice(np.flatnonzero(mask)))
            elif method=='greedy':action=min(np.flatnonzero(mask),key=lambda i:actions[i].features[-2])
            elif method=='fixed_direct':
                action=next(i for i,a in enumerate(actions) if a.features[0] and a.features[3])
            elif method=='equal_budget_retry':
                # Calls are genuinely charged one by one; no free lookahead.
                desired='FAST' if not backend.history else 'FALLBACK'
                action=next(i for i,a in enumerate(actions) if a.features[0] and bool(a.features[3])==(desired=='FAST'))
            else:raise ValueError(method)
            obs,reward,terminated,truncated,info=env.step(action)
        logs=backend.logs[begin:];result=info['result']
        rows.append({'method':method,'episode':index,'task_id':backend.task['task_id'],'success':result.success,
                     'calls':len(logs),'attempt_cap_sum':sum(x['attempt_cap'] for x in logs),
                     'planned_option_time_s':sum(x['option_duration_s'] for x in logs),
                     'decisions':list(result.decisions),'truncated':truncated,'trajectory':logs})
    return rows


def main(a):
    require_private_location(a.output);os.umask(0o077);a.output.mkdir(mode=0o700,exist_ok=False)
    torch.set_num_threads(2);data=json.loads(a.dataset.read_text());all_results=[]
    for baseline in ['fixed_direct','random','greedy','equal_budget_retry']:
        all_results+=evaluate(a.dataset,baseline,episodes=a.eval_episodes)
    for seed in a.seeds:
        torch.manual_seed(seed);np.random.seed(seed)
        env=DummyVecEnv([lambda s=seed+i:make_env(a.dataset,'train',s) for i in range(2)])
        policy=MaskablePPO('MultiInputPolicy',env,seed=seed,device='cpu',n_steps=128,batch_size=128,
            n_epochs=6,gamma=1.,gae_lambda=1.,learning_rate=3e-4,ent_coef=.015,
            policy_kwargs={'net_arch':[64,64]},verbose=0)
        before={k:v.detach().clone() for k,v in policy.policy.state_dict().items()}
        curves=[]
        class Callback(BaseCallback):
            def _on_step(self):
                for i,info in enumerate(self.locals['infos']):
                    if 'result' in info:
                        r=info['result'];curves.append({'steps':self.num_timesteps,'success':r.success,'cost':r.total_cost,
                            'decisions':len(r.decisions),'budget_exhausted':info['transition'].metadata.get('terminal_reason')=='BUDGET_EXHAUSTED'})
                return True
        start=time.perf_counter();policy.learn(total_timesteps=a.steps,callback=Callback())
        changed=any(not torch.equal(before[k],v) for k,v in policy.policy.state_dict().items())
        policy.save(a.output/f'policy_seed{seed}.zip')
        results=evaluate(a.dataset,'ppo',policy=policy,episodes=a.eval_episodes)
        for row in results:row['seed']=seed
        all_results+=results
        (a.output/f'learning_seed{seed}.json').write_text(json.dumps(curves,indent=2))
        logs=[row for e in env.envs for row in e.core.backend.logs]
        import gzip
        with gzip.open(a.output/f'training_actions_seed{seed}.json.gz','wt') as stream:json.dump(logs,stream)
        (a.output/f'training_manifest_seed{seed}.json').write_text(json.dumps({'seed':seed,'steps':policy.num_timesteps,
            'gradient_updates':policy._n_updates*(policy.n_steps*policy.n_envs//policy.batch_size),
            'training_epoch_updates':policy._n_updates,'parameters_changed':changed,'wall_s':time.perf_counter()-start,
            'training_episodes':len(curves),'multi_decision_episodes':sum(r['decisions']>=2 for r in curves),
            'evidence':'planning-only real cuRobo cache, NOT PhysX-online or real-robot',
            'discount':'gamma=1, lambda=1; finite-budget undiscounted SMDP, explicit measured solve + planned duration cost',
            'new_online_planning_calls':0,'cache_step_count':len(logs),'dataset_sha256':hashlib.sha256(a.dataset.read_bytes()).hexdigest()},indent=2))
        print('trained seed',seed,'changed',changed,'test',sum(r['success'] for r in results),'/',len(results),flush=True)
        env.close()
    (a.output/'FROZEN_EVALUATION.json').write_text(json.dumps(all_results,indent=2))
    summaries=[]
    for name in ['fixed_direct','random','greedy','equal_budget_retry','ppo']:
        selected=[r for r in all_results if r['method']==name]
        summaries.append({'method':name,'successes':sum(r['success'] for r in selected),'episodes':len(selected),
            'success_rate':float(np.mean([r['success'] for r in selected])),
            'mean_calls':float(np.mean([r['calls'] for r in selected])),
            'mean_option_time_s':float(np.mean([r['planned_option_time_s'] for r in selected]))})
    (a.output/'COMPARISON.json').write_text(json.dumps(summaries,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for seed in a.seeds:
        curve=json.loads((a.output/f'learning_seed{seed}.json').read_text())
        x=np.array([c['steps'] for c in curve]);y=np.array([c['success'] for c in curve],float)
        if len(y)>=30:
            axes[0].plot(x[29:],np.convolve(y,np.ones(30)/30,mode='valid'),label=f'seed {seed}')
            axes[1].plot(x[29:],np.convolve([c['cost'] for c in curve],np.ones(30)/30,mode='valid'),label=f'seed {seed}')
    axes[0].set(xlabel='real-cache Option interactions',ylabel='rolling success (30 episodes)',title='Actual PPO training, not PhysX online');axes[0].legend()
    axes[1].set(xlabel='cache Option interactions',ylabel='rolling measured/planned cost (s + call penalty)',title='Actual learning objective')
    axes[2].bar([s['method'] for s in summaries],[s['success_rate'] for s in summaries]);axes[2].tick_params(axis='x',rotation=30);axes[2].set(ylabel='within-budget completion',ylim=(0,1.05))
    axes[3].bar([s['method'] for s in summaries],[s['mean_calls'] for s in summaries]);axes[3].tick_params(axis='x',rotation=30);axes[3].set(ylabel='charged planner calls / episode')
    fig.suptitle('CONTROLLED | 6 station-derived cases, 4 train / 2 test; NOT unseen-workstation evidence')
    fig.tight_layout();fig.savefig(a.output/'SKILL_RL_LEARNING_AND_BASELINES.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8192);p.add_argument('--seeds',type=int,nargs='+',default=[31,41,51]);p.add_argument('--eval-episodes',type=int,default=100)
    main(p.parse_args())
