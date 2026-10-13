"""Small real-data cost experiment reusing cache/Gym/CASH update implementations.

Not long-horizon RL or physical success learning. Includes order randomization,
validation-only curves, final held-out scene evaluation and strong joint-greedy.
"""
import argparse,json,time,hashlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from sb3_contrib import MaskablePPO
from research.skill_handoff_lab.cash.cache_backend import CachedPlanningBackend,observation
from research.skill_handoff_lab.cash.gym_env import GymSkillHandoffEnv
from research.skill_handoff_lab.cash.model import CashActorCritic
from research.skill_handoff_lab.cash.trainer import CashBatch,cash_update,save_checkpoint

class PermutedBackend(CachedPlanningBackend):
    def reset(self,seed):
        obs=super().reset(seed);order=np.random.default_rng(seed+317).permutation(len(obs.actions))
        return replace(obs,actions=tuple(obs.actions[i] for i in order))

def tensors(obs):
    state=torch.tensor([obs.robot_state+obs.remaining_goal+obs.successor_constraints],dtype=torch.float32)
    features=torch.tensor([[a.features for a in obs.actions]],dtype=torch.float32)
    return state,features,torch.ones((1,len(obs.actions)),dtype=torch.bool)

def candidates(episode,reverse=False):
    o=observation(episode['observations']['root'])
    return replace(o,actions=o.actions[::-1]) if reverse else o

def outcomes(episode,obs):
    records={r['action_id']:r for r in episode['transitions']}
    return [records[a.action_id] for a in obs.actions]

def evaluate(episodes,ranker,normalization,ppo,cash):
    result=[]
    for e in episodes:
      for reverse in [False,True]:
        obs=candidates(e,reverse);s,x,mask=tensors(obs);rows=outcomes(e,obs)
        costs=np.array([r['execution_cost'] for r in rows]);fx=torch.cat((s[:,None,:].expand(-1,2,-1),x),dim=-1)
        norm=(fx-normalization[0])/normalization[1]
        with torch.no_grad():
            ri=int(ranker(norm).squeeze().argmin());ci=int(cash(s,x,mask)['logits'].argmax())
        packed={'state':s.numpy()[0],'action_features':x.numpy()[0],'stage':np.array([0,0,1],np.float32)}
        pi=int(ppo.predict(packed,deterministic=True,action_masks=mask.numpy()[0])[0])
        selected={'fixed_NORMAL':next(i for i,a in enumerate(obs.actions) if a.features[12]==0),
            'joint_greedy':int(np.argmin(x.numpy()[0,:,-1])),'ranker_cost':ri,'MaskablePPO':pi,'CASH_aux':ci,
            'oracle_K2_upper_bound':int(costs.argmin())}
        for method,index in selected.items():
            result.append({'group':e['scene_group'],'order_reversed':reverse,'method':method,
                'action_id':obs.actions[index].action_id,'cost_joint_rad':float(costs[index]),
                'regret_joint_rad':float(costs[index]-costs.min()),'planning_calls_budget':2 if method.startswith('oracle') else 1,
                'model_qualified_path':True,'physical_success':None})
        result.append({'group':e['scene_group'],'order_reversed':reverse,'method':'random_expected',
            'action_id':None,'cost_joint_rad':float(costs.mean()),'regret_joint_rad':float(costs.mean()-costs.min()),
            'planning_calls_budget':1,'model_qualified_path':True,'physical_success':None})
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int,default=2048)
    p.add_argument('--effect-coef',type=float,default=.1);a=p.parse_args()
    data=json.loads(a.dataset.read_text())
    if data.get('scope')!='ONE_STEP_PLANNING_COST_ONLY_CONDITIONAL_ON_ALL_OPTIONS_QUALIFIED':raise ValueError('wrong experiment scope')
    backends={s:CachedPlanningBackend(a.dataset,s) for s in ['train','validation','test']}
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    train=backends['train'].episodes;validation=backends['validation'].episodes;test=backends['test'].episodes
    # Supervised regression of measured cost, not BCE with invented failed tasks.
    features=[];labels=[]
    for e in train:
        o=candidates(e);s,x,_=tensors(o)
        features.extend(torch.cat((s.expand(2,-1),x[0]),dim=-1).tolist())
        labels.extend(r['execution_cost'] for r in outcomes(e,o))
    fx=torch.tensor(features);y=torch.tensor(labels);mean=fx.mean(0);scale=fx.std(0,unbiased=False).clamp_min(.05)
    ranker=torch.nn.Sequential(torch.nn.Linear(fx.shape[1],64),torch.nn.Tanh(),torch.nn.Linear(64,1))
    opt=torch.optim.Adam(ranker.parameters(),lr=1e-3);rank_curve=[]
    for epoch in range(160):
        loss=torch.nn.functional.mse_loss(ranker((fx-mean)/scale).squeeze(-1),y)
        opt.zero_grad();loss.backward();opt.step()
        vf=[];vy=[]
        for e in validation:
            o=candidates(e);s,x,_=tensors(o);vf.extend(torch.cat((s.expand(2,-1),x[0]),-1).tolist());vy.extend(r['execution_cost'] for r in outcomes(e,o))
        with torch.no_grad():vl=torch.nn.functional.mse_loss(ranker((torch.tensor(vf)-mean)/scale).squeeze(-1),torch.tensor(vy))
        rank_curve.append({'epoch':epoch,'train_mse':float(loss.detach()),'validation_mse':float(vl)})
    torch.save({'model':ranker.state_dict(),'mean':mean,'scale':scale,'seed':a.seed},a.output/'ranker_cost.pt')
    dims=data['dimensions']
    def make_env():return GymSkillHandoffEnv(PermutedBackend(a.dataset,'train'),**dims)
    env=DummyVecEnv([make_env,make_env]);ppo=MaskablePPO('MultiInputPolicy',env,device='cpu',seed=a.seed,
        n_steps=64,batch_size=64,n_epochs=4,policy_kwargs={'net_arch':[64,64]},verbose=0)
    ppo.set_logger(configure(str(a.output/'ppo'),['csv']))
    started=time.monotonic();ppo.learn(total_timesteps=a.steps);ppo_time=time.monotonic()-started
    ppo.save(a.output/'maskable_ppo');env.close()
    torch.manual_seed(a.seed);cash=CashActorCritic(dims['state_dim'],dims['action_dim'],hidden=64,effect_dim=6)
    optimizer=torch.optim.Adam(cash.parameters(),lr=3e-4);cash_curve=[];started=time.monotonic()
    # Horizon-one returns: terminal reward is exact, no fabricated multi-stage edge.
    for update in range(a.steps//64):
        ss=[];xx=[];aa=[];lp=[];rr=[];vv=[];effects=[]
        for _ in range(64):
            e=train[int(rng.integers(len(train)))];obs=candidates(e,bool(rng.integers(2)));s,x,mask=tensors(obs)
            with torch.no_grad():
                out=cash(s,x,mask);dist=Categorical(logits=out['logits']);action=dist.sample()
            row=outcomes(e,obs)[int(action)];reward=1.-row['execution_cost']
            ss.append(s[0]);xx.append(x[0]);aa.append(action[0]);lp.append(dist.log_prob(action)[0]);rr.append(reward);vv.append(out['value'][0]);effects.append(row['metadata']['successor_effect_target'])
        returns=torch.tensor(rr);advantage=returns-torch.stack(vv);advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        batch=CashBatch(torch.stack(ss),torch.stack(xx),torch.ones((64,2),dtype=torch.bool),torch.stack(aa),
            torch.stack(lp),advantage,returns,torch.tensor(effects))
        for _ in range(4):losses=cash_update(cash,optimizer,batch,effect_coef=a.effect_coef)
        val_returns=[]
        for e in validation:
            o=candidates(e);s,x,mask=tensors(o)
            with torch.no_grad():index=int(cash(s,x,mask)['logits'].argmax())
            val_returns.append(1.-outcomes(e,o)[index]['execution_cost'])
        cash_curve.append({'steps':(update+1)*64,'train_mean_return':float(returns.mean()),
            'validation_mean_return':float(np.mean(val_returns)),**losses})
    cash_time=time.monotonic()-started
    metadata={'scope':data['scope'],'seed':a.seed,'steps':len(cash_curve)*64,'effect_coef':a.effect_coef,
        'effect_targets':['model_path_valid','TCP_length_m','joint_length_rad','clearance_m','joint_margin_rad','orientation_error_over_3deg'],
        'physical_simulation':False,'full_chassis_safety_qualified':False}
    save_checkpoint(a.output/'cash_aux.pt',cash,optimizer,metadata)
    evaluation=evaluate(test,ranker,(mean,scale),ppo,cash);summary=[]
    if a.effect_coef==0:
        for row in evaluation:
            if row['method']=='CASH_aux':row['method']='CASH_no_aux'
    for method in sorted({r['method'] for r in evaluation}):
        rows=[r for r in evaluation if r['method']==method];agreements=[]
        for group in {r['group'] for r in rows}:
            pair=[r for r in rows if r['group']==group];agreements.append(pair[0]['action_id']==pair[1]['action_id'])
        summary.append({'method':method,'test_groups':len(rows)//2,'mean_cost_joint_rad':float(np.mean([r['cost_joint_rad'] for r in rows])),
            'mean_regret_joint_rad':float(np.mean([r['regret_joint_rad'] for r in rows])),
            'order_choice_agreement':float(np.mean(agreements)),'budget':rows[0]['planning_calls_budget']})
    manifest=dict(metadata,dataset_sha256=hashlib.sha256(a.dataset.read_bytes()).hexdigest(),
        train_groups=[e['scene_group'] for e in train],validation_groups=[e['scene_group'] for e in validation],test_groups=[e['scene_group'] for e in test],
        ppo_steps=ppo.num_timesteps,ppo_wall_s=ppo_time,cash_wall_s=cash_time,
        cash_steps_per_s=len(cash_curve)*64/cash_time,ranker_labels=len(labels),
        oracle_calls_to_collect=len(data['episodes'])*2,online_oracle_calls_during_training=0,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),
            Path(__file__).parents[1]/'cash/model.py',Path(__file__).parents[1]/'cash/trainer.py']},
        runtime={'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','threads':2},
        causal_scope='Offline candidate evaluation bandit: no verified preparation edge between different held start states',
        caveat='tiny all-feasible E0 cost benchmark; no HSR improvement or industrial generalization claim')
    for name,value in [('ranker_curve',rank_curve),('cash_curve',cash_curve),('evaluation',evaluation),('comparison',summary),('manifest',manifest)]:
        (a.output/(name+'.json')).write_text(json.dumps(value,indent=2))
    print(json.dumps({'manifest':manifest,'comparison':summary}))
if __name__=='__main__':main()
