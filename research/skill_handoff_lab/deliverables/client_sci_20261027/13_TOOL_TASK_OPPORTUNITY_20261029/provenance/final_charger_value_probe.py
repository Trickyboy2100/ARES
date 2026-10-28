"""Small matched-data skill-level Fitted Q probe, not a claimed new algorithm.

All actions share the same native200 executor. The margin auxiliary target is
measured terminal geometry; policy inputs never contain future outcomes.
"""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import sapien
from charger_probe import strong,pack_observation,ANGLES,DISTANCES
from pull_tool_physics import Recorder,Horizon,Failure,sp,serial

torch.set_num_threads(4)
class Net(nn.Module):
    def __init__(self,n):super().__init__();self.layers=nn.Sequential(nn.Linear(n,32),nn.Tanh(),nn.Linear(32,9))
    def forward(self,x):return self.layers(x).reshape(-1,3,3)

def collect(roots):
    data={}
    for root in roots:
        for r in json.loads((root/'RESULTS.json').read_text()):
            if not r['mode'].startswith('a'):continue
            with gzip.open(root/f"{r['seed']}_{r['mode']}.json.gz",'rt') as f:t=json.load(f)
            a=ANGLES.index(r['parameters']['angle']);c=DISTANCES.index(r['parameters']['standoff']);ds=r.get('decisions')
            if ds:
                obs=np.array(ds[0]['observation']);nxt=np.array(ds[1]['observation']) if len(ds)>1 and ds[1]['step']<200 else None
            else:
                obs=pack_observation(t['initial_measure'],0);lift=[s for s in t['steps'] if s['phase']=='LIFT'];align=[p for p in t['plans'] if p['phase']=='ALIGN'];nxt=pack_observation(lift[-1],lift[-1]['step'],ANGLES[a]) if lift and lift[-1]['grasped'] and align and align[0]['step']<200 else None
            final=r['final'];margin=np.clip([1-float(final['obj_to_goal_dist'][0])/.005,1-float(final['obj_to_goal_angle'][0])/.2],-5,1)
            data[r['seed'],a,c]=dict(obs=obs,next=nxt,y=float(r['success']),margin=margin)
    seeds=sorted({s for s,a,c in data});assert all((s,a,c) in data for s in seeds for a in range(3) for c in range(3));return data,seeds

def fit(root):
    out=root/'learning';out.mkdir(exist_ok=False);data,seeds=collect([root/'charger_design4',root/'charger_train8']);rows=[[],[]]
    for s in seeds:
        for a in range(3):
            for c in range(3):
                z=data[s,a,c]
                rows[0].append(dict(obs=z['obs'],action=a,y=z['y'],next=z['next'],margin=[0,0]))
                if z['next'] is not None:rows[1].append(dict(obs=z['next'],action=c,y=z['y'],next=None,margin=z['margin']))
    prefix_differences=[];prefix_availability_mismatches=0
    for seed in seeds:
        for action in range(3):
            next_states=[data[seed,action,c]['next'] for c in range(3)]
            prefix_availability_mismatches+=int(len({v is None for v in next_states})>1)
            valid=[v for v in next_states if v is not None]
            if valid:prefix_differences.append(max(float(np.max(abs(v-valid[0]))) for v in valid))
    raw=np.stack([r['obs'] for rs in rows for r in rs]);mean=raw.mean(0);std=np.maximum(raw.std(0),.03);n=len(mean);models={}
    for mode in ['supervised','fitted_q','margin_fitted_q']:
        torch.manual_seed(20261029);nets=[Net(n),Net(n)];losses={}
        for st in [1,0]:
            rr=rows[st];x=torch.tensor((np.stack([r['obs'] for r in rr])-mean)/std,dtype=torch.float32);act=torch.tensor([r['action'] for r in rr]);y=torch.tensor([r['y'] for r in rr],dtype=torch.float32);aux=torch.tensor(np.array([r['margin'] for r in rr]),dtype=torch.float32)
            if st==0 and mode!='supervised':
                with torch.no_grad():y=torch.tensor([float(nets[1](torch.tensor((r['next']-mean)/std,dtype=torch.float32)[None])[0,:,0].sigmoid().max()) if r['next'] is not None else 0. for r in rr])
            opt=torch.optim.Adam(nets[st].parameters(),lr=.002,weight_decay=.01)
            for epoch in range(300):
                pred=nets[st](x)[torch.arange(len(rr)),act];loss=nn.functional.binary_cross_entropy_with_logits(pred[:,0],y)
                if st==1 and mode=='margin_fitted_q':loss=loss+.2*nn.functional.smooth_l1_loss(pred[:,1:],aux)
                opt.zero_grad();loss.backward();opt.step()
            losses[str(st)]=float(loss)
        path=out/f'{mode}.pt';torch.save(dict(nets=[net.state_dict() for net in nets],mean=mean,std=std,n=n,mode=mode,train_seeds=seeds,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),path);models[mode]=losses
    totals={(a,c):sum(data[s,a,c]['y'] for s in seeds) for a in range(3) for c in range(3)};best=max(totals,key=lambda k:(totals[k],k==(1,1)))
    meta=dict(prefix_availability_mismatches=prefix_availability_mismatches,prefix_backup="each actual rollout; no assumed identical prefix",prefix_max_difference=max(prefix_differences,default=0),training_cases=len(seeds),physical_grid_episodes=len(data),stage_tokens=[len(rs) for rs in rows],best_fixed=list(best),fixed_successes=totals[best],finite_observed_union=sum(max(data[s,a,c]['y'] for a in range(3) for c in range(3)) for s in seeds),losses=models,training_epochs=300,seed=20261029,train_scope='12 design cases, no development cases',auxiliary='measured terminal position / angle tolerance margins; no future input at deployment')
    (out/'TRAINING.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))

def learned(path):
    b=torch.load(path,weights_only=False);nets=[Net(b['n']),Net(b['n'])]
    for m,s in zip(nets,b['nets']):m.load_state_dict(s);m.eval()
    def policy(o,st):
        with torch.no_grad():q=nets[st](torch.tensor((o-b['mean'])/b['std'],dtype=torch.float32)[None])[0,:,0].sigmoid().numpy()
        return int(q.argmax()),dict(scores=q.tolist())
    return policy

def evaluate(root):
    out=root/'charger_evaluation8';out.mkdir(exist_ok=False);meta=json.loads((root/'learning/TRAINING.json').read_text());fixed=meta['best_fixed'];env=Recorder(gym.make('PlugCharger-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));rows=[]
    def geometry(o,st):
        if st==0:return fixed[0],dict(rule='training-best grasp')
        u=env.unwrapped;p=env.active_planner;attached=sp(u.agent.tcp).inv()*sp(u.charger);costs=[];start=time.monotonic()
        for d in DISTANCES:
            target=u.goal_pose.sp*sapien.Pose([-d,0,0])*attached.inv();res=p.planner.plan_screw(np.r_[target.p,target.q],np.array(serial(u.agent.robot.get_qpos()[0])),time_step=u.control_timestep,use_point_cloud=False);costs.append(len(res['position']) if res['status']=='Success' else 10000)
        env.plans.append(dict(phase='POLICY_SEARCH',step=len(env.records),planning_wall_s=time.monotonic()-start,costs=costs));return int(np.argmin(costs)),dict(rule='minimum planned approach cost',costs=costs)
    policies=[('fixed_grid_search',lambda o,st:(fixed[st],{})),('geometric_search',geometry)]+[(m,learned(root/'learning'/f'{m}.pt')) for m in ['supervised','fitted_q','margin_fitted_q']]
    try:
        for seed in range(723000,723008):
            digest=None
            for name,policy in policies:
                env.video=None;env.progress=out/f'{seed}_{name}.progress.json';started=time.monotonic();error=None
                try:strong(env,seed,policy=policy)
                except (Horizon,Failure) as e:error=str(e)
                if digest is None:digest=env.initial_sha
                assert digest==env.initial_sha
                if len(env.records)<200:
                    q=np.array(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));env.phase='TERMINAL_HOLD'
                    grip=-1 if any(z['phase']=='GRASP' for z in env.records) else 1
                    while len(env.records)<200:env.step(np.r_[q,grip])
                row=dict(seed=seed,mode=name,success=env.records[-1]['native_success'],error=error,initial_sha256=env.initial_sha,steps=len(env.records),wall_s=time.monotonic()-started,planning_wall_s=sum(p['planning_wall_s'] for p in env.plans),decisions=env.decisions,final=serial(env.unwrapped.evaluate()))
                path=out/f'{seed}_{name}.json.gz'
                with gzip.open(path,'wt') as f:json.dump(dict(summary=row,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans),f,separators=(',',':'))
                row['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(row);(out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:row[k] for k in ['seed','mode','success','error','wall_s']}),flush=True)
    finally:env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['fit','eval'],required=True);a=p.parse_args();fit(a.root) if a.phase=='fit' else evaluate(a.root)
if __name__=='__main__':main()
