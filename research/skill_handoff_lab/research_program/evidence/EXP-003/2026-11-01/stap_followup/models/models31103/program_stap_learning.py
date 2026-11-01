"""Bounded real-data two-skill learners: STAP-style, fitted Q and one auxiliary.
All labels come from actual trajectories. Contract relabeling supplies geometric
auxiliary labels ONLY, never unexecuted successor reward. Finite-horizon gamma=1.
Foresight is an explicitly state-based fixed-nominal-suffix analogue, not VLA.
"""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from program_stap_kernel import Agent,Box,Critic,Primitive,port_plan

# Fixed common observation: no future physical state or native-success input.
def features(m,remaining,density,target_y,stage):
    return np.array([*m['q'][:7],*m['qvel'][:7],*m['tool'],*m['cube'],*m['tcp_T_tool'],*m['tool_linear_velocity'],*m['tool_angular_velocity'],remaining/100,density,target_y,stage],np.float32)
# indices: q0:7, qvel7:14, tool14:21,cube21:28,relative28:35,vel35:41,budget41,mass42,target43,stage44

def contract_labels(z,goal):
    from transforms3d.quaternions import quat2mat
    result=[]
    for s,y in zip(z,goal):
        cube=s[21:24];rel=s[28:35];target=np.array([-.615+.57,float(y)])
        angle=np.arctan2(cube[1]-target[1],cube[0]-target[0]);c,si=np.cos(angle),np.sin(angle);R=np.array([[c,-si,0],[si,c,0],[0,0,1.]])
        origin=cube-R@np.array([.126,.065,0]);origin[2]=.08
        tcp=origin+R@(-quat2mat(rel[3:]) .T@rel[:3])
        result.append([s[41],(.85-np.linalg.norm(tcp-np.array([-.615,0,0])))/.85,(.15-rel[0])/.15,np.cos(angle-2*np.arctan2(s[20],s[17]))])
    return np.array(result,np.float32)

class Net(nn.Module):
    def __init__(self,n,out=1):
        super().__init__();self.body=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,64),nn.ReLU());self.head=nn.Linear(64,out)
    def forward(self,x):return self.head(self.body(x))

class Model:
    def __init__(self,payload):
        self.payload=payload;self.mean=np.array(payload['mean'],np.float32);self.std=np.array(payload['std'],np.float32);self.nets={}
        for name,state in payload['weights'].items():
            out=45 if name=='dynamics' else 5 if name in ['fq','contract','generic','foresight'] else 1
            net=Net(46 if name not in ['actor0','actor1','value'] else 45,out);net.load_state_dict(state);net.eval();self.nets[name]=net
    def norm(self,s):return (s-torch.as_tensor(self.mean))/torch.as_tensor(self.std)
    def value(self,name,s,a=None):
        x=self.norm(s);x=x if a is None else torch.cat([x,a],-1);return self.nets[name](x)
    def q(self,stage,s,a):return torch.sigmoid(self.value('q'+str(stage),s,a)[:,0])
    def next(self,s,a):
        z=self.value('dynamics',s,a)*torch.as_tensor(self.std)+torch.as_tensor(self.mean)
        z[:,42:44]=s[:,42:44];z[:,44]=1.;return z
    @classmethod
    def load(cls,path):return cls(torch.load(path,map_location='cpu',weights_only=False))

class Dynamics:
    def __init__(self,m):self.m=m;self.action_space=Box([-1],[1]);self.device=torch.device('cpu')
    def decode(self,s,primitive=None):return s
    def rollout(self,obs,skeleton,policies,batch_size=None,time_index=False):
        n=batch_size or 1;s=obs.unsqueeze(0).repeat(n,1);ss=[s];aa=[]
        for t,p in enumerate(skeleton):
            policy=policies[t if time_index else p.idx_policy];a=policy.actor.predict(s);aa.append(a)
            s=self.m.next(s,a) if p.idx_policy==0 else s.clone();ss.append(s)
        ss=torch.stack(ss,1);aa=torch.stack(aa,1);return (ss,aa) if batch_size else (ss[0],aa[0])

class Actor:
    def __init__(self,m,stage):self.m=m;self.stage=stage
    def predict(self,s):return torch.tanh(self.m.value('actor'+str(self.stage),s))

class Selector:
    def __init__(self,path,method,samples=16):self.model=Model.load(path);self.method=method;self.samples=samples;self.logs=[]
    def choose(self,env,stage):
        obs=features(env.measure(),100-len(env.records),env.unwrapped.density_scale,env.target_y,stage);start=time.monotonic();m=self.model
        with torch.no_grad():
            if self.method=='stap':
                policies=[Agent(Box([-1],[1]),Actor(m,k),Critic(lambda s,a,k=k:m.q(k,s,a))) for k in [0,1]]
                plan=port_plan(obs,[Primitive(k) for k in range(stage,2)],policies,Dynamics(m),num_iterations=4,num_samples=self.samples,num_elites=4,keep_elites_fraction=.25,population_decay=.8)
                action=float(plan['actions'][0,0]);log={k:plan[k] for k in ['p_success','trajectory_queries','skill_model_queries','iterations']};log['predicted_states']=plan['states'].tolist();log['planned_actions']=plan['actions'].tolist()
            elif stage==1 and self.method=='foresight':
                action=None;log=dict(selection='unchanged strong geometric suffix used for actual predictor labels',skill_model_queries=0)
            else:
                grid=torch.linspace(-1,1,33).reshape(-1,1);states=torch.from_numpy(obs).repeat(33,1)
                vals=m.q(1,states,grid) if stage==1 else torch.sigmoid(m.value(self.method,states,grid)[:,0])
                idx=int(vals.argmax());action=float(grid[idx,0]);log=dict(skill_model_queries=33,predicted_value=float(vals[idx]),all_values=vals.tolist(),selection='bounded learned value grid; same33 candidates for fitted-Q variants')
        log.update(stage=stage,step=len(env.records),method=self.method,normalized_action=action,wall_s=time.monotonic()-start,observation=obs.tolist());self.logs.append(log)
        return None if action is None else (.0875+.0175*action if stage==0 else -.065+.01*action)


def collect(root):
    rows=[]
    for path in sorted(root.rglob('TRAJECTORY.json.gz')):
        d=json.load(gzip.open(path,'rt'));r=d['summary'];manifest=json.loads((path.parent/'MANIFEST.json').read_text())
        if not manifest.get('target_feedback') or r.get('target_y') is None:continue
        first=d['decisions'][0];stable=r['stability'];boundary=(stable['checks'][-1]['step'] if stable and stable['checks'] else 0)
        if not boundary:continue
        post=d['steps'][boundary-1];density=r['density_scale'];goal=r['target_y'];a0=(r['grasp_x']-.0875)/.0175
        a1=(d['decisions'][1]['parameters']['hook_y_m']+.065)/.01 if len(d['decisions'])>1 else 0.
        rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=r['seed'],density=density,goal=goal,s0=features(d['initial_measure'],100,density,goal,0).tolist(),s1=features(post,100-boundary,density,goal,1).tolist(),a0=a0,a1=a1,r0=float(stable['pass_']),r1=float(r['task_success']),has_suffix=len(d['decisions'])>1,auto_suffix=manifest['hook_y'] is None))
    return rows


def train(root,out,seed,updates,variants):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed);rows=collect(root);assert rows
    s0=torch.tensor([x['s0'] for x in rows]);s1=torch.tensor([x['s1'] for x in rows]);a0=torch.tensor([[x['a0']] for x in rows]);a1=torch.tensor([[x['a1']] for x in rows]);r0=torch.tensor([[x['r0']] for x in rows]);r1=torch.tensor([[x['r1']] for x in rows]);suffix=torch.tensor([x['has_suffix'] for x in rows]);auto=torch.tensor([x['has_suffix'] and x['auto_suffix'] for x in rows]);assert auto.any() and suffix.any()
    all_s=torch.cat([s0,s1]);mean=all_s.mean(0);std=all_s.std(0).clamp(min=.05);norm=lambda s:(s-mean)/std
    nets={name:Net(46,45 if name=='dynamics' else 1) for name in ['q0','q1','dynamics']};nets['value']=Net(45)
    opt=torch.optim.Adam([p for net in nets.values() for p in net.parameters()],lr=.001);bce=nn.BCEWithLogitsLoss();logs=[]
    for i in range(updates):
        q0=nets['q0'](torch.cat([norm(s0),a0],-1));q1=nets['q1'](torch.cat([norm(s1[suffix]),a1[suffix]],-1));z=nets['dynamics'](torch.cat([norm(s0),a0],-1));v=nets['value'](norm(s1[auto]))
        loss=bce(q0,r0)+bce(q1,r1[suffix])+((z-norm(s1))**2).mean()+bce(v,r1[auto]);opt.zero_grad();loss.backward();opt.step()
        if i%100==0:logs.append(dict(phase='common_models',step=i,loss=float(loss)))
    with torch.no_grad():
        grid=torch.linspace(-1,1,33);states=norm(s1).repeat_interleave(33,0);actions=grid.repeat(len(rows)).reshape(-1,1);fqtarget=r0*torch.sigmoid(nets['q1'](torch.cat([states,actions],-1))).reshape(len(rows),33).max(1).values[:,None]
        vstarget=r0*torch.sigmoid(nets['value'](norm(s1)))
    # Same rows and actor/Q capacity. Auxiliary targets cannot change rewards.
    alt=s0.repeat(2,1);alt[:len(rows),43]=-.08;alt[len(rows):,43]=.08;az=s1.repeat(2,1);aa=a0.repeat(2,1)
    contract=torch.tensor(contract_labels(az.numpy(),alt[:,43].numpy()));generic=az[:,[28,29,30,41]]
    aux_stats={}
    for name in variants:
        torch.manual_seed(seed+100);net=Net(46,5);nets[name]=net;optim=torch.optim.Adam(net.parameters(),lr=.001);label=vstarget if name=='foresight' else fqtarget
        aux=contract if name=='contract' else generic;am=aux.mean(0);asd=aux.std(0).clamp(min=.05);aux_stats[name]={'mean':am.tolist(),'std':asd.tolist()}
        for i in range(updates):
            pred=net(torch.cat([norm(s0),a0],-1));loss=bce(pred[:,:1],label)
            if name in ['contract','generic']:
                auxpred=net(torch.cat([norm(alt),aa],-1))[:,1:];loss=loss+.1*((auxpred-(aux-am)/asd)**2).mean()
            optim.zero_grad();loss.backward();optim.step()
            if i%100==0:logs.append(dict(phase=name,step=i,loss=float(loss)))
    # Learned policies initialize STAP, not a hand-selected CEM mean.
    for k,states in [(0,s0),(1,s1[suffix])]:
        actor=Net(45);nets['actor'+str(k)]=actor;optim=torch.optim.Adam(actor.parameters(),lr=.001)
        for i in range(200):
            act=torch.tanh(actor(norm(states)));value=nets['q'+str(k)](torch.cat([norm(states),act],-1));loss=-torch.sigmoid(value).mean()+.01*(act**2).mean();optim.zero_grad();loss.backward();optim.step()
    payload=dict(mean=mean.tolist(),std=std.tolist(),weights={n:v.state_dict() for n,v in nets.items()},seed=seed,aux_stats=aux_stats);torch.save(payload,out/'MODEL.pt')
    (out/'DATASET.json').write_text(json.dumps(rows,indent=2));(out/'TRAINING.json').write_text(json.dumps(dict(seed=seed,episodes=len(rows),shared_recorded_physics_steps=100*len(rows),fresh_physics_steps=0,suffix_rows=int(suffix.sum()),auto_suffix_rows=int(auto.sum()),successes=int(r1.sum()),optimizer_updates=(1+len(variants))*updates+400,updates_per_value_phase=updates,actor_initialization_updates=400,gamma=1.,objective='undiscounted full-task success at native horizon, remaining time observable',logs=logs,scope='Bounded offline fitted-Q and mechanism baselines; not SAC/PPO, not converged official STAP checkpoints. Models share exact physical dataset. Foresight fixed-geometric-suffix predictor uses only its actually executed auto-suffix rows; no imagined success labels. Contract relabeling only auxiliary loss; generic auxiliary and ordinary Q are ablations.',dataset_sha256=hashlib.sha256((out/'DATASET.json').read_bytes()).hexdigest()),indent=2))
    for name in ['program_stap_learning.py','program_stap_kernel.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    print('TRAINED',seed,len(rows),'rows; optimizer steps',(1+len(variants))*updates+400)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=31103);p.add_argument('--updates',type=int,default=800);p.add_argument('--variants',default='fq,foresight');a=p.parse_args();train(a.root,a.out,a.seed,a.updates,a.variants.split(','))
