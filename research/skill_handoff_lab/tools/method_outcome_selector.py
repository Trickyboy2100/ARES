"""One-decision paired-outcome selector and matched real-physics evaluation.

This is a contextual bandit at the first place boundary, not a new SMDP or PPO
loss. Features are current state/previous measured effect; labels are obtained
only from separately charged cold-reset training rollouts.
"""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from method_stack_physics import make, ParameterStack, serial

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

FEATURE_NAMES=(['object_goal_'+x for x in 'xyz']+['tcp_object_'+x for x in 'xyz']+
               ['q'+str(i) for i in range(9)]+['object_quat_'+str(i) for i in range(4)]+
               ['goal_quat_'+str(i) for i in range(4)]+['effect_'+x for x in 'xyz']+
               ['remaining_budget']+['next_goal_'+x for x in 'xyz'])

def features(cube,base,tcp,q,effect,remaining):
    cube,base,tcp,q,effect=map(np.asarray,[cube,base,tcp,q,effect])
    remaining=np.broadcast_to(np.asarray(remaining).reshape(-1,1),(len(cube),1))
    return np.concatenate([cube[:,:3]-base[:,:3],tcp[:,:3]-cube[:,:3],q,cube[:,3:],base[:,3:],effect,remaining,base[:,:3]],axis=1).astype(np.float32)

def transition_features(row):
    return features(row['cubeA'],row['cubeB'],row['tcp'],row['q'],row['authority']['effect_m'],(50-row['step'])/50)

def net():return nn.Sequential(nn.Linear(len(FEATURE_NAMES),64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))

def fit(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    variants=['nominal','height_minus','height_plus'];contexts=[];outcomes=[];times=[];ids=[];source_hashes={};missing=0
    for batch in range(32):
        records=[json.loads((args.data/f'{batch:03d}_{v}.json').read_text()) for v in variants]
        for v in variants:source_hashes[f'{batch:03d}_{v}.json']=sha(args.data/f'{batch:03d}_{v}.json')
        assert len({r['summary']['initial_sha256'] for r in records})==1
        for i,seed in enumerate(records[0]['summary']['seeds']):
            states=records[0]['transitions'];boundary=next((j for j,r in enumerate(states) if r['authority']['phase_before'][i]>=4),None)
            if boundary is None or boundary==0:
                missing+=1;continue  # No legal decision boundary; deployment keeps nominal.
            context=transition_features(states[boundary-1])[i]
            for r in records[1:]:assert np.max(abs(transition_features(r['transitions'][boundary-1])[i]-context))<1e-5,'Pair prefix differs before parameter use'
            contexts.append(context);ids.append(seed);outcomes.append([r['summary']['success'][i] for r in records]);completion=[]
            for r in records:
                successful=np.array([x['success'][i] for x in r['transitions']],bool)
                completion.append((np.flatnonzero(~successful)[-1]+2) if successful[-1] and not successful.all() else (1 if successful.all() else 50))
            times.append(completion)
    X=np.array(contexts);Y=np.array(outcomes,dtype=np.float32);T=np.array(times);D=Y[:,1:]-Y[:,0:1]
    ids=np.array(ids);train=ids%4!=3;calibration=~train;mean=X[train].mean(0);std=np.maximum(X[train].std(0),.01);Z=(X-mean)/std
    torch.save(dict(X=X,Y=Y,T=T,D=D,seeds=ids,feature_names=FEATURE_NAMES,source_sha256=source_hashes),args.output/'paired_dataset.pt')
    modes=['paired','no_effect','no_next_contract','no_pairing'];metrics={}
    for mode in modes:
        mask=np.ones(X.shape[1],np.float32)
        if mode=='no_effect':mask[[i for i,n in enumerate(FEATURE_NAMES) if n.startswith('effect_')]]=0
        if mode=='no_next_contract':mask[[i for i,n in enumerate(FEATURE_NAMES) if n.startswith(('object_goal_','goal_quat_','next_goal_'))]]=0
        target=Y if mode=='no_pairing' else np.column_stack([np.zeros(len(D)),D])
        ensemble=[]
        for seed in [31,37,43,47,53]:
            torch.manual_seed(seed);model=net();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=np.random.default_rng(seed);indices=np.flatnonzero(train);bootstrap=rng.choice(indices,len(indices),replace=True)
            xx=torch.tensor(Z[bootstrap]*mask);yy=torch.tensor(target[bootstrap],dtype=torch.float32)
            for epoch in range(240):
                prediction=model(xx);loss=((prediction-yy)**2).mean();optimizer.zero_grad();loss.backward();optimizer.step()
            ensemble.append(model.state_dict())
        bundle=dict(models=ensemble,mean=mean,std=std,mask=mask,mode=mode,source='ACTUAL_COLD_RESET_PAIRED_PHYSICS',features=FEATURE_NAMES,actions=[[0,0,0],[0,0,-1],[0,0,1]],lambda_step=0.0,threshold=.05,uncertainty_multiplier=1.,train_seed_ids=ids[train].tolist(),calibration_seed_ids=ids[calibration].tolist(),sealed_test_used=False)
        target_file=args.output/(mode+'.pt');torch.save(bundle,target_file)
        score=Selector(target_file);pred=score.scores(X[calibration]);true=D[calibration];metrics[mode]=dict(calibration_mse=float(np.mean((pred.mean(0)-true)**2)),model_sha256=sha(target_file))
    summary=dict(source_kind='NEW_PHYSICAL_SIMULATION_TRAINING_LABELS',conditions_with_boundary=len(X),no_boundary_conditions=missing,feature_names=FEATURE_NAMES,total_cold_reset_interactions=512*3*50,fit_conditions=int(train.sum()),calibration_conditions=int(calibration.sum()),models=metrics,independent_method_gain_established=False,type='ONE_BOUNDARY_CONTEXTUAL_BANDIT',label_formula='Ycandidate-Ynominal; every paired episode really executes50 steps, so actual execution time differences are zero; stable completion time is diagnostic only',train_pair_cost_included=True)
    (args.output/'FIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

class Selector:
    def __init__(self,path):
        self.bundle=torch.load(path,map_location='cpu',weights_only=False);self.models=[]
        for state in self.bundle['models']:
            model=net();model.load_state_dict(state);model.eval();self.models.append(model)
    def scores(self,x):
        xx=torch.tensor(((x-self.bundle['mean'])/self.bundle['std'])*self.bundle['mask'],dtype=torch.float32)
        with torch.inference_mode():
            absolute=np.stack([m(xx).numpy() for m in self.models])
            return absolute[:,:,1:]-absolute[:,:,:1]
    def action(self,x,abstain=True,uncertainty=True):
        scores=self.scores(x);value=scores.mean(0)-(scores.std(0) if uncertainty else 0)
        choice=value.argmax(1)+1
        if abstain:choice[value.max(1)<=self.bundle['threshold']]=0
        return np.asarray(self.bundle['actions'],np.float32)[choice],choice,value

def evaluate(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    upstream=args.upstream;spec=importlib.util.spec_from_file_location('pinned_ppo_eval',upstream);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    jobs=[('nominal',None,None),('fixed_height_minus',None,[0,0,-1]),('fixed_height_plus',None,[0,0,1])]
    if args.models:
        for seed in [31,37,43]:
            for mode in ['residual','parameter']:
                checkpoint=args.models/f'{mode}_{seed}_102400/final_ckpt.pt'
                if checkpoint.exists():jobs.append((f'{mode}_ppo_{seed}',checkpoint,mode))
    if args.selector:
        for mode in ['paired','no_effect','no_next_contract','no_pairing']:
            jobs.append((mode,args.selector/(mode+'.pt'),'selector'))
        jobs.extend([('no_abstain',args.selector/'paired.pt','selector'),('no_uncertainty',args.selector/'paired.pt','selector')])
    if args.cem:
        cem=json.loads(args.cem.read_text())['best']['parameter'];jobs.append(('cem_global_height',None,cem))
    if args.only:jobs=[job for job in jobs if job[0] in args.only.split(',')]
    import gymnasium as gym
    rows=[];batch_hashes={}
    for label,checkpoint,mode in jobs:
        native=gym.make('StackCube-v1',num_envs=16,obs_mode='state',control_mode='pd_ee_delta_pos',sim_backend='physx_cuda',render_backend='gpu',render_mode='rgb_array',reconfiguration_freq=1)
        env=ParameterStack(native,'residual' if mode=='residual' else 'parameter');u=env.unwrapped;agent=None;selector=None
        if checkpoint and mode!='selector':agent=module.Agent(env).to(u.device);agent.load_state_dict(torch.load(checkpoint,map_location=u.device,weights_only=True));agent.eval()
        if mode=='selector':selector=Selector(checkpoint)
        for batch in range(args.cases//16):
            seeds=list(range(args.seed+batch*16,args.seed+(batch+1)*16));obs,_=env.reset(seed=seeds);initial=serial(u.get_state_dict());digest=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
            if batch not in batch_hashes:batch_hashes[batch]=digest
            assert digest==batch_hashes[batch],'Evaluation initial-state mismatch'
            writers={}
            if args.video:
                import imageio.v2 as imageio
                for index in [int(x) for x in args.video_indices.split(',')]:writers[index]=imageio.get_writer(args.output/f'{label}_case{seeds[index]}.mp4',fps=20,codec='libx264')
            states=[];selected=np.zeros((16,3),np.float32);decided=np.zeros(16,bool);decisions=[]
            for step in range(50):
                if agent is not None:
                    with torch.inference_mode():action=agent.get_action(obs,deterministic=True)
                elif selector is not None:
                    mask=(env.phase.cpu().numpy()>=4)&~decided
                    if mask.any():
                        x=features(serial(u.cubeA.pose.raw_pose),serial(u.cubeB.pose.raw_pose),serial(u.agent.tcp.pose.raw_pose),serial(u.agent.robot.get_qpos()),serial(env.effect),(50-step)/50)
                        proposed,choice,value=selector.action(x,abstain=label!='no_abstain',uncertainty=label!='no_uncertainty');selected[mask]=proposed[mask];decided[mask]=True
                        decisions.append(dict(before_step=step+1,mask=mask.tolist(),choice=choice.tolist(),estimated_gain=value.tolist(),features_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
                    action=torch.tensor(selected,device=u.device)
                else:action=torch.tensor(mode or [0,0,0],device=u.device).expand(16,3)
                obs,reward,term,trunc,info=env.step(action)
                if writers:
                    from PIL import Image,ImageDraw,ImageFont
                    frames=u.render_rgb_array().detach().cpu().numpy()
                    for index,writer in writers.items():
                        im=Image.new('RGB',(512,576),'white');im.paste(Image.fromarray(frames[index]),(0,42));draw=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13);draw.text((8,4),f'SAPIEN / PhysX | StackCube-v1 | case {seeds[index]}',font=font,fill='#203b51');draw.text((8,22),f'{label} | step {step+1}/50 | native physics',font=font,fill='#203b51');draw.text((8,555),'ACTUAL SIMULATION / NO PHYSICAL ROBOT',font=font,fill='#203b51');writer.append_data(np.asarray(im))
                states.append(serial(dict(step=step+1,cubeA=u.cubeA.pose.raw_pose,cubeB=u.cubeB.pose.raw_pose,tcp=u.agent.tcp.pose.raw_pose,q=u.agent.robot.get_qpos(),reward=reward,success=info['success'],authority=info['authority'])))
            for writer in writers.values():writer.close()
            final=serial(u.evaluate());entry=dict(policy=label,seeds=seeds,success=final['success'],initial_sha256=digest,checkpoint_sha256=sha(checkpoint) if checkpoint else None,steps=50,selected=selected.tolist(),decisions=decisions)
            file=args.output/f'{label}_{batch:03d}.json';file.write_text(json.dumps(dict(summary=entry,initial=initial,transitions=states),separators=(',',':')));entry['trajectory_sha256']=sha(file);rows.append(entry);print(json.dumps({k:entry[k] for k in ['policy','seeds','success']}),flush=True)
            (args.output/'RESULTS.json').write_text(json.dumps(dict(source_kind='NEW_PHYSICAL_SIMULATION',task='StackCube-v1',native_horizon=50,rows=rows,physical_interactions=len(rows)*16*50,sealed_test_used=False),indent=2)+'\n')
        env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('fit');q.add_argument('--data',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('eval');q.add_argument('--output',type=Path,required=True);q.add_argument('--upstream',type=Path,required=True);q.add_argument('--models',type=Path);q.add_argument('--selector',type=Path);q.add_argument('--cem',type=Path);q.add_argument('--only');q.add_argument('--video',action='store_true');q.add_argument('--video-indices',default='0');q.add_argument('--cases',type=int,default=64);q.add_argument('--seed',type=int,default=200000)
    args=p.parse_args();fit(args) if args.command=='fit' else evaluate(args)
