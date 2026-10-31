"""CPU-only frozen-critic inspection. No simulated counterfactual outcomes."""
import argparse, gzip, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(row): return (row['seed'], tuple(row['bias_mm']))
def episodes(folder):
    file=folder/'EPISODES.json'
    return json.loads(file.read_text()) if file.exists() else json.loads((folder/'RESULTS.json').read_text())
def trace(row):
    assert sha(row['path'])==row['sha256']
    return json.load(gzip.open(row['path'],'rt'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--sb3',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.sb3));from stable_baselines3 import SAC
    controls={}
    for folder in a.runs.glob('eval_controls_*'):
        if not folder.is_dir(): continue
        for row in episodes(folder):
            if row['label']=='geometry':controls[key(row)]=row
    shared={key(r):r for r in episodes(a.runs/'eval_sac')}
    projected={key(r):r for r in episodes(a.runs/'eval_sac_zero_z')}
    rows=[];models={}
    for method in ['sac','effect_sac']:
        train=a.runs/f'{method}_pilot_1024_retry';modelpath=train/'MODEL.zip'
        model=SAC.load(modelpath,device='cpu');model.policy.set_training_mode(False)
        spec=importlib.util.spec_from_file_location('snapshot_contract',train/'program_contract_effect.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        models[method]={'checkpoint':str(modelpath),'sha256':sha(modelpath),'contract_sha256':sha(train/'program_contract_effect.py')}
        for row in episodes(a.runs/f'eval_{method}'):
            k=key(row);t=trace(row);s=trace(shared[k]);raw=np.array(s['transitions'][0]['obs'],dtype=np.float32)
            own=np.array(t['transitions'][0]['obs'],dtype=np.float32)
            effect=module.ContractEffect();effect.reset(raw);obs=effect.observation(method=='effect_sac');assert obs.shape==(90,)
            actor=np.asarray(model.predict(obs,deterministic=True)[0]);zeroz=actor.copy();zeroz[1]=0
            actions=np.array([np.zeros(3),actor,zeroz],dtype=np.float32);obst=torch.tensor(np.repeat(obs[None],3,axis=0))
            with torch.no_grad():qs=torch.cat(model.critic(obst,torch.tensor(actions)),dim=1).numpy()
            qmin=qs.min(1);order=np.argsort(-qmin);names=['zero','actor','actor_zero_z']
            own_effect=module.ContractEffect();own_effect.reset(own)
            reproduced=np.asarray(model.predict(own_effect.observation(method=='effect_sac'),deterministic=True)[0]);logged=np.asarray(t['transitions'][0]['requested'])
            assert np.max(np.abs(reproduced-logged))<1e-5
            action_t=torch.tensor(actor[None],requires_grad=True);ob=torch.tensor(obs[None]);value=torch.cat(model.critic(ob,action_t),dim=1).min();value.backward();grad=action_t.grad.numpy()[0]
            comparisons={}
            for label,other in [('zero_policy',controls.get(k)),('sac_zero_z_policy',projected.get(k))]:
                if other:
                    ot=trace(other);otherraw=np.array(ot['transitions'][0]['obs'])
                    comparisons[label]={'final_success':bool(other['success']),'raw21_max_abs_delta':float(np.max(np.abs(raw-otherraw))),'same_initial_sha':other['initial_sha256']==row['initial_sha256'],'trace_sha256':other['sha256']}
            rows.append(dict(method=method,seed=row['seed'],bias_mm=row['bias_mm'],final_success=bool(row['success']),trace_sha256=row['sha256'],reference_raw21_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),own_vs_shared_first_obs_max_abs_delta=float(np.max(np.abs(raw-own))),actor_reproduction_max_abs_delta=float(np.max(np.abs(reproduced-logged))),actions=actions.tolist(),q1=qs[:,0].tolist(),q2=qs[:,1].tolist(),q_min=qmin.tolist(),rank=[names[i] for i in order],actor_minus_zero_q=float(qmin[1]-qmin[0]),actor_minus_zero_z_q=float(qmin[1]-qmin[2]),actor_action_gradient=grad.tolist(),physical_full_policy_comparisons=comparisons))
    result={'protocol':'CPU frozen critics on shared first suffix observation; four reset-history copies reconstructed using checkpoint source; no physics or optimizer updates. Physical outcomes are full policies, not one-action counterfactuals. Soft Q is not success probability.','models':models,'action_order':['zero','actor','actor_zero_z'],'rows':rows,'script_sha256':sha(__file__)}
    for method in models:
        selected=[r for r in rows if r['method']==method]
        models[method].update(cases=len(selected),success=sum(r['final_success'] for r in selected),actor_ranked_above_zero=sum(r['actor_minus_zero_q']>0 for r in selected),actor_ranked_above_zero_z=sum(r['actor_minus_zero_z_q']>0 for r in selected),mean_actor_action=np.mean([r['actions'][1] for r in selected],axis=0).tolist(),max_shared_observation_difference=max(r['own_vs_shared_first_obs_max_abs_delta'] for r in selected),mean_actor_minus_zero_q=float(np.mean([r['actor_minus_zero_q'] for r in selected])))
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(models,indent=2))
if __name__=='__main__':main()
