"""Fresh physical execution of trained two-boundary values and strong controls."""
import argparse,gzip,json,time,random,hashlib
from pathlib import Path
from program_tool_effect_probe import *
from program_effect_value import Learned,features,GRASPS,MODES,dataset
class Selector:
 def __init__(self,model,method):self.model=model;self.method=method;self.logs=[];self.planned=None
 def choose(self,env,stage):
  obs=features(env.measure(),100-len(env.records),env.unwrapped.density_scale,0,stage);idx,log=self.model.choose(self.method,obs,stage,self.planned)
  if stage==0:self.planned=log.get('planned_mode')
  else:env.effect_mode=MODES[idx]
  self.logs.append(dict(stage=stage,step=len(env.records),selected_index=idx,observation=obs.tolist(),**log))
  return GRASPS[idx] if stage==0 else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[745020,745021]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);task.TracePlanner=EffectPlanner;model=Learned(a.model)
 train=[r for r in dataset(a.data) if r['seed']<745014];fixed=max([(sum(r['y'] for r in train if r['g']==g and r['a']==k),g,k) for g in range(3) for k in range(3)],key=lambda x:x[0]);methods=['fixed','geometry_baseline','geometry_frozen12','geometry_slow3','ordinary','supervised','stap_open','stap_replan','foresight'];rows=[]
 for mass in [.75,1.25]:
  env=ContactRecorder(gym.make('ARES-ToolLoad-v0',density_scale=mass,num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));env.allow_unstable=False;env.stability_max=10;env.acceleration=6.;env.target_y=None;env.target_feedback=False;env.grasp_candidates=GRASPS
  try:
   for seed in a.seeds:
    for method in methods:
     random.seed(31131);np.random.seed(31131);torch.manual_seed(31131);env.effect_mode=MODES[fixed[2]] if method=='fixed' else method.replace('geometry_','') if method.startswith('geometry_') else 'baseline';env.parameter_selector=None if method=='fixed' or method.startswith('geometry_') else Selector(model,method);env.stability=None;error=None;start=time.monotonic()
     try:task.execute_chain(env,seed,grasp_x=GRASPS[fixed[1]] if method=='fixed' else None,hook_y=None,transfer_stride=1,pull_policy='goal')
     except Exception as e:error=repr(e)
     q=np.array(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));env.phase='TERMINAL_HOLD'
     while len(env.records)<100:env.step(np.r_[q,-1.])
     r=dict(seed=seed,mass=mass,method=method,mode=env.effect_mode,grasp=env.decisions[0]['parameters']['grasp_x_m'],success=env.records[-1]['native_success'],error=error,stability=env.stability,initial_sha256=env.initial_sha,steps=100,wall_s=time.monotonic()-start,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest())
     path=a.out/f'{seed}_m{mass}_{method}.json.gz'
     with gzip.open(path,'wt') as f:json.dump(dict(summary=r,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans,decisions=env.decisions,skill_results=env.skill_results,policy_logs=[] if env.parameter_selector is None else env.parameter_selector.logs),f)
     r.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());rows.append(r);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(seed,mass,method,r['grasp'],r['mode'],r['success'],error,flush=True)
     if error and not error.startswith(('Horizon(', 'Failure(')):raise RuntimeError('Unexpected software error; recorded attempt is invalid: '+error)
  finally:env.close()
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(seeds=a.seeds,methods=methods,best_fixed_training_only=fixed,split='FRESH_DEVELOPMENT_NOT_SEALED',native_horizon=100,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest()),indent=2))
 for name in [Path(__file__).name,'program_tool_effect_probe.py','program_effect_value.py','program_stap_tool.py','program_multiskill_pull.py','program_multiskill_contact_probe.py','pull_tool_physics.py']:(a.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
if __name__=='__main__':main()
