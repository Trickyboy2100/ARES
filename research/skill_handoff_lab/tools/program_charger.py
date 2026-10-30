"""EXP-001: instrumented native200 controls, no state setters or learned policy."""
import argparse,gzip,hashlib,json,random,time,subprocess
from pathlib import Path
import numpy as np,torch,gymnasium as gym
import charger_probe as cp
from pull_tool_physics import Recorder,TracePlanner,Horizon,Failure,serial,sp

class GoalReached(Exception):pass

class Measure(Recorder):
 def reset(self,**kw):self.calls=[];return super().reset(**kw)
 def step(self,action):
  ret=super().step(action);z=self.records[-1];info=serial(self.unwrapped.evaluate());z['goal_position_error_m']=info['obj_to_goal_dist'][0];z['goal_angle_error_rad']=info['obj_to_goal_angle'][0];z['joint_command_error_rad']=(np.asarray(action[:7])-np.asarray(z['q'][:7])).tolist()
  if getattr(self,'stop_on_success',False) and z['native_success']:raise GoalReached()
  return ret
class Planner(TracePlanner):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw)
  for name in ['plan_screw','plan_qpos_to_pose','IK']:
   original=getattr(self.planner,name)
   def wrap(*aa,_fn=original,_name=name,**kk):
    start=time.monotonic();r=_fn(*aa,**kk);self.env.calls.append(dict(kind=_name,phase=self.env.phase,step=len(self.env.records),wall_s=time.monotonic()-start,status=r[0] if _name=='IK' else r['status']));return r
   setattr(self.planner,name,wrap)
cp.TracePlanner=Planner

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',default='721000,723002,724000,724001,724002,724003');p.add_argument('--variant',default='frozen');p.add_argument('--single-angle',type=float);p.add_argument('--single-distance',type=float);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);random.seed(31031);np.random.seed(31031);torch.manual_seed(31031)
 source=Path(__file__);(a.out/'source.py').write_bytes(source.read_bytes());rows=[];env=Measure(gym.make('PlugCharger-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));manifest=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),dependencies={n:hashlib.sha256((source.parent/n).read_bytes()).hexdigest() for n in ['charger_probe.py','pull_tool_physics.py']+(['program_charger_feedback.py'] if a.variant!='frozen' else [])},task='PlugCharger-v1',horizon=200,seeds=a.seeds,variant=a.variant,python_numpy_torch_seed=31031,mplib_seed='no public setter in pinned 0.1.1; RRT variability measured, not assumed controlled',gpu=torch.cuda.get_device_name(0),native_task_modified=False,optimizer_steps=0,restricted_field_data=False)
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 if a.variant!='frozen':(a.out/'feedback_source.py').write_bytes((source.parent/'program_charger_feedback.py').read_bytes())
 try:
  for seed in map(int,a.seeds.split(',')):
   digest=None
   for angle,d in ([(a.single_angle,a.single_distance)] if a.single_angle is not None else [(15.,.025),(45.,.05),(45.,.075)]):
    error=None;start=time.monotonic();name=f'{seed}_a{angle}_d{d}';env.progress=a.out/f'{name}.progress.json'
    env.stop_on_success=a.variant.endswith('_stop')
    try:
     if a.variant in ['frozen','frozen_stop']:cp.strong(env,seed,angle,d)
     else:
      from program_charger_feedback import strong
      strong(env,seed,angle,d,Planner,a.variant)
    except GoalReached:error=None
    except (Horizon,Failure) as e:error=str(e)
    except Exception as e:
     with gzip.open(a.out/f'{name}_CRASH.json.gz','wt') as f:json.dump(dict(error=repr(e),steps=env.records,calls=env.calls),f)
     raise
    env.stop_on_success=False
    if digest is None:digest=env.initial_sha
    assert digest==env.initial_sha
    if len(env.records)<200:
     q=np.array(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));grip=-1 if any(z['phase']=='GRASP' for z in env.records) else 1;env.phase='TERMINAL_HOLD'
     while len(env.records)<200:env.step(np.r_[q,grip])
    final=serial(env.unwrapped.evaluate());row=dict(seed=seed,angle=angle,standoff=d,success=bool(final['success'][0]),error=error,initial_sha256=env.initial_sha,steps=len(env.records),wall_s=time.monotonic()-start,plan_calls=len(env.calls),rrt_calls=sum(z['kind']=='plan_qpos_to_pose' for z in env.calls),planning_wall_s=sum(z['wall_s'] for z in env.calls),final=final)
    path=a.out/f'{name}.json.gz'
    with gzip.open(path,'wt') as f:json.dump(dict(summary=row,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans,calls=env.calls),f,separators=(',',':'))
    row['trajectory']=path.name;row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 finally:env.close()
 manifest.update(episodes=len(rows),physical_steps=sum(z['steps'] for z in rows));(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
