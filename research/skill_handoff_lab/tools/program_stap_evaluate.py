"""Frozen actual-physics evaluation of learned two-skill selectors. No training."""
import argparse,gzip,hashlib,json,time,random
from pathlib import Path
from program_stap_tool import *
from program_stap_learning import Selector

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--methods',default='fixed,stap,fq,foresight');ap.add_argument('--seeds',default='743020,743021');ap.add_argument('--targets',default='-.08,.08');ap.add_argument('--fixed-hook',type=float,default=-.055);ap.add_argument('--fixed-grasp',type=float,default=.105);ap.add_argument('--samples',type=int,default=16);ap.add_argument('--masses',default='.75,1.25');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(31104);np.random.seed(31104);random.seed(31104);rows=[]
    for density in map(float,a.masses.split(',')):
        env=ContactRecorder(gym.make('ARES-ToolLoad-v0',density_scale=density,num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1))
        env.allow_unstable=False;env.stability_max=10;env.acceleration=6.;env.target_feedback=True
        try:
            for seed in map(int,a.seeds.split(',')):
                for target in map(float,a.targets.split(',')):
                    env.target_y=target
                    for method in a.methods.split(','):
                        case_rng=31104+seed+int(density*100)+int((target+1)*1000);torch.manual_seed(case_rng);np.random.seed(case_rng);random.seed(case_rng)
                        out=a.out/f'{seed}_m{density}_y{target}_{method}';out.mkdir();env.parameter_selector=None if method=='fixed' else Selector(a.model,method,a.samples);env.stability=None;error=None;start=time.monotonic()
                        sources={}
                        for name in ['program_stap_evaluate.py','program_stap_tool.py','program_stap_learning.py','program_stap_kernel.py','program_multiskill_pull.py','program_multiskill_contact_probe.py','pull_tool_physics.py','handoff_physics.py']:
                            b=Path(__file__).with_name(name).read_bytes();(out/name).write_bytes(b);sources[name]=hashlib.sha256(b).hexdigest()
                        manifest=dict(task='ARES-ToolTarget-v0',native_parent='PullCubeTool-v1',native_goal_retained=True,additional_goal_tolerance_m=.04,seed=seed,density_scale=density,target_y=target,method=method,fixed_grasp=a.fixed_grasp,fixed_hook=a.fixed_hook,case_rng_seed=case_rng,cem_samples_per_dimension=a.samples,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),source_sha256=sources,horizon=100,training=False,acceleration=6.,velocity=1.8,target_feedback=True,scope='ACTUAL_OFFLINE_PHYSICS; learned selector or fixed strong baseline; no hardware')
                        (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
                        try:execute_chain(env,seed,grasp_x=a.fixed_grasp,hook_y=a.fixed_hook)
                        except Exception as exc:error=repr(exc)
                        q=np.asarray(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));env.phase='TERMINAL_HOLD'
                        while len(env.records)<100:env.step(np.r_[q,-1.])
                        dist=float(np.linalg.norm(np.array(env.records[-1]['cube'][:2])-np.array([sp(env.unwrapped.agent.robot).p[0]+.57,target])))
                        native=bool(env.records[-1]['native_success']);success=native and dist<=.04
                        env.skill_results.append(dict(skill='USE_TOOL',boundary_step=len(env.records),physical_success=success,native_success=native,target_distance_m=dist,measurement=env.measure(),remaining_steps=0,error=error))
                        logs=[] if env.parameter_selector is None else env.parameter_selector.logs
                        row=dict(seed=seed,density_scale=density,target_y=target,method=method,success=native,task_success=success,target_distance_m=dist,error=error,steps=len(env.records),stability=env.stability,initial_sha256=env.initial_sha,decisions=len(env.decisions),wall_s=time.monotonic()-start,model_queries=sum(z['skill_model_queries'] for z in logs),selector_wall_s=sum(z['wall_s'] for z in logs))
                        data=dict(summary=row,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans,decisions=env.decisions,skill_results=env.skill_results,selector_logs=logs)
                        with gzip.open(out/'TRAJECTORY.json.gz','wt') as f:json.dump(data,f,separators=(',',':'))
                        row.update(path=str(out/'TRAJECTORY.json.gz'),sha256=hashlib.sha256((out/'TRAJECTORY.json.gz').read_bytes()).hexdigest());(out/'RESULTS.json').write_text(json.dumps([row],indent=2));rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        finally:env.close()
if __name__=='__main__':main()
