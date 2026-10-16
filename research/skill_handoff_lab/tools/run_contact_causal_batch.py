"""Stage action interventions on predeclared seed31, development cases only."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--gpu',type=int,required=True);p.add_argument('--part',type=int,choices=[0,1],required=True)
a=p.parse_args();root=a.root.resolve()
jobs=[('phase0_feedback','ppo'),('phase1_feedback','ppo'),('zero','ppo'),('none','goal_feedback')]
reports=[]
for intervention,controller in jobs[a.part::2]:
    name=f'causal_seed31_{intervention}' if controller=='ppo' else 'contact_goal_feedback_eval'
    output=root/'runs'/name
    if output.exists():raise FileExistsError(output)
    cmd=[str(root/'envs/isaac51-check/bin/python'),str(root/'runs/contact_causal_engine_20261016/run_contact_composition.py'),
        '--asset',str(root/'runs/contact_composition_asset_20261016'),'--output',str(output),
        '--gpu',str(a.gpu),'--num-envs','16','--steps','0','--seed','31','--controller',controller,
        '--eval-seed','1731','--eval-episodes','64','--skip-capture','--action-intervention',intervention]
    if controller=='ppo':cmd+=['--load-policy',str(root/'runs/contact_v5_boundary_seed31/policy.zip')]
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
    with (root/'runs'/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=proc.returncode
    result={'name':name,'exit_code':code,'command':cmd}
    if (output/'manifest.json').exists():result['result']=json.loads((output/'manifest.json').read_text())
    reports.append(result);(root/f'runs/contact_causal_part{a.part}.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'name':name,'exit_code':code,'successes':result.get('result',{}).get('test_successes')}),flush=True)
