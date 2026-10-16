"""Bounded correctness diagnosis: 2 active Cartesian actions, two fixed noise priors."""
import argparse,json,subprocess,os,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--noise',choices=['low','high'],required=True);a=p.parse_args();r=a.root.resolve();rows=[]
for split,steps in [('train',65536),('validation',0)]:
    name=f'v6b_active2_{a.noise}31'+('_validation' if not steps else '');output=r/'runs'/name
    if output.exists():raise FileExistsError(output)
    cmd=[str(r/'envs/isaac51-check/bin/python'),str(r/'runs/contact_diagnostic_v6_engine/run_contact_v6.py'),
        '--asset',str(r/'runs/contact_handoff_narrow_asset'),'--output',str(output),'--gpu',str(a.gpu),'--num-envs','16',
        '--steps',str(steps),'--controller','ppo','--policy-mode','active_gaussian','--initial-log-std',str(-1.5 if a.noise=='low' else 0.),
        '--gamma','1','--rollout-steps','256','--episode-seconds','12','--task-version','v6b','--seed','31',
        '--scenario-file',str(r/'runs/contact_handoff_v6b_configs'/(split+'.json')),'--eval-seed','3801',
        '--eval-episodes','64' if not steps else '32','--skip-capture']
    if not steps:cmd+=['--load-policy',str(r/'runs'/f'v6b_active2_{a.noise}31'/'policy.zip')]
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
    with (r/'runs'/(name+'.log')).open('w') as log:
        process=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=process.returncode
    if code or not (output/'RUN_COMPLETE.json').exists():raise RuntimeError(name)
    m=json.loads((output/'manifest.json').read_text());rows.append({'command':cmd,'result':m})
    (r/f'runs/active_gaussian_{a.noise}.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'name':name,'success':m['test_successes']}),flush=True)
