"""Exactly one paired 32k pilot per arm, no automatic seed/hyperparameter search."""
import argparse,subprocess,json,os,time,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--arm',choices=['hybrid_joint','hybrid_conditional'],required=True);a=p.parse_args();r=a.root.resolve();reports=[]
for split,steps in [('train',32768),('validation',0)]:
    name=f'v6b_{a.arm}_pilot31'+('_validation' if split=='validation' else '')
    output=r/'runs'/name
    if output.exists():raise FileExistsError(output)
    cmd=[str(r/'envs/isaac51-check/bin/python'),str(r/'runs/contact_handoff_v6_engine/run_contact_v6.py'),
        '--asset',str(r/'runs/contact_handoff_narrow_asset'),'--output',str(output),'--gpu',str(a.gpu),'--num-envs','16',
        '--steps',str(steps),'--seed','31','--controller','residual','--switch-mode','recover','--policy-mode',a.arm,
        '--action-cost','executed','--gamma','1','--rollout-steps','256','--task-version','v6b','--episode-seconds','12',
        '--scenario-file',str(r/'runs/contact_handoff_v6b_configs'/(split+'.json')),'--eval-seed','3801',
        '--eval-episodes','64' if not steps else '32','--skip-capture']
    if not steps:cmd+=['--load-policy',str(r/'runs'/f'v6b_{a.arm}_pilot31'/'policy.zip')]
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
    with (r/'runs'/(name+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=480)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode
    if code or not (output/'RUN_COMPLETE.json').exists():raise RuntimeError(f'pilot failed, preserve {output}')
    m=json.loads((output/'manifest.json').read_text());reports.append({'command':cmd,'result':m})
    (r/f'runs/hybrid_{a.arm}_batch.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'name':name,'success':m['test_successes']}),flush=True)
