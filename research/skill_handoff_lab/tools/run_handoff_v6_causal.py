"""Same-weight physical interventions and separately counted training video."""
import argparse,json,os,subprocess,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
a=p.parse_args();r=a.root.resolve();reports=[]
jobs=[('v6b_rule_recovery_seed31','v6b_residual_seed31',0,'validation',['--rule-recovery']),
      ('v6b_recover_force_resume31','v6b_recover_seed31',0,'validation',['--force-switch-signal','1']),
      ('v6b_actual_training_video','v6b_recover_seed31',16384,'train',[])]
for name,model,steps,split,extra in jobs:
    output=r/'runs'/name
    if output.exists():raise FileExistsError(output)
    cmd=[str(r/'envs/isaac51-check/bin/python'),str(r/'runs/contact_handoff_v6_engine/run_contact_v6.py'),
        '--asset',str(r/'runs/contact_handoff_narrow_asset'),'--output',str(output),'--gpu',str(a.gpu),'--num-envs','16',
        '--steps',str(steps),'--seed','31','--controller','residual','--switch-mode','recover','--gamma','1','--rollout-steps','256',
        '--task-version','v6b','--episode-seconds','12','--scenario-file',str(r/'runs/contact_handoff_v6b_configs'/(split+'.json')),
        '--eval-seed','3801','--eval-episodes','32' if steps else '64','--load-policy',str(r/'runs'/model/'policy.zip')]+extra
    if not steps:cmd+=['--skip-capture']
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
    with (r/'runs'/(name+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=480)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode
    if code or not (output/'RUN_COMPLETE.json').exists():raise RuntimeError(name)
    m=json.loads((output/'manifest.json').read_text());reports.append({'command':cmd,'result':m})
    (r/'runs/v6b_causal_batch.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'name':name,'success':m['test_successes']}),flush=True)
