"""Predeclared final tests, no optimization or validation-based checkpoint picking."""
import argparse,json,os,subprocess,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--part',type=int,choices=[0,1,2],required=True);a=p.parse_args();r=a.root.resolve();jobs=[]
for model in ['residual','recover']:
    for seed in [31,41,51]:jobs.append((f'v6b_{model}_seed{seed}','validation',True))
for model in ['ppo','residual','recover']:
    for seed in [31,41,51]:
        for split in ['test_id','test_geometry','test_dynamics']:jobs.append((f'v6b_{model}_seed{seed}',split,False))
for split in ['test_id','test_geometry','test_dynamics']:
    jobs.extend([(None,split,False),('v6b_bootstrap_all_seed31',split,False)])
reports=[]
for model,split,mid in jobs[a.part::3]:
    name=f'v6b_{split}_{model or "damped"}'+('_32k' if mid else '')
    output=r/'runs'/name
    if output.exists():raise FileExistsError(output)
    cfg=json.loads((r/'runs/contact_handoff_v6b_configs'/(split+'.json')).read_text())
    old=json.loads((r/'runs'/model/'run_config.json').read_text()) if model else {}
    controller=old.get('controller','damped');mode=old.get('switch_mode','fixed')
    cmd=[str(r/'envs/isaac51-check/bin/python'),str(r/'runs/contact_handoff_v6_engine/run_contact_v6.py'),
        '--asset',str(r/'runs/contact_handoff_narrow_asset'),'--output',str(output),'--gpu',str(a.gpu),'--num-envs','16',
        '--steps','0','--seed',str(old.get('seed',31)),'--controller',controller,'--switch-mode',mode,
        '--task-version','v6b','--episode-seconds','12','--scenario-file',str(r/'runs/contact_handoff_v6b_configs'/(split+'.json')),
        '--eval-seed',str(cfg['case_seed']),'--eval-episodes','32','--skip-capture']
    if model:
        path=r/'runs'/model/('checkpoints/policy_32768_after_update.zip' if mid else 'policy.zip')
        if not path.exists():raise FileNotFoundError(path)
        cmd+=['--load-policy',str(path)]
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
    with (r/'runs'/(name+'.log')).open('w') as log:
        process=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=process.returncode
    if code or not (output/'RUN_COMPLETE.json').exists():raise RuntimeError(f'Preserve failed eval {name}; do not silently record failure label')
    m=json.loads((output/'manifest.json').read_text());reports.append({'command':cmd,'result':m})
    (r/f'runs/v6b_final_test_part{a.part}.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'name':name,'success':m['test_successes']}),flush=True)
