"""Frozen geometry evaluation of ALL predeclared seeds, never best-seed selection."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--gpu',type=int,required=True);p.add_argument('--part',type=int,choices=[0,1],required=True)
a=p.parse_args();root=a.root.resolve()
jobs=[(c,None) for c in ['nominal','fixed_push','random','goal_feedback']]
jobs += [('ppo',f'contact_v5_{m}_seed{s}') for m in ['standard','boundary'] for s in [31,41,51]]
jobs += [('ppo','contact_v5_all_mc_seed31')]
env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
reports=[]
for controller,model in jobs[a.part::2]:
    name='heldout_'+(model or controller);output=root/'runs'/name
    if output.exists():raise FileExistsError(output)
    if model and not (root/'runs'/model/'policy.zip').exists():raise FileNotFoundError(model)
    cmd=[str(root/'envs/isaac51-check/bin/python'),str(root/'runs/contact_eval_engine_20261016/run_contact_composition.py'),
        '--asset',str(root/'runs/contact_composition_asset_20261016'),'--output',str(output),
        '--gpu',str(a.gpu),'--num-envs','16','--steps','0','--seed','31','--controller',controller,
        '--scenario-file',str(root/'runs/FROZEN_HELDOUT_SCENARIOS.json'),'--eval-seed','2703',
        '--eval-episodes','64','--skip-capture']
    if model:cmd+=['--load-policy',str(root/'runs'/model/'policy.zip')]
    start=time.monotonic()
    with (root/'runs'/(name+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode
    report={'name':name,'exit_code':code,'wall_s':time.monotonic()-start,'command':cmd}
    if (output/'manifest.json').exists():report['result']=json.loads((output/'manifest.json').read_text())
    reports.append(report)
    (root/f'runs/heldout_batch_{a.part}.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'name':name,'exit_code':code,'successes':report.get('result',{}).get('test_successes')}),flush=True)
