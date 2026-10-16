"""Bounded, predeclared single-GPU study queue; only own process groups stopped."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--gpu',type=int,required=True);p.add_argument('--kind',choices=['baseline_boundary','standard_ablation'],required=True)
a=p.parse_args();root=a.root.resolve();jobs=[]
if a.kind=='baseline_boundary':
    jobs=[(f'contact_v5_{c}_eval',c,'standard',31,0,True) for c in ['nominal','fixed_push','random']]
    jobs += [(f'contact_v5_boundary_seed{s}','ppo','boundary',s,32768,s!=31) for s in [31,41,51]]
else:
    jobs=[(f'contact_v5_standard_seed{s}','ppo','standard',s,32768,True) for s in [31,41,51]]
    jobs += [('contact_v5_all_mc_seed31','ppo','all',31,32768,True)]
env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
reports=[];manifest=root/f'runs/contact_v5_batch_{a.kind}.json'
for name,controller,credit,seed,steps,skip in jobs:
    output=root/'runs'/name
    if output.exists():raise FileExistsError(f'refuse to overwrite prior result: {output}')
    cmd=[str(root/'envs/isaac51-check/bin/python'),str(root/'runs/run_contact_composition.py'),
        '--asset',str(root/'runs/contact_composition_asset_20261016'),'--output',str(output),
        '--gpu',str(a.gpu),'--num-envs','16','--steps',str(steps),'--seed',str(seed),
        '--controller',controller,'--family','both','--credit-mode',credit,'--gamma','1.0',
        '--rollout-steps','512','--eval-episodes','64']+(['--skip-capture'] if skip else [])
    begin=time.monotonic();log=root/'runs'/(name+'.log');status='COMPLETED_PROCESS'
    with log.open('w') as stream:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=480)
        except subprocess.TimeoutExpired:
            status='TIMEOUT';os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode
    if code and status!='TIMEOUT':status='SOFTWARE_ERROR'
    report={'name':name,'status':status,'exit_code':code,'wall_s':time.monotonic()-begin,'command':cmd}
    if (output/'manifest.json').exists():report['result']=json.loads((output/'manifest.json').read_text())
    reports.append(report);manifest.write_text(json.dumps({'gpu':a.gpu,'jobs':reports,'predeclared_jobs':jobs},indent=2))
    print(json.dumps({'name':name,'status':status,'successes':report.get('result',{}).get('test_successes')}),flush=True)
