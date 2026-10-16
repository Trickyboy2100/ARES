"""Bounded single-GPU queue. Check artifact completion, not Kit exit code alone."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--gpu',type=int,required=True)
p.add_argument('--kind',choices=['mechanism','plain','baseline','method','ablation'],required=True)
p.add_argument('--method',choices=['learned','recover'],default='learned')
a=p.parse_args();r=a.root.resolve();cfg=r/'runs/contact_handoff_v6b_configs'
jobs=[]
if a.kind=='mechanism':
    jobs=[(f'v6b_paired_dwell{n}',0,'paired','fixed',31,'validation',None,
        ['--switch-dwell',str(n),'--legacy-policy',str(r/'runs/contact_v5_boundary_seed31/policy.zip')]) for n in [0,12,30]]
    jobs += [('v6b_damped_validation',0,'damped','fixed',31,'validation',None,[])]
else:
    for seed in ([31] if a.kind=='ablation' else [31,41,51]):
        mode=a.method if a.kind=='method' else 'fixed'
        label='bootstrap_all' if a.kind=='ablation' else ('ppo' if a.kind=='plain' else ('residual' if a.kind=='baseline' else a.method))
        name=f'v6b_{label}_seed{seed}'
        extra=['--credit-mode','all'] if a.kind=='ablation' else []
        controller='ppo' if a.kind=='plain' else 'residual'
        jobs.append((name,65536,controller,mode,seed,'train',None,extra))
        jobs.append((name+'_validation',0,controller,mode,seed,'validation',name,[]))
reports=[];env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
for name,steps,controller,mode,seed,split,model,extra in jobs:
    output=r/'runs'/name
    if output.exists():raise FileExistsError(output)
    cmd=[str(r/'envs/isaac51-check/bin/python'),str(r/'runs/contact_handoff_v6_engine/run_contact_v6.py'),
        '--asset',str(r/'runs/contact_handoff_narrow_asset'),'--output',str(output),'--gpu',str(a.gpu),
        '--num-envs','16','--steps',str(steps),'--seed',str(seed),'--controller',controller,'--switch-mode',mode,
        '--gamma','1','--rollout-steps','256','--episode-seconds','12','--task-version','v6b',
        '--scenario-file',str(cfg/(split+'.json')),'--eval-seed','3801','--eval-episodes','64' if split=='validation' else '32',
        '--skip-capture']+extra
    if model:cmd+=['--load-policy',str(r/'runs'/model/'policy.zip')]
    start=time.monotonic()
    with (r/'runs'/(name+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode
    row={'name':name,'command':cmd,'exit_code':code,'wall_s':time.monotonic()-start,
         'complete':(output/'RUN_COMPLETE.json').exists() and not (output/'exception.txt').exists()}
    if (output/'manifest.json').exists():row['result']=json.loads((output/'manifest.json').read_text())
    reports.append(row);(r/f'runs/handoff_batch_{a.kind}_{a.gpu}.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'name':name,'complete':row['complete'],'success':row.get('result',{}).get('test_successes')}),flush=True)
    if not row['complete']:raise RuntimeError('Stop bounded queue on software/timeout failure; preserve all outputs')
