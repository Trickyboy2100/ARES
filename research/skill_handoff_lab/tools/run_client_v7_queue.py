"""Bounded v7 paired PhysX pilot, one isolated application per GPU."""
import argparse,json,os,signal,subprocess,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
a=p.parse_args();r=a.root.resolve();d=r/'runs/client_sci_pilot_20261027/formal';jobs=[]
if a.gpu==0:
    for method,frame in [('B1','body'),('B2','task')]:
        for seed in [31,41,51]:
            name=f'{method}_seed{seed}';extra=['--blind-outcome']
            jobs.extend([(name,16384,seed,'train',None,frame,extra),(name+'_validation',0,seed,'validation',name,frame,extra)])
    jobs.append(('B0_validation',0,31,'validation',None,'task',[]))
    for signal in [-1.,1.]:jobs.append((f'B4_parameter_{signal:+.1f}',0,31,'validation',None,'task',['--force-switch-signal',str(signal)]))
elif a.gpu==1:
    for seed in [31,41,51]:
        name=f'B3_seed{seed}';jobs.extend([(name,16384,seed,'train',None,'task',[]),(name+'_validation',0,seed,'validation',name,'task',[])])
    for name,extra in [('B3_no_outcome_intervention',['--blind-outcome']),('B3_no_next_intervention',['--blind-next'])]:
        jobs.append((name,0,31,'validation','B3_seed31','task',extra))
    jobs.extend([('B3_no_next_seed31',16384,31,'train',None,'task',['--blind-next']),
                 ('B3_no_next_seed31_validation',0,31,'validation','B3_no_next_seed31','task',['--blind-next'])])
else:raise ValueError('Only GPUs0/1 authorized for this queue')
reports=[];environment=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
for name,steps,seed,split,model,frame,extra in jobs:
    output=d/name
    if output.exists():raise FileExistsError(output)
    is_control=name.startswith(('B0','B4'))
    cmd=[str(r/'envs/isaac51-check/bin/python'),str(d/'engine/run_contact_v7.py'),
        '--asset',str(r/'runs/contact_handoff_narrow_asset'),'--output',str(output),'--gpu',str(a.gpu),
        '--num-envs','16','--family','slot','--steps',str(steps),'--seed',str(seed),
        '--controller','damped' if is_control else 'residual','--frame',frame,'--method','outcome',
        '--gamma','1','--rollout-steps','256','--episode-seconds','12','--task-version','v6b',
        '--checkpoint-every','16384','--scenario-file',str(d.parent/'configs'/(split+'.json')),
        '--eval-seed','102741','--eval-episodes','8' if steps else '32','--skip-capture']+extra
    if model:cmd+=['--load-policy',str(d/model/'policy.zip')]
    start=time.monotonic()
    with (d/(name+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,cwd=r,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode
    row={'name':name,'command':cmd,'exit_code':code,'wall_s':time.monotonic()-start,
         'complete':(output/'RUN_COMPLETE.json').exists() and not (output/'exception.txt').exists()}
    if (output/'manifest.json').exists():row['manifest']=json.loads((output/'manifest.json').read_text())
    reports.append(row);(d/f'batch_gpu{a.gpu}.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'name':name,'complete':row['complete'],'success':row.get('manifest',{}).get('test_successes')}),flush=True)
    if not row['complete']:raise RuntimeError('Stop queue on actual failure, preserve all evidence')
