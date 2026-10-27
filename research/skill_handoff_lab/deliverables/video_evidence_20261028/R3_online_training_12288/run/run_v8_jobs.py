"""Explicit bounded v8 experiment commands; one sequential worker per GPU."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--jobs',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();base=r/'runs/client_sci_closed_loop_20261027';spec=json.loads(a.jobs.read_text());rows=[]
assert len({x['gpu'] for x in spec})==1 and spec[0]['gpu'] in [0,1]
for job in spec:
    out=base/job['round']/job['name']
    if out.exists():raise FileExistsError(out)
    command=[str(r/'envs/isaac51-check/bin/python'),str(base/job['round']/job.get('engine_subdir','engine')/'run_contact_v8.py'),
        '--asset',str(r/'runs/contact_handoff_narrow_asset'),'--output',str(out),'--gpu',str(job['gpu']),
        '--num-envs',str(job.get('num_envs',16)),'--family',job.get('family','slot'),'--steps',str(job.get('steps',0)),
        '--controller',job.get('controller','residual'),'--seed',str(job.get('seed',31)),
        '--frame','task','--task-version','v6b','--gamma','1','--rollout-steps','256',
        '--episode-seconds','12','--checkpoint-every','16384','--scenario-file',str(base/'configs'/job.get('scenario','development.json')),
        '--eval-seed',str(job.get('eval_seed',102841)),'--eval-episodes',str(job.get('eval_episodes',32)),
        '--skip-capture','--contact-following','--log-contact-history']+job.get('extra',[])
    if job.get('load_policy'):command+=['--load-policy',str(base/job['load_policy'])]
    environment=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
    start=time.monotonic()
    with (base/job['round']/(job['name']+'.log')).open('w') as log:
        child=subprocess.Popen(command,cwd=r,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=child.returncode
    row={'job':job,'command':command,'exit_code':code,'wall_s':time.monotonic()-start,
         'complete':(out/'RUN_COMPLETE.json').exists() and not (out/'exception.txt').exists()}
    if (out/'manifest.json').exists():row['manifest']=json.loads((out/'manifest.json').read_text())
    rows.append(row);(base/('batch_'+a.jobs.stem+'.json')).write_text(json.dumps(rows,indent=2))
    print(json.dumps({'name':job['name'],'complete':row['complete'],'success':row.get('manifest',{}).get('test_successes')}),flush=True)
    if not row['complete']:raise RuntimeError('Stop on software failure/timeout; preserve all outputs')
