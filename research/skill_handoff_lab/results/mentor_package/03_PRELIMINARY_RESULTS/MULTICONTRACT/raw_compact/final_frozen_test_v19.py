"""Evaluate all predeclared checkpoints once, no model selection or retries."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--methods',nargs='+',required=True)
p.add_argument('--wait-for-training',action='store_true')
a=p.parse_args();run=a.root/'runs';records=[]
if a.wait_for_training:
    deadline=time.monotonic()+1200
    queue=run/f'multicontract_v19_gpu{a.gpu}_queue.json'
    while time.monotonic()<deadline:
        completed=json.loads(queue.read_text()) if queue.exists() else []
        if any(r['returncode'] for r in completed):raise SystemExit('Training grid failure, final test not launched.')
        if len(completed)==len(a.methods)*3*2:break
        time.sleep(5)
    else:raise SystemExit('Bounded wait expired, no GPU process started.')
for method in a.methods:
    for seed in [31,47,59]:
        model=f'multicontract_v19_{method}_seed{seed}'
        # Checkpoint budget was frozen at 16,384 before any validation results.
        checkpoint=run/'r1_20261014'/model/'ppo_physics.zip'
        if not checkpoint.exists():raise FileNotFoundError(checkpoint)
        argv=[sys.executable,str(run/'run_multitask_v17.py'),'--root',str(a.root),
            '--script',str(run/'multitask_v19.py'),'--asset',str(run/'asset_mesh_colliders_v5'),
            '--name',model+'_test','--num-envs','16','--industrial','--controller',method,
            '--gpu',str(a.gpu),'--seed',str(seed),'--task-suite',str(run/'multicontract_v5.json'),
            '--reward-revision','v2','--timeout','240','--steps','0','--split','test',
            '--eval-episodes','64','--process-audit','--skip-capture','--load-policy',str(checkpoint)]
        started=time.time();result=subprocess.run(argv)
        records.append({'method':method,'seed':seed,'argv':argv,'returncode':result.returncode,'wall_s':time.time()-started})
        (run/f'multicontract_v19_gpu{a.gpu}_test_queue.json').write_text(json.dumps(records,indent=2))
        if result.returncode:raise SystemExit('No retry or checkpoint substitution.')
