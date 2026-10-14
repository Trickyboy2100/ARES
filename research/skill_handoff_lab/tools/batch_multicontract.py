"""Bounded two-worker experiment grid. Each invocation owns one GPU only.

Frozen validation is reported, not used for checkpoint selection. Final test
is deliberately untouched. No retry, driver maintenance or shared cleanup.
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--gpu',type=int,required=True);p.add_argument('--methods',nargs='+',required=True)
a=p.parse_args();run=a.root/'runs';records=[]
for method in a.methods:
    for seed in [31,47,59]:
        name=f'multicontract_v19_{method}_seed{seed}'
        common=[sys.executable,str(run/'run_multitask_v17.py'),'--root',str(a.root),
            '--script',str(run/'multitask_v19.py'),'--asset',str(run/'asset_mesh_colliders_v5'),
            '--num-envs','16','--industrial','--controller',method,'--gpu',str(a.gpu),
            '--seed',str(seed),'--task-suite',str(run/'multicontract_v5.json'),
            '--reward-revision','v2','--timeout','300']
        for phase in ['train','validation']:
            argv=common+['--name',name+('_val' if phase=='validation' else ''),
                '--steps','16384' if phase=='train' else '0','--split',phase,
                '--eval-episodes','32' if phase=='train' else '64','--process-audit']
            if phase=='validation':argv+=['--load-policy',str(run/'r1_20261014'/name/'ppo_physics.zip'),'--action-audit','--skip-capture']
            started=time.time();result=subprocess.run(argv)
            records.append({'method':method,'seed':seed,'phase':phase,'argv':argv,
                'returncode':result.returncode,'started_unix':started,'wall_s':time.time()-started})
            (run/f'multicontract_v19_gpu{a.gpu}_queue.json').write_text(json.dumps(records,indent=2))
            if result.returncode:raise SystemExit('Bounded grid stopped on actual experiment failure; no automatic retry.')
