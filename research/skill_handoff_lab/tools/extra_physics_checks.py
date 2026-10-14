"""Bounded baseline and integration probes after GPU1 final grid exits."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run=a.root/'runs'
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    path=run/'multicontract_v19_gpu1_test_queue.json';rows=json.loads(path.read_text()) if path.exists() else []
    if any(r['returncode'] for r in rows):raise SystemExit('Prior experiment failure: no additional launch.')
    if len(rows)==6:break
    time.sleep(5)
else:raise SystemExit('Bounded queue wait expired.')
common=[sys.executable,str(run/'run_fullchain_v20.py'),'--root',str(a.root),
    '--script',str(run/'fullchain_v20.py'),'--asset',str(run/'asset_mesh_colliders_v5'),
    '--industrial','--gpu','1','--seed','31','--task-suite',str(run/'multicontract_v5.json'),
    '--reward-revision','v2','--timeout','240','--steps','0','--process-audit','--action-audit']
records=[]
for split in ['validation','test']:
    argv=common+['--name','multicontract_v20_nominal_'+split,'--num-envs','16','--controller','nominal',
        '--split',split,'--eval-episodes','64','--skip-capture']
    rc=subprocess.run(argv).returncode;records.append({'argv':argv,'returncode':rc})
    (run/'multicontract_v20_extra_queue.json').write_text(json.dumps(records,indent=2))
    if rc:raise SystemExit('Baseline failure; no retry.')
argv=common+['--name','multicontract_v20_fullchain','--num-envs','1','--controller','contract',
    '--split','train','--eval-episodes','1','--load-policy',str(run/'r1_20261014/multicontract_v19_contract_seed31/ppo_physics.zip'),
    '--preplace-result',str(run/'r1_20261014/mechanism_a_execution/0/result.json')]
rc=subprocess.run(argv).returncode;records.append({'argv':argv,'returncode':rc})
(run/'multicontract_v20_extra_queue.json').write_text(json.dumps(records,indent=2));raise SystemExit(rc)
