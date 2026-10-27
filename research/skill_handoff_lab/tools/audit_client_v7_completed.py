"""Bounded local independent audits of completed synthetic pilot traces."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
a=p.parse_args();w=a.workspace.resolve()
validator=w/'repos/ARES/research/skill_handoff_lab/tools/validate_contact_composition.py'
def audit(d):
    if (d/'INDEPENDENT_TASK_AND_CONTACT.json').exists():return {'run':d.name,'status':'already audited'}
    cmd=[str(w/'envs/kinematics/bin/python'),str(validator),'--run',str(d),'--asset',str(w/'runs/contact_handoff_20261016_r1/asset_narrow'),
         '--urdf',str(w/'repos/ARES/isaac_sim/simforge/robot/jaka_minicobo_gripper.urdf')]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(w/'repos/ARES')+':'+str(w/'repos/ARES-R/src'))
    r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
    (d/'independent_audit.log').write_text(r.stdout)
    if r.returncode:return {'run':d.name,'status':'audit_error','exit_code':r.returncode,'output':r.stdout[-1000:]}
    v=json.loads((d/'INDEPENDENT_TASK_AND_CONTACT.json').read_text())
    return {'run':d.name,'status':'complete','raw':v['raw_successes'],'independent':v['independent_successes']}
dirs=[d for d in sorted(a.root.iterdir()) if (d/'RUN_COMPLETE.json').exists() and (d/'actual_physics_trace.npz').exists()]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for row in pool.map(audit,dirs):print(json.dumps(row),flush=True)
