"""Research-only seven-DOF output compatibility, no ARES-R source mutation.

The frozen planner can return six arm joints plus a locked gripper joint. The
old reporting code reshapes to six BEFORE selecting named arm joints. Correct
only that output shape; do not change planning, collision, constraints or RNG.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types
from ares_r.motion import production_scene_worker as original
from research.skill_handoff_lab.bench.oracle_executor import OracleExecutor


def main(a):
    source = Path(original.__file__).read_text()
    before = 'raw=plan.position.detach().cpu().numpy().reshape(-1,6)'
    after = 'raw=plan.position.detach().cpu().numpy().reshape(-1,len(plan.joint_names))'
    assert source.count(before) == 1, 'Fail closed if readonly worker version changed'
    patched = source.replace(before, after)
    module = types.ModuleType('ares_r.motion.production_scene_worker_research')
    module.__package__ = 'ares_r.motion'
    module.__file__ = __file__
    exec(compile(patched, str(Path(original.__file__)), 'exec'), module.__dict__)
    a.output.mkdir(mode=0o700, exist_ok=False)
    manifest = {'publication': 'CONTROLLED_NOT_PUBLIC', 'production_source_modified': False,
                'original_worker_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'research_worker_sha256': hashlib.sha256(patched.encode()).hexdigest(),
                'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'change': 'reshape using returned joint_names then retain existing ARM_NAMES selection',
                'planner_constraints_changed': False}
    (a.output/'WORKER_COMPATIBILITY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    executor=OracleExecutor();executor.worker=module
    for i,request in enumerate(json.loads(a.batch.read_text())):
        report=executor.execute(request,a.output/str(i))
        print(json.dumps({'job':i,'status':report['status'],'failure_class':report['failure_class'],'wall_s':report['wall_s']}),flush=True)
        if report['failure_class'] in ('SOFTWARE_ERROR','OOM','DRIVER_COMPATIBILITY'):break


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
