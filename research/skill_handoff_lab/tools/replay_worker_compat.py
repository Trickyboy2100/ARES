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
import time
from ares_r.motion import production_scene_worker as original
from research.skill_handoff_lab.bench.oracle_executor import OracleExecutor


def main(a):
    source = Path(original.__file__).read_text()
    before = 'raw=plan.position.detach().cpu().numpy().reshape(-1,6)'
    after = 'raw=plan.position.detach().cpu().numpy().reshape(-1,len(plan.joint_names))'
    assert source.count(before) == 1, 'Fail closed if readonly worker version changed'
    patched = source.replace(before, after)
    if a.reuse_goal_runtime:
        # Cache-only optimization. These values affect plan() or the goal, not
        # MotionPlannerCfg; retain seeds, collision grid and tool criteria.
        patched=patched.replace('"parameters":params,"orientation_lock":signature_orientation',
            '"parameters":{k:v for k,v in params.items() if k not in ("max_attempts",)},"orientation_lock":{k:v for k,v in (signature_orientation or {}).items() if k!="target_R_body_tcp"}')
    if a.exact_kd:
        begin=patched.index('    def clearance_by_sample(rows):')
        end=patched.index('    def clearance_details(rows):',begin)
        replacement='''    def clearance_by_sample(rows):
        from fast_proxy_collision import per_sample_clearances
        rows=np.asarray(rows,dtype=float).reshape(-1,6)
        current=planner.compute_kinematics(state(rows)).robot_spheres.detach().cpu().numpy().reshape(len(rows),-1,4)
        current=current[:,current[0,:,3]>0,:]
        return per_sample_clearances(current,cuboids)
'''
        patched=patched[:begin]+replacement+patched[end:]
        if a.gpu_post:
            patched=patched.replace('from fast_proxy_collision import per_sample_clearances','from fast_proxy_collision import gpu_per_sample_clearances')
            patched=patched.replace('current=planner.compute_kinematics(state(rows)).robot_spheres.detach().cpu().numpy().reshape(len(rows),-1,4)',
                'current=planner.compute_kinematics(state(rows)).robot_spheres.reshape(len(rows),-1,4)')
            patched=patched.replace('return per_sample_clearances(current,cuboids)','return gpu_per_sample_clearances(current,cuboids)')
    module = types.ModuleType('ares_r.motion.production_scene_worker_research')
    module.__package__ = 'ares_r.motion'
    module.__file__ = __file__
    exec(compile(patched, str(Path(original.__file__)), 'exec'), module.__dict__)
    kd_source=None
    if a.exact_kd:
        from fast_proxy_collision import independent_validator
        function,kd_source=independent_validator()
        module.validate_dense_world=function
        from ares_r.motion import independent_path_validation
        independent_path_validation.validate_dense_world=function
    a.output.mkdir(mode=0o700, exist_ok=False)
    manifest = {'publication': 'CONTROLLED_NOT_PUBLIC', 'production_source_modified': False,
                'original_worker_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'research_worker_sha256': hashlib.sha256(patched.encode()).hexdigest(),
                'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'change': 'reshape using returned joint_names then retain existing ARM_NAMES selection',
                'planner_constraints_changed': False,'cache_only_reuse_optimization':a.reuse_goal_runtime,
                'exact_kd':a.exact_kd,'planner_gpu_post':a.gpu_post,'kd_validator_sha256':hashlib.sha256(kd_source.encode()).hexdigest() if kd_source else None}
    (a.output/'WORKER_COMPATIBILITY_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    executor=OracleExecutor();executor.worker=module
    i=0;deadline=time.monotonic()+a.follow_seconds
    while True:
        batch=json.loads(a.batch.read_text())
        if i>=len(batch):
            if not a.follow_seconds or time.monotonic()>deadline or a.batch.with_suffix('.done').exists():break
            time.sleep(.5);continue
        request=batch[i]
        report=executor.execute(request,a.output/str(i))
        print(json.dumps({'job':i,'status':report['status'],'failure_class':report['failure_class'],'wall_s':report['wall_s']}),flush=True)
        if report['failure_class'] in ('SOFTWARE_ERROR','OOM','DRIVER_COMPATIBILITY'):break
        i+=1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-goal-runtime',action='store_true')
    p.add_argument('--exact-kd',action='store_true')
    p.add_argument('--gpu-post',action='store_true')
    p.add_argument('--follow-seconds',type=int,default=0)
    main(p.parse_args())
