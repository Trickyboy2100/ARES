"""Persistent planning-only executor; never imports a hardware command API.

Run one process per GPU. Requests/results live in immutable per-job directories.
ARES-R's worker is imported read-only and its runtime cache survives each job.
Software failures and missing evidence are UNKNOWN, not failed RL transitions.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, sys, time, traceback
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def failure_class(error):
    value=str(error).lower()
    if "out of memory" in value: return "OOM"
    if "unsupported ptx" in value or "driver" in value: return "DRIVER_COMPATIBILITY"
    return "SOFTWARE_ERROR"


class OracleExecutor:
    def __init__(self):
        from ares_r.motion import production_scene_worker
        self.worker=production_scene_worker

    def execute(self, request_path, directory):
        request_path=Path(request_path).resolve(); directory=Path(directory).resolve()
        request=json.loads(request_path.read_text())
        if request.get("planning_only") is not True or request.get("execution_allowed") is not False:
            raise ValueError("planning-only execution-blocked request required")
        directory.mkdir(parents=True,exist_ok=False)
        frozen=directory/"request.json"; frozen.write_bytes(request_path.read_bytes())
        result_path=directory/"result.json"; start=time.perf_counter()
        error=None; old_argv=sys.argv
        with (directory/"worker.log").open("w") as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:
                    sys.argv=["production_scene_worker",str(frozen),str(result_path)]
                    self.worker.main()
                except SystemExit as exc:
                    if exc.code and not result_path.exists(): error=repr(exc)
                except Exception as exc:
                    error=repr(exc); traceback.print_exc()
                finally: sys.argv=old_argv
        result=json.loads(result_path.read_text()) if result_path.exists() else {}
        independent=result.get("independent_dense_validation") or {}
        points=result.get("trajectory_points_rad") or []
        if len(points)>=2:
            import numpy as np
            from ares_r.motion.independent_path_validation import validate_dense_world
            # ARES-R only calls its independent validator when an execution tool
            # envelope is present. Research always performs the second check.
            independent=validate_dense_world(points,request,subdivisions=2)
            limits=request.get("site_soft_limits")
            limits_ok=None
            if limits:
                rows=np.asarray(points);limits_ok=bool(np.all(rows>=np.asarray(limits['lower_rad']))
                    and np.all(rows<=np.asarray(limits['upper_rad'])))
        else: limits_ok=None
        # Planner success alone never becomes an independently validated label.
        validation={"collision":independent.get("collision_free"),
            "attached_object":bool(request.get("attached_object_collision")) if independent else None,
            "orientation":(result.get("orientation_validation") or {}).get("passed"),
            "soft_limits":limits_ok,"self_collision":None,
            "scene_binding":result.get("scene_digest")==request.get("scene_digest") if result else None}
        status="SOFTWARE_ERROR" if error else (
            "PLANNER_PATH_REQUIRES_QUALIFICATION" if result.get("observed_result")=="SUCCESS"
            else "PLANNER_NO_VALIDATED_PATH")
        report={"status":status,"failure_class":failure_class(error) if error else None,
            "error":error,"label":None,"planning_only":True,"execution_allowed":False,
            "wall_s":time.perf_counter()-start,"independent_validation":validation,
            "executor_sha256":sha256(__file__),"ares_r_worker_sha256":sha256(self.worker.__file__) if hasattr(self.worker,'__file__') else 'ARTIFICIAL_FIXTURE',
            "request_sha256":sha256(frozen),"source_artifacts":[
                {"role":"request","path":str(frozen),"sha256":sha256(frozen)},
                {"role":"log","path":str(directory/"worker.log"),"sha256":sha256(directory/"worker.log")}],
            "independent_dense_validation":independent,
            "missing_qualification":[key for key,value in validation.items() if value is not True]}
        if result_path.exists(): report["source_artifacts"].append(
            {"role":"result","path":str(result_path),"sha256":sha256(result_path)})
        (directory/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
        return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--batch",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    jobs=json.loads(args.batch.read_text()); executor=OracleExecutor()
    for index,path in enumerate(jobs):
        report=executor.execute(path,args.output/str(index))
        print(json.dumps({"job":index,**report}),flush=True)
        if report["failure_class"] in ("OOM","DRIVER_COMPATIBILITY","SOFTWARE_ERROR"):
            break  # Bounded failure: do not repeat the same installation/runtime error.


if __name__=="__main__": main()
