"""CPU-prepared cube/held-tool clearance contract; not a validated new controller.

Pinned old mplib has one attached geometry and no pairwise allowed-collision
matrix. TRANSFER checks robot+attached mesh against cube. HOOK/PULL keep cube
against robot but disable attached checks for deliberate tool/cube contact.
This also disables other attached checks: never claim full-world validation.
"""
from pathlib import Path
import hashlib
import numpy as np
import mplib
import trimesh

def native_l_tool_mesh(path):
    path=Path(path)
    handle=trimesh.creation.box([.2,.05,.05]);handle.apply_translation([.1,0,0])
    hook=trimesh.creation.box([.05,.1,.05]);hook.apply_translation([.175,.05,0])
    # Concatenated native collision primitives, not a filled single AABB.
    mesh=trimesh.util.concatenate([handle,hook]);mesh.export(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bounds=mesh.bounds.tolist(),primitive_count=2)

class CubeClearanceContract:
    def __init__(self,planner,mesh_path):
        self.planner=planner;self.mesh_path=str(mesh_path);self.mode='PREFIX';self.audit=[]
    def synchronize(self,mode,cube_pose,tcp_T_tool):
        if mode not in ['PREFIX','TRANSFER','HOOK','PULL']:raise ValueError(mode)
        self.mode=mode;world=self.planner.planning_world
        world.remove_normal_object('native_target_cube')
        use_attach=mode=='TRANSFER'
        if mode!='PREFIX':
            pose=np.asarray(cube_pose,dtype=float);fcl=mplib.pymp.collision_detection.fcl
            world.set_normal_object('native_target_cube',fcl.CollisionObject(fcl.Box(.04,.04,.04),pose[:3],pose[3:]))
            self.planner.update_attached_mesh(self.mesh_path,np.asarray(tcp_T_tool,dtype=float))
        world.set_use_attach(use_attach)
        kwargs=dict(use_point_cloud=False,use_attach=use_attach)
        self.audit.append(dict(mode=mode,cube_pose=np.asarray(cube_pose).tolist(),tcp_T_tool=np.asarray(tcp_T_tool).tolist(),planner_kwargs=kwargs,scope='cube-vs-robot always in suffix; attached-vs-cube onlyTRANSFER; no full-world claim'))
        return kwargs

def audit_saved_robot_overlap(trace):
    """Static URDF/FCL check of saved end-of-control frames, no simulator."""
    import gzip,json
    trace=Path(trace);d=json.load(gzip.open(trace,'rt'))
    robot_root=Path('/home/andyee/robot-research/ARES-Skilllab/external/benchmark_scout_20261027/ManiSkill/mani_skill/assets/robots/panda')
    urdf=robot_root/'panda_v2.urdf';srdf=robot_root/'panda_v2.srdf'
    p=mplib.Planner(urdf=str(urdf),srdf=str(srdf),move_group='panda_hand_tcp')
    p.set_base_pose(np.array(next(iter(d['initial']['articulations'].values()))[0][:7]))
    fcl=mplib.pymp.collision_detection.fcl;rows=[]
    for s in d['steps']:
        pose=np.asarray(s['cube']);p.planning_world.set_normal_object('cube',fcl.CollisionObject(fcl.Box(.04,.04,.04),pose[:3],pose[3:]))
        hits=p.check_for_env_collision(qpos=np.array(s['q']),with_point_cloud=False,use_attach=False)
        rows.append(dict(step=s['step'],phase=s['phase'],robot_cube_overlap_pairs=len(hits)))
    return dict(trace=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),srdf_sha256=hashlib.sha256(srdf.read_bytes()).hexdigest(),frames=len(rows),overlap_frames=sum(x['robot_cube_overlap_pairs']>0 for x in rows),scope='CPU static end-frame URDF/FCL robot-versus-cube overlap only. Zero overlap does not imply no PhysX contact or no between-frame collision. Held tool excluded.',rows=rows)

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(audit_saved_robot_overlap(a.trace),indent=2))
