"""Bounded CPU-only public MPLib query diagnostic; zero new physics or deployment."""
import argparse,json,gzip,hashlib,time
from pathlib import Path
import numpy as np,mplib
p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--urdf',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=json.load(gzip.open(a.trace,'rt'));plan=next(z for z in d['plans']if z['phase']=='TRANSFER'and z.get('target'));q=np.array(d['steps'][plan['step']-1]['q']);base=list(d['initial']['articulations'].values())[0][0][:7]
planner=mplib.Planner(urdf=str(a.urdf),srdf=str(a.urdf.with_suffix('.srdf')),move_group='panda_hand_tcp');planner.set_base_pose(np.array(base));planner.joint_vel_limits=np.array(planner.joint_vel_limits)*1.8;planner.joint_acc_limits=np.array(planner.joint_acc_limits)*6;rows=[]
for method in ['screw','qpos','IK']:
 t=time.monotonic()
 try:
  if method=='screw':r=planner.plan_screw(np.array(plan['target']),q,time_step=.05,qpos_step=.01,wrt_world=True,use_point_cloud=False);status=r['status'];n=len(r.get('position',[]))
  elif method=='qpos':r=planner.plan_qpos_to_pose(np.array(plan['target']),q,time_step=.05,planning_time=.5,wrt_world=True,use_point_cloud=False);status=r['status'];n=len(r.get('position',[]))
  else:status,sol=planner.IK(planner.transform_goal_to_wrt_base(np.array(plan['target'])),q,n_init_qpos=8,threshold=1e-3);n=0 if sol is None else len(sol)
  rows.append(dict(method=method,status=status,result_count=n,wall_s=time.monotonic()-t))
 except RuntimeError as e:rows.append(dict(method=method,error=repr(e),wall_s=time.monotonic()-t))
a.out.write_text(json.dumps(dict(source=str(a.trace),sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),urdf_sha256=hashlib.sha256(a.urdf.read_bytes()).hexdigest(),queries=rows,scope='CPU-only saved state query, no new task rollout; no proof of globally infeasible target',new_physics_steps=0),indent=2));print(rows)
