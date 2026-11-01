"""CPU static intended-release geometry; not a counterfactual physics rollout."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import mplib

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 asset=Path('/home/andyee/robot-research/ARES-Skilllab/external/benchmark_scout_20261027/ManiSkill/mani_skill/assets/robots/panda');urdf=asset/'panda_v3.urdf';srdf=asset/'panda_v3.srdf';planner=mplib.Planner(urdf=str(urdf),srdf=str(srdf),move_group='panda_hand_tcp');fcl=mplib.pymp.collision_detection.fcl;rows=[]
 for path in sorted(a.root.glob('*/TRAJECTORY.json.gz')):
  d=json.load(gzip.open(path,'rt'));pre=[s for s in d['transitions'] if s['segment']=='A_PREPLACE'];place=[s for s in d['transitions'] if s['segment']=='A_PLACE']
  if not pre or not place:continue
  planner.set_base_pose(np.array(next(iter(d['initial']['articulations'].values()))[0][:7]));B=np.array(pre[-1]['state']['actors']['cubeB'][0][:7]);planner.planning_world.set_normal_object('support_B',fcl.CollisionObject(fcl.Box(.04,.04,.04),B[:3],B[3:]))
  q=np.array(place[-1]['q'][0]);q[-2:]=.04
  hits=planner.check_for_env_collision(qpos=q,with_point_cloud=False,use_attach=False)
  rows.append(dict(run=path.parent.name,success=d['summary']['success'],axis_degrees=d['summary']['axis_degrees'],axis_aware_release=d['summary'].get('axis_aware_release',False),B_pose_before_place=B.tolist(),arm_state_step=place[-1]['step'],intended_open_q=q.tolist(),pairs=[dict(link1=x.link_name1,link2=x.link_name2,object1=x.object_name1,object2=x.object_name2) for x in hits],overlap=len(hits)>0,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 data=dict(scope='CPU STATIC INTENDED-OPEN GEOMETRY ONLY. Recorded arm pose at endPLACE + fullyopen fingers0.04m each + recorded B beforePLACE. No new physics, no force attribution, no counterfactual success label.',urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),srdf_sha256=hashlib.sha256(srdf.read_bytes()).hexdigest(),rows=rows)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(data,indent=2));print(json.dumps([dict(run=x['run'],overlap=x['overlap'],pairs=x['pairs']) for x in rows],indent=2))
if __name__=='__main__':main()
