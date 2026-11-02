"""One event vocabulary across old and new measured tool experiments.
20Hz pose vs100Hz force are never silently promoted to matching timestamps.
"""
import json,gzip,hashlib,argparse,collections
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def audit(path):
 d=json.load(gzip.open(path,'rt'));steps=d.get('steps',d.get('records',[]));s=d.get('summary',{});results=d.get('skill_results',[])
 if not steps:return None
 boundary=results[0]['boundary_step'] if results else next((z['step'] for z in reversed(steps) if z['phase']=='LIFT'),None)
 if boundary is None:return None
 seq=[z for z in steps if z['step']>=boundary];ref=seq[0]['tcp_T_tool'];rot=Rotation.from_quat(np.asarray([z['tcp_T_tool'][3:]for z in seq])[:,[1,2,3,0]]);angle=(rot[0].inv()*rot).magnitude()*180/np.pi;drift=np.linalg.norm(np.array([z['tcp_T_tool'][:3] for z in seq])-ref[:3],axis=1)*1000
 sub=[z for x in seq for z in x.get('physics_substep_contacts',[]) if z['control_step']>boundary];first=next((z for z in sub if z['force_norms_N'].get('held_tool',0)>.5),None)
 if first is None:first=next((dict(control_step=z['step'],force20Hz_N=z['tool_cube_force_N'])for z in seq if z['step']>boundary and z.get('tool_cube_force_N',0)>.5),None)
 before=[i for i,z in enumerate(seq) if first is None or z['step']<first['control_step']];plans=[z for z in d.get('plans',[]) if z['phase'] in ['TRANSFER','HOOK','PULL']];phases={}
 for phase in ['TRANSFER','HOOK','PULL']:
  zz=[z for z in steps if z['phase']==phase]
  if zz:phases[phase]=dict(start_step=zz[0]['step'],end_step=zz[-1]['step'],cube_displacement_m=(np.array(zz[-1]['cube'][:3])-np.array(zz[0]['cube'][:3])).tolist(),cube_base_end_m=zz[-1]['cube_base_distance_m'])
 return dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),boundary=boundary,boundary_verified=bool(results),boundary_definition='verified stable PICK_TOOL' if results else 'end_LIFT surrogate; no verified stability in source',first_tool_cube_contact=first,precontact_rotation_deg=float(max(angle[before])) if before else None,precontact_translation_mm=float(max(drift[before])) if before else None,rotation_peak_deg=float(max(angle)),translation_peak_mm=float(max(drift)),native_success=bool(steps[-1]['native_success']),terminal_distance_m=steps[-1]['cube_base_distance_m'],remaining_at_boundary=100-boundary,phases=phases,arrival=[dict(phase=z['phase'],endpoint=z.get('endpoint_fk'),arrival=z.get('arrival'),attempts=z.get('attempts'))for z in plans],contact_force_threshold_N=.5,pose_rate_Hz=20,force_rate_Hz=100 if sub else 20,causal_limit='precontact rotation uses last20Hz state strictly before first detected force step, not an exact100Hz pose measurement')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
 for folder in ['pilot','feedback']:
  for r in json.loads((a.root/folder/'RESULTS.json').read_text()):
   z=audit(Path(r['path']));z.update(group=folder,seed=r['seed'],mass=r['mass'],grasp=r['grasp'],mode=r['mode']);rows.append(z)
 legacy=a.root.parent/'multiskill_slowdown'
 for path in legacy.glob('*/TRAJECTORY.json.gz'):
  z=audit(path)
  if z:rows.append(dict(group='historical_slowdown',**z))
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'UNIFIED_EVENTS.json').write_text(json.dumps(rows,indent=2));print([(r.get('group'),r.get('seed'),r.get('mode'),r['precontact_rotation_deg'],r['native_success'])for r in rows])
if __name__=='__main__':main()
