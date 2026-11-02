"""Read every prior intervention/negative trace; verify hashes and physical labels."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np

def rotation(q):
 w,x,y,z=np.asarray(q)/np.linalg.norm(q);return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();groups={};failures=[];seen=set()
 for f in sorted(a.root.rglob('RESULTS.json')):
  data=json.loads(f.read_text())
  if not isinstance(data,list):continue
  records=[]
  for row in data:
   path=Path(row.get('path',''))
   if not path.is_file()or path.suffix!='.gz':continue
   if str(path)in seen:continue
   seen.add(str(path));actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==row['sha256'],str(path)
   d=json.load(gzip.open(path,'rt'));s=d['steps'];assert len(s)==100;assert bool(s[-1]['native_success'])==bool(row['success'])
   contacts=[z['step']for z in s if max([v['force_norms_N']['held_tool']for v in z.get('physics_substep_contacts',[])]+[z['tool_cube_force_N']])>.5]
   hook=next((z for z in reversed(s)if z['phase']=='HOOK'),None);g=None
   if hook:
    local=rotation(hook['tool'][3:]).T@(np.array(hook['cube'][:3])-hook['tool'][:3]);g=dict(step=hook['step'],remaining=100-hook['step'],cube_in_tool_m=local.tolist(),tool_tilt_deg=float(np.rad2deg(np.arccos(np.clip(rotation(hook['tool'][3:])[2,2],-1,1)))) )
   rec=dict(seed=row['seed'],grasp=row.get('grasp'),success=bool(row['success']),error=row.get('error'),first_force_step=next(iter(contacts),None),contact_steps=len(contacts),hook=g,final_distance_m=s[-1]['cube_base_distance_m'],path=str(path),sha256=actual)
   records.append(rec)
   if not row['success']and ('eval_matched_prefix/fixed875'in str(path) or 'design/'in str(path)):failures.append(rec)
  if records:groups[str(f.relative_to(a.root))]=dict(episodes=len(records),success=sum(z['success']for z in records),errors=dict(collections.Counter(str(z['error'])for z in records)),records=records)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(source=str(a.root),verified_unique_full_episodes=len(seen),groups=groups,nominal_failures=failures,no_new_physics=True),indent=2));print('verified',len(seen),'full trajectories;',len(groups),'groups')
if __name__=='__main__':main()
