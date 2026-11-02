"""Single-factor C center refresh after approach; preserve all other primitives."""
import argparse,gzip,json,time
from pathlib import Path
from program_pyramid_joint_probe import Strong
from handoff_vector import *
class Refresh(Strong):
 def grasp(self,i,actor,name,yaw_offset=0.):
  gen=super().grasp(i,actor,name,yaw_offset)
  # Intercept only C_DESCEND target at move time; reach path and grasp angle unchanged.
  yield from gen
 def move(self,i,pose,label):
  if label=='C_DESCEND':
   previous=self.pose(self.u.cubeC,i)[:3]
   # Existing compute_grasp_info depth .025 with 40mm cube puts center at actor center.
   # Preserve exact commanded Z offset and orientation; update XY only.
   target=pose.p.copy();delta=previous[:2]-self.c_start[i][:2];target[:2]+=delta
   self.plans.append(dict(env=i,label='MEASURED_C_CENTER_REFRESH',step=self.steps,planning_wall_s=0.,xy_correction_m=delta.tolist()))
   pose=sapien.Pose(target,pose.q)
  yield from super().move(i,pose,label)
 def skill(self,i,stage,choice):
  if not hasattr(self,'c_start'):self.c_start={}
  if stage==1:self.c_start[i]=self.pose(self.u.cubeC,i)[:3].copy()
  yield from super().skill(i,stage,choice)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[];ex=Refresh(n=2,trace=True)
 try:
  for aa in range(3):
   for cc in range(3):
    np.random.seed(31130);torch.manual_seed(31130)
    r=ex.run([420002,420023],lambda obs,i,stage:([aa,cc][stage],{}));r.update(mode='refresh',a=aa,c=cc);path=a.out/f'refresh_A{aa}_C{cc}.json.gz'
    with gzip.open(path,'wt') as f:json.dump(dict(summary=r,initial=ex.initial,transitions=ex.rows,plans=ex.plans),f)
    r.update(path=str(path),sha256=sha(path));rows.append(r);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(aa,cc,r['success'],r['errors'],flush=True)
 finally:ex.close()
 for name in [Path(__file__).name,'program_pyramid_joint_probe.py','handoff_vector.py','handoff_physics.py']:(a.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
if __name__=='__main__':main()
