"""Retain legacy two real parameters; isolate equivalent-place planning.
Old scene IDs are diagnosis, never evaluation data. No object state injection.
"""
import argparse,gzip,json,time,hashlib
from pathlib import Path
from handoff_vector import *
from transforms3d.quaternions import mat2quat
class Strong(VectorHandoff):
 def skill(self,i,stage,choice):
  if stage==0:yield from super().skill(i,stage,choice);return
  start=self.steps;u=self.u;actor=u.cubeC;name='C';request=float(ACTIONS[choice]);yield from self.grasp(i,actor,name,yaw_offset=request*.35)
  A,B=[self.pose(x,i) for x in [u.cubeA,u.cubeB]];goal=(A[:3]+B[:3])/2;goal[2]=max(A[2],B[2])+.043
  current=quat2mat(self.vec(u.agent.tcp.pose.q,i));local=current.T@(self.xyz(actor,i)-self.vec(u.agent.tcp.pose.p,i));base=quat2mat(u.agent.build_grasp_pose(np.array([0,0,-1]),np.array([0,1,0]),np.zeros(3)).q);valid=[];queries=[]
  for k in range(4):
   t=k*np.pi/2;R=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])@base;q=mat2quat(R);pos=goal-R@local;qq=self.q(i);cost=0;ok=True
   for z in [.07,0,.06]:
    r=self.planners[i].plan_screw(np.r_[pos+[0,0,z],q],qq,time_step=.05,use_point_cloud=False)
    if r['status']!='Success':ok=False;break
    cost+=len(r['position']);qq=np.r_[r['position'][-1],qq[-2:]]
   queries.append(dict(quarter=k,valid=ok,points=cost))
   if ok:valid.append((cost,pos,q))
  self.plans.append(dict(env=i,label='SYMMETRY_QUERY',step=self.steps,planning_wall_s=0.,candidates=queries))
  if not valid:raise SkillFailure('NO_FEASIBLE_SYMMETRIC_PLACE')
  _,pos,q=min(valid,key=lambda x:x[0]);place=sapien.Pose(pos,q)
  yield from self.move(i,sapien.Pose(pos+[0,0,.07],q),'C_PREPLACE');yield from self.move(i,place,'C_PLACE');self.grip[i]=1;yield from self.hold(i,6,'C_OPEN');yield from self.move(i,sapien.Pose(pos+[0,0,.06],q),'C_RETREAT')
  consecutive=0
  for _ in range(12):
   yield from self.hold(i,1,'C_SETTLE');r=self.verify(i,actor,goal,1);consecutive=consecutive+1 if r['valid'] else 0
   if consecutive>=3:break
  r.update(valid=consecutive>=3,consecutive_valid_steps=consecutive,requested=request,applied_rad=request*.35,goal=goal.tolist(),physical_steps=self.steps-start);self.boundaries[i].append(r)
  if not r['valid']:raise SkillFailure('C_BOUNDARY_NOT_VERIFIED')
  self.option_records[i].append(r)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
 for mode,cls in [('legacy',VectorHandoff),('strong',Strong)]:
  ex=cls(n=2,trace=True)
  try:
   for aa in range(3):
    for cc in range(3):
     np.random.seed(31130);torch.manual_seed(31130)
     r=ex.run([420002,420023],lambda obs,i,stage:([aa,cc][stage],{}));r.update(mode=mode,a=aa,c=cc)
     path=a.out/f'{mode}_A{aa}_C{cc}.json.gz'
     with gzip.open(path,'wt') as f:json.dump(dict(summary=r,initial=ex.initial,transitions=ex.rows,plans=ex.plans),f)
     r.update(path=str(path),sha256=sha(path));rows.append(r);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(mode,aa,cc,r['success'],r['errors'],flush=True)
  finally:ex.close()
 for name in [Path(__file__).name,'handoff_vector.py','handoff_physics.py']:(a.out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
if __name__=='__main__':main()
