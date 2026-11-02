"""Bounded real-physics successor orientation interventions; no learning claim.

Native geometry, initial distribution and horizon. Additional yaw requirement is
an adaptation contract, reported separately from native success. No state replay
or state setters in physics. Inherits existing shared support/control primitives.
"""
import argparse,gzip,json,hashlib,time,random
from pathlib import Path
from handoff_physics import *
from transforms3d.quaternions import mat2quat

class ContractExecutor(HandoffExecutor):
    symmetry_aware=False
    feasible_symmetry=False
    def execute_top(self,offset,yaw):
        self.env.stage=self.stage
        self.grasp(self.u.cubeC,'C')
        A=npv(self.u.cubeA.pose.raw_pose);B=npv(self.u.cubeB.pose.raw_pose)
        axis=A[:2]-B[:2];axis/=max(np.linalg.norm(axis),1e-6)
        goal=(A[:3]+B[:3])/2;goal[:2]+=axis*offset*.008;goal[2]=max(A[2],B[2])+.043
        tcpR=quat2mat(npv(self.u.agent.tcp.pose.q));objR=quat2mat(npv(self.u.cubeC.pose.q))
        attachedR=tcpR.T@objR
        t=np.deg2rad(yaw);desired=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
        if self.symmetry_aware:
            choices=[]
            for quarter in range(4):
                angle=t+quarter*np.pi/2
                desired_q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
                rotation=desired_q@attachedR.T
                choices.append((np.trace(tcpR.T@rotation),desired_q))
            desired=max(choices,key=lambda x:x[0])[1]
        if self.feasible_symmetry:
            feasible=[];self.orientation_queries=[]
            local=tcpR.T@(npv(self.u.cubeC.pose.p)-npv(self.u.agent.tcp.pose.p))
            for quarter in range(4):
                angle=t+quarter*np.pi/2
                obj=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
                rot=obj@attachedR.T;q=mat2quat(rot);pos=goal-rot@local
                pre=sapien.Pose(pos+[0,0,.07],q)
                result=self.planner.move_to_pose_with_screw(pre,dry_run=True)
                valid=not isinstance(result,int);cost=None
                if valid:
                    target=self.planner._transform_pose_for_planning(sapien.Pose(pos,q))
                    start=np.r_[result['position'][-1],npv(self.u.agent.robot.get_qpos())[-2:]]
                    after=self.planner.planner.plan_screw(np.r_[target.p,target.q],start,time_step=self.u.control_timestep,use_point_cloud=False)
                    valid=after['status']=='Success'
                    if valid:cost=len(result['position'])+len(after['position']);feasible.append((cost,obj))
                self.orientation_queries.append(dict(quarter=quarter,valid=valid,native_path_points=cost))
            if not feasible:raise SkillFailure('NO_EQUIVALENT_PREPLACE_PLACE_PATH')
            desired=min(feasible,key=lambda x:x[0])[1]
        targetR=desired@attachedR.T;targetq=mat2quat(targetR)
        local=tcpR.T@(npv(self.u.cubeC.pose.p)-npv(self.u.agent.tcp.pose.p))
        place=sapien.Pose(goal-targetR@local,targetq)
        self.move(sapien.Pose(place.p+[0,0,.07],targetq),'C_PREPLACE');self.move(place,'C_PLACE')
        self.release_verify(self.u.cubeC,goal,place,'C',False);self.stage=2

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--feasible-symmetry',action='store_true');p.add_argument('--symmetry-aware',action='store_true');p.add_argument('--seeds',type=int,nargs='+',default=[744000]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in [Path(__file__).name,'handoff_physics.py','method_stack_physics.py']:
        src=Path(__file__).with_name(name);(a.out/name).write_bytes(src.read_bytes());sources[name]=sha(src)
    (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task='ARES-PyramidOrientation-adaptation',native='StackPyramid-v1',horizon=250,seeds=a.seeds,split='NEW_TRAIN_DESIGN_NOT_SEALED',support_gap_normalized=[-1,0,1],top_offset_normalized=[-1,0,1],successor_yaw_degrees=[0,45],additional_yaw_tolerance_deg=10,geometry='native unchanged',controller='shared HandoffExecutor stride3 with six settling updates; actual attachment compensation for desired object yaw',competing_mechanisms=['support footprint changes available top support under requested yaw','grasp/tracking/controller failures independent of successor contract'],optimizer_updates=0,symmetry_aware=a.symmetry_aware,feasible_symmetry=a.feasible_symmetry,source_sha256=sources),indent=2))
    rows=[];ex=ContractExecutor();ex.symmetry_aware=a.symmetry_aware;ex.feasible_symmetry=a.feasible_symmetry
    try:
        for seed in a.seeds:
            for gap in [-1,0,1]:
                for yaw in [0,45]:
                    for offset in [-1,0,1]:
                        random.seed(31120);np.random.seed(31120);torch.manual_seed(31120)
                        start=time.monotonic();ex.orientation_queries=[];obs=ex.reset(seed);dec=[dict(boundary_step=0,parameters=dict(support_gap_normalized=gap),observation=obs.tolist())];error=None
                        try:
                            ex.execute([gap]);dec.append(dict(boundary_step=ex.env.steps,parameters=dict(top_offset_normalized=offset,desired_object_yaw_deg=yaw),observation=ex.observe().tolist(),previous_skill_result=ex.boundaries[-1]));ex.execute_top(offset,yaw)
                        except (BudgetEnd,SkillFailure) as e:error=str(e) or 'NATIVE_HORIZON'
                        while ex.env.steps<250:ex.hold(1,'TERMINAL_HOLD')
                        R=quat2mat(npv(ex.u.cubeC.pose.q));actual=float(np.degrees(np.arctan2(R[1,0],R[0,0])));yawerr=abs((actual-yaw+45)%90-45)
                        native=bool(ex.u.evaluate()['success'].item());r=dict(seed=seed,gap=gap,yaw=yaw,offset=offset,success=native,native_success=native,contract_success=bool(native and yawerr<=10),yaw_error_deg=yawerr,error=error,orientation_queries=ex.orientation_queries,steps=ex.env.steps,initial_sha256=ex.env.initial_sha,boundaries=ex.boundaries,planning_wall_s=sum(x['planning_wall_s'] for x in ex.plans),wall_s=time.monotonic()-start)
                        path=a.out/f'{seed}_g{gap}_y{yaw}_o{offset}.json.gz'
                        with gzip.open(path,'wt') as f:json.dump(dict(summary=r,initial=ex.env.initial,transitions=ex.env.rows,plans=ex.plans,decisions=dec),f)
                        r.update(path=str(path),sha256=sha(path));rows.append(r);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['boundaries','initial_sha256','sha256']}),flush=True)
    finally:ex.close()
if __name__=='__main__':main()
