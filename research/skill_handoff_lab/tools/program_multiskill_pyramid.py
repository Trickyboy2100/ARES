"""Native StackPyramid placement-axis scout using existing verified skills.

Two actions: support placement axis, then actual-support-relative top offset.
No changed task geometry, success criteria, horizon, or inter-skill reset.
"""
import argparse,gzip,hashlib,json,time,random,subprocess
from pathlib import Path
from handoff_physics import *

class AxisExecutor(HandoffExecutor):
    axis_aware_release=False
    selected_axis=0.
    def placement_q(self,angle):
        closing=np.array([-np.sin(angle),np.cos(angle),0.]) if self.axis_aware_release and abs(angle)>1e-8 else np.array([0,1,0])
        return self.u.agent.build_grasp_pose(np.array([0,0,-1]),closing,np.zeros(3)).q
    def grasp_q(self,actor):
        g=compute_grasp_info_by_obb(obb(actor),approaching=np.array([0,0,-1]),target_closing=np.array([0,1,0]),depth=.025)
        return self.u.agent.build_grasp_pose(np.array([0,0,-1]),g['closing'],g['center']).q
    def support_goal(self,angle,predicted_grasp_q=None):
        B=npv(self.u.cubeB.pose.raw_pose);axis=np.array([np.cos(angle),np.sin(angle),0.])
        # Projection of two native square footprints onto the chosen axis.
        a_projection=abs(axis[:2]).sum()
        if self.axis_aware_release and abs(angle)>1e-8:
            source_q=npv(self.u.agent.tcp.pose.q) if predicted_grasp_q is None else predicted_grasp_q
            predicted_R=quat2mat(self.placement_q(angle))@quat2mat(source_q).T@quat2mat(npv(self.u.cubeA.pose.q))
            a_projection=abs(predicted_R.T@axis).sum()
        radius=.02*(a_projection+abs(quat2mat(B[3:]).T@axis).sum())
        goal=B[:3]+axis*(radius+.003);goal[2]=.022
        return goal
    def axis_search(self):
        search_start=time.monotonic()
        u=self.u;start=npv(u.agent.robot.get_qpos());A=npv(u.cubeA.pose.p);C=npv(u.cubeC.pose.p)
        q=u.agent.build_grasp_pose(np.array([0,0,-1]),np.array([0,1,0]),np.zeros(3)).q
        candidates=[]
        # Same kinematic planner for every candidate. No future physics labels.
        for degrees in [0,90,-90,180]:
            angle=np.deg2rad(degrees);enhanced=self.axis_aware_release and degrees!=0
            q_place=self.placement_q(angle);q_a=self.grasp_q(u.cubeA) if enhanced else q;q_c=self.grasp_q(u.cubeC) if enhanced else q
            goal=self.support_goal(angle,predicted_grasp_q=q_a);top=(goal+npv(u.cubeB.pose.p))/2;top[2]=.063
            qq=start.copy();cost=0;valid=True
            # SAT occupancy test against the untouched top cube at its actual pose.
            Caxes=quat2mat(npv(u.cubeC.pose.q))[:2,:2];delta=goal[:2]-C[:2]
            test_axes=[np.array([1.,0.]),np.array([0.,1.]),Caxes[:,0],Caxes[:,1]]
            occupied=all(abs(delta@axis)<=.02*(abs(axis).sum()+abs(Caxes.T@axis).sum())+.003 for axis in test_axes)
            if occupied:
                candidates.append(dict(axis_degrees=degrees,valid=False,predicted_motion_steps=0,goal=goal.tolist(),failure='SUPPORT_TARGET_OCCUPIED_BY_C'))
                continue
            # Include both skill approaches; gripper center approximately cube center.
            targets=[A+[0,0,.06],A,A+[0,0,.10],goal+[0,0,.09],goal,goal+[0,0,.06],C+[0,0,.06],C,C+[0,0,.10],top+[0,0,.07],top,top+[0,0,.06]]
            target_qs=[q_a]*3+[q_place]*3+[q_c]*3+[q_place]*3
            for xyz,orientation in zip(targets,target_qs):
                pose=sapien.Pose(xyz,orientation);r=self.planner.planner.plan_screw(np.r_[pose.p,pose.q],qq,time_step=u.control_timestep,use_point_cloud=False)
                if r['status']!='Success':valid=False;break
                cost+=int(np.ceil(len(r['position'])/3));qq=np.r_[r['position'][-1],qq[-2:]]
            candidates.append(dict(axis_degrees=degrees,valid=valid,predicted_motion_steps=cost,goal=goal.tolist()))
        good=[x for x in candidates if x['valid']]
        chosen=min(good,key=lambda x:x['predicted_motion_steps'])['axis_degrees'] if good else 0
        return chosen,candidates,time.monotonic()-search_start
    def execute_axis(self,degrees):
        start=self.env.steps;self.env.stage=0;self.stage=0;u=self.u
        self.selected_axis=degrees;self.grasp(u.cubeA,'A');goal=self.support_goal(np.deg2rad(degrees))
        target_q=self.placement_q(np.deg2rad(degrees))
        local=quat2mat(npv(u.agent.tcp.pose.q)).T@(npv(u.cubeA.pose.p)-npv(u.agent.tcp.pose.p));offset=-quat2mat(target_q)@local
        self.move(sapien.Pose(goal+offset+[0,0,.09],target_q),'A_PREPLACE');place=sapien.Pose(goal+offset,target_q)
        self.move(place,'A_PLACE');self.release_verify(u.cubeA,goal,place,'A',True)
        self.stage=1
        return dict(stage=0,axis_degrees=degrees,goal=goal.tolist(),physical_steps=self.env.steps-start,skill_result=self.boundaries[-1])

    def execute(self,action):
        if not self.axis_aware_release or self.selected_axis==0:return super().execute(action)
        request=np.asarray(action,dtype=float);assert request.shape==(1,);clipped=np.clip(request,-1,1);start=self.env.steps;wall=time.monotonic();self.env.stage=self.stage;u=self.u
        self.grasp(u.cubeC,'C');A=npv(u.cubeA.pose.raw_pose);B=npv(u.cubeB.pose.raw_pose);axis=A[:2]-B[:2];axis/=max(np.linalg.norm(axis),1e-6);goal=(A[:3]+B[:3])/2;goal[:2]+=axis*clipped[0]*.008;goal[2]=max(A[2],B[2])+.043
        target_q=self.placement_q(np.arctan2(axis[1],axis[0]))
        local=quat2mat(npv(u.agent.tcp.pose.q)).T@(npv(u.cubeC.pose.p)-npv(u.agent.tcp.pose.p));offset=-quat2mat(target_q)@local
        self.move(sapien.Pose(goal+offset+[0,0,.07],target_q),'C_PREPLACE');place=sapien.Pose(goal+offset,target_q);self.move(place,'C_PLACE');self.release_verify(u.cubeC,goal,place,'C',False)
        record=dict(stage=self.stage,request=request.tolist(),clipped=clipped.tolist(),applied_m=float(clipped[0]*.008),goal=goal.tolist(),physical_steps=self.env.steps-start,wall_s=time.monotonic()-wall,skill_result=self.boundaries[-1]);self.stage+=1;return self.observe(),record

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--axis',type=float);p.add_argument('--top-offset',type=float,default=0);p.add_argument('--label',default='axis_search');p.add_argument('--axis-aware-release',action='store_true');a=p.parse_args()
    if a.axis is not None and a.axis not in [-90,0,90,180]:p.error('axis must be -90,0,90,180 degrees')
    if not -1<=a.top_offset<=1:p.error('normalized top-offset must lie in [-1,1]')
    a.out.mkdir(parents=True,exist_ok=False);random.seed(31031);np.random.seed(31031);torch.manual_seed(31031);root=Path(__file__).parent;sources={}
    for name in ['program_multiskill_pyramid.py','handoff_physics.py','method_stack_physics.py']:
        data=(root/name).read_bytes();sources[name]=hashlib.sha256(data).hexdigest();(a.out/name).write_bytes(data)
    pinned=Path('/home/andyee/robot-research/ARES-Skilllab/external/benchmark_scout_20261027/ManiSkill');native=pinned/'mani_skill/envs/tasks/tabletop/stack_pyramid.py'
    (a.out/'MANIFEST.json').write_text(json.dumps(dict(task='StackPyramid-v1',native_horizon=250,seed=a.seed,label=a.label,requested_axis_degrees=a.axis,requested_top_offset=a.top_offset,axis_aware_release=a.axis_aware_release,rule_scope='deterministic controller repair; not RL; axis0exactlegacy; nonzeroaxes use perpendicular placing grip and actual attachment footprint',native_source_sha256=sha(native),maniskill_revision=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip(),maniskill_license_sha256=sha(pinned/'LICENSE'),random_seed=31031,initial_distribution='NATIVE_UNCHANGED',source_sha256=sources,hardware_commands=False,optimizer_updates=0,parameter_authority='support axis four cardinal directions; top offset +-8mm',search_scope='full two-skill kinematic approach cost, predicted attachment; not contact outcome oracle'),indent=2))
    ex=AxisExecutor();ex.axis_aware_release=a.axis_aware_release;start=time.monotonic();decisions=[];options=[];error=None
    try:
        obs=ex.reset(a.seed);chosen,candidates,search_wall=ex.axis_search();axis=chosen if a.axis is None else a.axis
        decisions.append(dict(boundary_step=0,skill='PLACE_SUPPORT',parameters=dict(axis_degrees=axis),observation=obs.tolist(),selection='full_suffix_kinematic_axis_search' if a.axis is None else 'fixed_axis_intervention',candidates=candidates,search_wall_s=search_wall,next_skill_requirement='released stable support pair, feasible top placement'))
        try:
            options.append(ex.execute_axis(axis))
            decisions.append(dict(boundary_step=ex.env.steps,skill='PLACE_TOP',parameters=dict(top_offset_m=a.top_offset*.008,frame='measured A-minus-B normalized XY axis'),observation=ex.observe().tolist(),previous_skill_result=ex.boundaries[-1],next_skill_requirement='native released stable pyramid by250'))
            _,r=ex.execute([a.top_offset]);options.append(r)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}' or 'NATIVE_HORIZON'
        # Preserve native full-terminal evaluation, including post-completion drift.
        while ex.env.steps<250:ex.hold(1,'TERMINAL_HOLD')
        success=bool(ex.u.evaluate()['success'].item())
        row=dict(seed=a.seed,label=a.label,axis_aware_release=a.axis_aware_release,success=success,error=error,steps=ex.env.steps,initial_sha256=ex.env.initial_sha,decisions=len(decisions),axis_degrees=axis,top_offset_m=a.top_offset*.008,search_wall_s=search_wall,boundaries=ex.boundaries,options=options,wall_s=time.monotonic()-start,planning_wall_s=sum(x['planning_wall_s'] for x in ex.plans))
        path=a.out/'TRAJECTORY.json.gz';payload=json.dumps(dict(summary=row,initial=ex.env.initial,transitions=ex.env.rows,plans=ex.plans,decisions=decisions),separators=(',',':'))
        with gzip.open(path,'wt') as f:f.write(payload)
        row.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());(a.out/'RESULTS.json').write_text(json.dumps([row],indent=2));print(json.dumps(row),flush=True)
    finally:ex.close()
if __name__=='__main__':main()
