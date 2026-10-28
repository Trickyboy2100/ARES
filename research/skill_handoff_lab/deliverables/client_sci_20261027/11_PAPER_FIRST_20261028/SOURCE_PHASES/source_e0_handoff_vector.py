"""Batched native physics, independent MPLib planning and verified option results.

The scheduler interleaves real servo programs; it never substitutes outcomes or
sets states between options. All sixteen environments physically run250 steps.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch,gymnasium as gym,sapien,mplib
import mani_skill.envs
from transforms3d.quaternions import quat2mat
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb
from handoff_physics import SkillFailure,serial,sha

ACTIONS=np.array([-1.,0.,1.])
class VectorHandoff:
    def __init__(self,n=16,trace=False):
        self.n=n;self.trace=trace;self.env=gym.make('StackPyramid-v1',num_envs=n,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1);self.u=self.env.unwrapped;self.planners=[];self.writer=None;self.plan_cache={}
    def vec(self,t,i):return t[i].detach().cpu().numpy().copy()
    def xyz(self,actor,i):return self.vec(actor.pose.p,i)
    def pose(self,actor,i):return self.vec(actor.pose.raw_pose,i)
    def q(self,i):return self.vec(self.u.agent.robot.get_qpos(),i)
    def reset(self,seeds):
        self.env.reset(seed=seeds);u=self.u;self.seeds=seeds;self.steps=0;self.rows=[];self.plans=[];self.option_records=[[] for _ in seeds];self.decisions=[[] for _ in seeds];self.boundaries=[[] for _ in seeds];self.stage=np.zeros(self.n,int);self.errors=[None]*self.n;self.grip=np.ones(self.n);self.initial=serial(u.get_state_dict());self.initial_sha=hashlib.sha256(json.dumps(self.initial,sort_keys=True).encode()).hexdigest();self.start=[self.pose(x,i) for i in range(self.n) for x in [u.cubeA,u.cubeB,u.cubeC]]
        if not self.planners:
            for i in range(self.n):
                p=mplib.Planner(urdf=u.agent.urdf_path,srdf=u.agent.urdf_path.replace('.urdf','.srdf'),user_link_names=[x.get_name() for x in u.agent.robot.get_links()],user_joint_names=[x.get_name() for x in u.agent.robot.get_active_joints()],move_group='panda_hand_tcp');p.set_base_pose(self.pose(u.agent.robot,i));p.joint_vel_limits=np.asarray(p.joint_vel_limits)*.9;p.joint_acc_limits=np.asarray(p.joint_acc_limits)*.9;self.planners.append(p)
    def observe(self,i):
        u=self.u;A,B,C=[self.pose(x,i) for x in [u.cubeA,u.cubeB,u.cubeC]];tcp=self.vec(u.agent.tcp.pose.raw_pose,i);q=self.q(i);v=self.vec(u.agent.robot.get_qvel(),i);axis=A[:2]-B[:2];dist=np.linalg.norm(axis);axis=axis/max(dist,1e-6)
        geom=np.r_[A[:3]-B[:3],C[:3]-(A[:3]+B[:3])/2,tcp[:3]-C[:3],A[3:],B[3:],C[3:],q,v]
        effect=np.r_[A[:3]-self.start[i*3][:3],B[:3]-self.start[i*3+1][:3],C[:3]-self.start[i*3+2][:3]]
        contract=np.r_[dist,np.sqrt(.04**2+.04**2)+.005-dist,axis,.02,.01,.5,(250-self.steps)/250]
        return np.r_[geom,effect,contract,float(self.stage[i])].astype(np.float32)
    def move(self,i,pose,label):
        p=self.planners[i];q=self.q(i);start=time.monotonic();state=np.r_[q,self.vec(self.u.agent.robot.get_qvel(),i),*[np.r_[self.pose(x,i),self.vec(x.linear_velocity,i),self.vec(x.angular_velocity,i)] for x in [self.u.cubeA,self.u.cubeB,self.u.cubeC]]];target=np.r_[pose.p,pose.q];key=(self.seeds[i],label,tuple(d['action'] for d in self.decisions[i]));cached=self.plan_cache.get(key);cache_hit=cached is not None and np.max(abs(cached[0]-state))<2e-6 and np.max(abs(cached[1]-target))<2e-6
        result=cached[2] if cache_hit else p.plan_screw(target,q,time_step=.05,use_point_cloud=False)
        if result['status']!='Success':result=p.plan_screw(np.r_[pose.p,pose.q],q,time_step=.05,use_point_cloud=False)
        fallback=False
        if result['status']!='Success':
            fallback=True;result=p.plan_qpos_to_pose(np.r_[pose.p,pose.q],q,time_step=.05,use_point_cloud=False,wrt_world=True,planning_time=.5)
        wall=time.monotonic()-start
        if result['status']!='Success':
            self.plans.append(dict(env=i,label=label,step=self.steps,status=result['status'],start_q=q.tolist(),target_p=pose.p.tolist(),target_q=pose.q.tolist(),planning_wall_s=wall,fallback_rrt=fallback,plan_cache_hit=bool(cache_hit)))
            raise SkillFailure('PLANNER_'+label+':'+result['status'])
        if not cache_hit:self.plan_cache[key]=(state.copy(),target.copy(),{k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in result.items()})
        positions=result['position'];self.plans.append(dict(env=i,stage=int(self.stage[i]),step=self.steps,label=label,start_q=q.tolist(),target_p=pose.p.tolist(),target_q=pose.q.tolist(),position=positions.tolist(),planning_wall_s=wall,fallback_rrt=fallback,plan_cache_hit=bool(cache_hit)))
        indices=list(range(0,len(positions),3))
        if indices[-1]!=len(positions)-1:indices.append(len(positions)-1)
        for index in indices:yield np.r_[positions[index],self.grip[i]],label
        for _ in range(6):
            if np.max(abs(self.q(i)[:7]-positions[-1]))<.025:break
            yield np.r_[positions[-1],self.grip[i]],label+'_TRACK'
    def hold(self,i,n,label):
        q=self.q(i)[:7]
        for _ in range(n):yield np.r_[q,self.grip[i]],label
    def grasp(self,i,actor,name,yaw_offset=0.):
        import trimesh
        box=trimesh.primitives.Box(extents=[.04,.04,.04],transform=np.eye(4));transform=np.eye(4);pose=self.pose(actor,i).astype(float);transform[:3,:3]=quat2mat(pose[3:]);transform[:3,3]=pose[:3];box.apply_transform(transform)
        g=compute_grasp_info_by_obb(box,approaching=np.array([0,0,-1]),target_closing=np.array([0,1,0]),depth=.025);cs,sn=np.cos(yaw_offset),np.sin(yaw_offset);closing=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])@g['closing'];closing*=1 if np.dot(closing,quat2mat(self.vec(self.u.agent.tcp.pose.q,i))[:,1])>=0 else -1;grasp=self.u.agent.build_grasp_pose(np.array([0,0,-1]),closing,g['center'])
        yield from self.move(i,grasp*sapien.Pose([0,0,-.06]),name+'_REACH');yield from self.move(i,grasp,name+'_DESCEND');self.grip[i]=-1;yield from self.hold(i,6,name+'_CLOSE')
        if not bool(self.u.agent.is_grasping(actor)[i].item()):raise SkillFailure(name+'_NOT_GRASPED')
        yield from self.move(i,sapien.Pose(grasp.p+[0,0,.10],grasp.q),name+'_LIFT')
        if not bool(self.u.agent.is_grasping(actor)[i].item()):raise SkillFailure(name+'_LOST_ON_LIFT')
    def geometry(self,i):
        from scipy.spatial import ConvexHull
        A,B,C=[self.pose(x,i).astype(float) for x in [self.u.cubeA,self.u.cubeB,self.u.cubeC]]
        corners=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [-.02,.02]])
        def poly(p):
            points=(corners@quat2mat(p[3:]).T+p[:3])[:,:2];return points[ConvexHull(points).vertices]
        def area(points):return abs(np.cross(points,np.roll(points,-1,axis=0)).sum())/2 if len(points)>=3 else 0.
        def overlap(subject,clip):
            output=list(subject)
            for a,b in zip(clip,np.roll(clip,-1,axis=0)):
                old=output;output=[]
                if not old:break
                prev=old[-1];dprev=np.cross(b-a,prev-a)
                for point in old:
                    d=np.cross(b-a,point-a)
                    if (d>=-1e-12)!=(dprev>=-1e-12):output.append(prev+(point-prev)*(dprev/(dprev-d)))
                    if d>=-1e-12:output.append(point)
                    prev=point;dprev=d
            return float(area(np.asarray(output)))
        midpoint=(A[:2]+B[:2])/2;projected=np.array([[-.02,-.02],[.02,-.02],[.02,.02],[-.02,.02]])+midpoint;actual=poly(C)
        return dict(predicted_centered_min_overlap_m2=min(overlap(projected,poly(A)),overlap(projected,poly(B))),actual_min_overlap_m2=min(overlap(actual,poly(A)),overlap(actual,poly(B))))
    def verify(self,i,actor,goal,stage):
        u=self.u;A,B,C=[self.pose(x,i) for x in [u.cubeA,u.cubeB,u.cubeC]];opening=float(self.q(i)[-2:].sum());not_grasped=not bool(u.agent.is_grasping(actor)[i].item());static=all(bool(x.is_static(lin_thresh=.01,ang_thresh=.5)[i].item()) for x in [u.cubeA,u.cubeB,u.cubeC]);robot=bool(np.max(abs(self.vec(u.agent.robot.get_qvel(),i)[:7]))<.15);p=self.xyz(actor,i);target_ok=np.linalg.norm(p[:2]-goal[:2])<.015;on_table=abs(p[2]-.02)<.008;height_ok=abs(p[2]-goal[2])<.008
        table=u.scene.actors['table-workspace'];ground=float(np.linalg.norm(self.vec(u.scene.get_pairwise_contact_forces(actor,table),i)));ca=float(np.linalg.norm(self.vec(u.scene.get_pairwise_contact_forces(u.cubeC,u.cubeA),i)));cb=float(np.linalg.norm(self.vec(u.scene.get_pairwise_contact_forces(u.cubeC,u.cubeB),i)))
        valid=opening>.065 and not_grasped and static and robot and target_ok and ((on_table and ground>1e-4) if stage==0 else (height_ok and max(ca,cb)>1e-4))
        return dict(valid=bool(valid),stage=stage,step=self.steps,opening_m=opening,not_grasped=not_grasped,objects_static=static,robot_static=robot,target_xy_error_m=float(np.linalg.norm(p[:2]-goal[:2])),on_table=bool(on_table),height_error_m=float(p[2]-goal[2]),actor_table_force_N=ground,C_A_force_N=ca,C_B_force_N=cb,AB_distance_m=float(np.linalg.norm(A[:2]-B[:2])),native_AB_margin_m=float(np.sqrt(.04**2+.04**2)+.005-np.linalg.norm(A[:2]-B[:2])),A=A.tolist(),B=B.tolist(),C=C.tolist(),**self.geometry(i))
    def skill(self,i,stage,choice):
        u=self.u;start=self.steps;actor=u.cubeA if stage==0 else u.cubeC;name='A' if stage==0 else 'C';request=float(ACTIONS[choice]);yield from self.grasp(i,actor,name,yaw_offset=(request*.35 if stage==1 else 0.))
        A,B=[self.pose(x,i) for x in [u.cubeA,u.cubeB]];target_q=u.agent.build_grasp_pose(np.array([0,0,-1]),np.array([0,1,0]),np.zeros(3)).q
        if stage==0:
            initial_dir=self.start[i*3][:2]-self.start[i*3+1][:2];theta=np.arctan2(initial_dir[1],initial_dir[0])+request*np.pi/2;direction=np.array([np.cos(theta),np.sin(theta)]);gap=.02*(1+abs(quat2mat(B[3:].astype(float))[:2,:].T@direction).sum())+.003;goal=B[:3].copy()+np.r_[gap*direction,0];goal[2]=.022;target_q=u.agent.build_grasp_pose(np.array([0,0,-1]),np.r_[-direction[1],direction[0],0],np.zeros(3)).q
        else:
            axis=A[:2]-B[:2];axis/=max(np.linalg.norm(axis),1e-6);goal=(A[:3]+B[:3])/2;goal[2]=max(A[2],B[2])+.043
        closing=quat2mat(target_q)[:,1];closing*=1 if np.dot(closing,quat2mat(self.vec(u.agent.tcp.pose.q,i))[:,1])>=0 else -1;target_q=u.agent.build_grasp_pose(np.array([0,0,-1]),closing,np.zeros(3)).q
        local=quat2mat(self.vec(u.agent.tcp.pose.q,i)).T@(self.xyz(actor,i)-self.vec(u.agent.tcp.pose.p,i));offset=-quat2mat(target_q)@local
        yield from self.move(i,sapien.Pose(goal+offset+[0,0,.09 if stage==0 else .07],target_q),name+'_PREPLACE');place=sapien.Pose(goal+offset,target_q);yield from self.move(i,place,name+'_PLACE');self.grip[i]=1;yield from self.hold(i,6,name+'_OPEN');yield from self.move(i,sapien.Pose(place.p+[0,0,.06],place.q),name+'_RETREAT')
        consecutive=0;r=None
        for _ in range(12):
            yield from self.hold(i,1,name+'_SETTLE');r=self.verify(i,actor,goal,stage);consecutive=consecutive+1 if r['valid'] else 0
            if consecutive>=3:break
        r.update(valid=consecutive>=3,consecutive_valid_steps=consecutive,requested=request,clipped=request,applied_rad=request*(np.pi/2 if stage==0 else .35),goal=goal.tolist(),physical_steps=self.steps-start);self.boundaries[i].append(r)
        if not r['valid']:raise SkillFailure(name+'_BOUNDARY_NOT_VERIFIED')
        self.option_records[i].append(r)
    def run(self,seeds,policy,video_indices=()):
        started=time.monotonic();self.reset(seeds);generators=[];done=np.zeros(self.n,bool);self.logical_finish=np.full(self.n,250);labels=['RESET']*self.n
        def decision(i):
            obs=self.observe(i);choice,meta=policy(obs,i,int(self.stage[i]));assert choice in [0,1,2];self.decisions[i].append(dict(observation=obs.tolist(),action=int(choice),stage=int(self.stage[i]),step=self.steps,**meta));return self.skill(i,int(self.stage[i]),choice)
        for i in range(self.n):generators.append(decision(i))
        for step in range(250):
            commands=[]
            for i in range(self.n):
                if not done[i]:
                    while True:
                        try:command,label=next(generators[i]);labels[i]=label;break
                        except StopIteration:
                            if self.stage[i]==0:self.stage[i]=1;generators[i]=decision(i);continue
                            done[i]=True;self.stage[i]=2;self.logical_finish[i]=self.steps;break
                        except SkillFailure as e:self.errors[i]=str(e);done[i]=True;self.stage[i]=3;self.logical_finish[i]=self.steps;break
                if done[i]:command=np.r_[self.q(i)[:7],self.grip[i]];labels[i]='TERMINAL_HOLD'
                commands.append(command)
            action=np.array(commands);assert action.shape==(self.n,8) and np.isfinite(action).all();_,_,_,_,info=self.env.step(action);self.steps+=1
            if self.trace:self.rows.append(serial(dict(step=self.steps,stages=self.stage.tolist(),segments=labels.copy(),action=action,q=self.u.agent.robot.get_qpos(),qvel=self.u.agent.robot.get_qvel(),tcp=self.u.agent.tcp.pose.raw_pose,state=self.u.get_state_dict(),info=info)))
            if self.writer:
                frames=self.u.render_rgb_array().detach().cpu().numpy()
                for i,writer in self.writer.items():writer.append_data(frames[i])
        # No251st action: unfinished skills are native-budget failures.
        for i in range(self.n):
            if not done[i]:self.errors[i]='NATIVE_HORIZON'
        native=serial(self.u.evaluate()['success']);self.result=dict(seeds=seeds,initial_sha256=self.initial_sha,success=native,boundaries=self.boundaries,options=self.option_records,decisions=self.decisions,errors=self.errors,logical_finish_steps=self.logical_finish.tolist(),physical_interactions=self.n*250,native_horizon=250,wall_s=time.monotonic()-started,planning_wall_s=sum(p['planning_wall_s'] for p in self.plans),planner_calls=len(self.plans));return self.result
    def close(self):self.env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=411000);p.add_argument('--cases',type=int,default=16);p.add_argument('--grid',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--nominal-only',action='store_true');a=p.parse_args();assert a.cases%16==0;a.output.mkdir(parents=True,exist_ok=False);ex=VectorHandoff(trace=True);rows=[]
    pairs=[(i,j) for i in range(3) for j in range(3)] if a.grid else [(1,1),(0,1),(2,1),(1,0),(1,2)]
    if a.nominal_only:pairs=[(1,1)]
    for batch in range(a.cases//16):
        digest=None
        for first,second in pairs:
            label=f'A{first}_C{second}'
            if a.video and batch==0 and (first,second) in [(1,1),(0,1),(1,0)]:
                import imageio.v2 as imageio
                ex.writer={i:imageio.get_writer(a.output/f'{label}_case{a.seed+i}.mp4',fps=20,codec='libx264') for i in [0,3]}
            try:r=ex.run(list(range(a.seed+batch*16,a.seed+(batch+1)*16)),lambda o,i,s:([first,second][s],{}))
            except Exception:
                import traceback
                (a.output/'FAILED.json').write_text(json.dumps(dict(physical_interactions=ex.steps*ex.n,completed_steps=ex.steps,error=traceback.format_exc(),initial=ex.initial,transitions=ex.rows,plans=ex.plans),separators=(',',':')))
                raise
            if ex.writer:
                for w in ex.writer.values():w.close()
                ex.writer=None
            if digest is None:digest=r['initial_sha256']
            assert digest==r['initial_sha256'];r['policy']=label;file=a.output/f'{batch:03d}_{label}.json';file.write_text(json.dumps(dict(summary=r,initial=ex.initial,transitions=ex.rows,plans=ex.plans),separators=(',',':')));r['trajectory_sha256']=sha(file);rows.append(r);print(json.dumps(dict(batch=batch,policy=label,success=sum(r['success']),stage1_verified=sum(len(x)>0 and x[0]['valid'] for x in r['boundaries']),steps=r['physical_interactions'],wall_s=r['wall_s'],errors=r['errors'])),flush=True);(a.output/'RESULTS.json').write_text(json.dumps(dict(rows=rows,physical_interactions=sum(x['physical_interactions'] for x in rows),executor_source_sha256=sha(__file__),scope='NEW_DESIGN_PHYSICS_NOT_TEST',task='StackPyramid-v1'),indent=2)+'\n')
    ex.close()
if __name__=='__main__':main()
