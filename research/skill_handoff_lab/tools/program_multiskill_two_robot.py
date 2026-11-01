"""Native100 TwoRobotPickCube physical opportunity scout, not an RL method.

Two real decisions: left push target, then measured-state right grasp direction.
Uses pinned native Panda planners through agent views; no task/state modifications.
The initial controller is a development baseline pending physical validation.
"""
import os
import argparse, gzip, hashlib, json, random, time, traceback
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import sapien
import torch
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from transforms3d.quaternions import quat2mat
from handoff_physics import serial

class Horizon(Exception): pass
class Failure(Exception): pass

def pose(actor): return actor.pose.sp

def digest(value): return hashlib.sha256(json.dumps(serial(value),sort_keys=True).encode()).hexdigest()

class Scout:
    def __init__(self,seed,stride=1):
        self.env=gym.make('TwoRobotPickCube-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1)
        self.env.reset(seed=seed);self.u=self.env.unwrapped;self.seed=seed;self.stride=stride
        self.keys=list(self.u.agent.agents_dict);self.agents=self.u.agent.agents
        self.initial=serial(self.u.get_state_dict());self.initial_sha=digest(self.initial)
        self.rows=[];self.plans=[];self.decisions=[];self.skill_results=[];self.phase='RESET'
        self.targets={k:np.r_[serial(a.robot.get_qpos()[0,:7]),1.] for k,a in zip(self.keys,self.agents)}
        self.solvers=[]
        for a in self.agents:
            # Only the planner's read-only agent reference is changed. All physical
            # commands still enter the one native MultiAgent environment as a dict.
            view=SimpleNamespace(agent=a,control_mode='pd_joint_pos',control_timestep=self.u.control_timestep,scene=self.u.scene)
            proxy=SimpleNamespace(unwrapped=view)
            self.solvers.append(PandaArmMotionPlanningSolver(proxy,vis=False,base_pose=a.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=.9,joint_acc_limits=.9))
        self.initial_measure=self.measure(contact_valid=False)
    def measure(self,contact_valid=True):
        u=self.u;cube=pose(u.cube);out=dict(cube=np.r_[cube.p,cube.q].tolist(),goal=serial(u.goal_site.pose.raw_pose[0]),cube_linear_velocity=serial(u.cube.linear_velocity[0]),cube_angular_velocity=serial(u.cube.angular_velocity[0]),native_success=bool(u.evaluate()['success'][0]),contact_valid=contact_valid,remaining_steps=100-int(u.elapsed_steps[0]))
        out['cube_goal_distance_m']=float(np.linalg.norm(cube.p-pose(u.goal_site).p))
        for i,a in enumerate(self.agents):
            tcp=pose(a.tcp)
            out[str(i)]=dict(q=serial(a.robot.get_qpos()[0]),qvel=serial(a.robot.get_qvel()[0]),tcp=np.r_[tcp.p,tcp.q].tolist(),tcp_T_cube=np.r_[(tcp.inv()*cube).p,(tcp.inv()*cube).q].tolist(),grasped=bool(a.is_grasping(u.cube)[0]),static=bool(a.is_static(.2)[0]),robot_cube_force_N={link.name:float(torch.linalg.vector_norm(u.scene.get_pairwise_contact_forces(link,u.cube)[0])) for link in a.robot.get_links()} if contact_valid else {})
        return out
    def step(self):
        if len(self.rows)>=100: raise Horizon('NATIVE100')
        ret=self.env.step(self.targets)
        self.rows.append(dict(step=len(self.rows)+1,phase=self.phase,action=serial(self.targets),measurement=self.measure(),state=serial(self.u.get_state_dict()),info=serial(ret[-1])))
        return ret
    def hold(self,n,label):
        self.phase=label
        for _ in range(n): self.step()
    def plan(self,i,target,label):
        start=time.monotonic();s=self.solvers[i]
        r=s.move_to_pose_with_screw(target,dry_run=True)
        if isinstance(r,int):
            r=s.planner.plan_qpos_to_pose(np.r_[target.p,target.q],np.asarray(serial(self.agents[i].robot.get_qpos()[0])),time_step=self.u.control_timestep,planning_time=.5,wrt_world=True,use_point_cloud=False)
        good=isinstance(r,dict) and r.get('status')=='Success'
        row=dict(phase=label,arm=i,step=len(self.rows),target=np.r_[target.p,target.q].tolist(),planning_wall_s=time.monotonic()-start,status='Success' if good else 'FAIL',position=serial(r['position']) if good else None)
        self.plans.append(row)
        if not good:raise Failure('PLAN_'+label)
        q=np.asarray(r['position']);q=q if len(q) else np.asarray(serial(self.agents[i].robot.get_qpos()[0,:7]))[None]
        idx=list(range(0,len(q),self.stride))
        if idx[-1]!=len(q)-1:idx.append(len(q)-1)
        row['executed_indices']=idx;return q[idx]
    def follow(self,paths,label):
        self.phase=label
        for t in range(max(map(len,paths.values()))):
            for i,q in paths.items():self.targets[self.keys[i]][:7]=q[min(t,len(q)-1)]
            self.step()
    def move(self,i,target,label):self.follow({i:self.plan(i,target,label)},label)
    def run(self,push_y=.05,grasp_axis='geometry',controller='original',align_grasp_sign=False,axis_objective='pregrasp'):
        left,right=self.agents;cube=pose(self.u.cube)
        self.decisions.append(dict(skill='LEFT_PUSH',boundary_step=0,parameters=dict(push_target_y_m=push_y),observation=self.initial_measure,next_skill_requirement='cube at or beyond center, low residual velocity and reachable by right grasp'))
        # Closed fingertips push the negative-y face; approach from above avoids
        # side-swiping before the commanded push. Contact geometry is audited by
        # actual link/cube forces, not inferred from the commanded TCP location.
        pushquat=left.build_grasp_pose(np.array([0,0,-1.]),np.array([1.,0,0]),cube.p).q
        self.targets[self.keys[0]][-1]=-1
        start=cube.p+np.array([0,-.045,.08]);self.move(0,sapien.Pose(start,pushquat),'LEFT_REACH')
        start[2]=.027;self.move(0,sapien.Pose(start,pushquat),'LEFT_CONTACT_APPROACH')
        # Reobserve lateral drift between short segments; no future-success data.
        for _ in range(10):
            cp=pose(self.u.cube).p
            if cp[1]>=push_y-.005:break
            tcp=pose(left.tcp).p
            endpoint=np.array([cp[0],min(push_y-.025,tcp[1]+(.30 if controller=='continuous-clear' else .05)),.027])
            if endpoint[1]<=tcp[1]+.001:break
            self.move(0,sapien.Pose(endpoint,pushquat),'LEFT_PUSH')
        retreat=pose(left.tcp).p+np.array([0,-.025,.08])
        if controller=='continuous-clear':retreat[1]=-.20;retreat[2]=.15
        self.move(0,sapien.Pose(retreat,pushquat),'LEFT_RELEASE')
        self.hold(2,'PUSH_SETTLE')
        m=self.measure();ok=bool(m['cube'][1]>=0 and abs(m['cube'][2]-.02)<.02)
        self.skill_results.append(dict(skill='LEFT_PUSH',boundary_step=len(self.rows),physical_success=ok,measurement=m,postcondition='cube y>=0 and on table; velocity retained, not silently declared settled'))
        if not ok:raise Failure('PUSH_NOT_IN_RIGHT_WORKSPACE')
        cube=pose(self.u.cube);R=quat2mat(cube.q)
        choices=[('cube_x',R[:,0]),('cube_y',R[:,1])] if grasp_axis=='geometry' else [(grasp_axis,R[:,0 if grasp_axis=='cube_x' else 1])]
        candidates=[]
        for label,axis in choices:
            axis=axis.copy();axis[2]=0;axis/=np.linalg.norm(axis)
            if align_grasp_sign and np.dot(axis,quat2mat(pose(right.tcp).q)[:,1])<0:axis=-axis
            target=right.build_grasp_pose(np.array([0,0,-1.]),axis,cube.p)
            try:
                path=self.plan(1,target*sapien.Pose([0,0,-.065]),'RIGHT_GRASP_SEARCH_'+label)
                cost=len(path)
                if axis_objective=='full-suffix':
                    # Nominal rigid-grasp prediction, NOT a future physical outcome.
                    # Evaluate the complete successor under shared path decimation.
                    started=time.monotonic();qq=np.r_[path[-1],serial(right.robot.get_qpos()[0,-2:])]
                    attachment=target.inv()*cube
                    lift=sapien.Pose(target.p+np.array([0,0,.06]),target.q)
                    goal=sapien.Pose(pose(self.u.goal_site).p,cube.q)*attachment.inv()
                    predicted=[];valid=True;cost+=6+6  # close + declared static reserve
                    for segment,goalpose in [('DESCEND',target),('LIFT',lift),('TO_GOAL',goal)]:
                        rr=self.solvers[1].planner.plan_screw(np.r_[goalpose.p,goalpose.q],qq,time_step=self.u.control_timestep,use_point_cloud=False)
                        ok=rr.get('status')=='Success';n=len(rr['position']) if ok else 0
                        if not ok:valid=False;break
                        idx=list(range(0,max(n,1),self.stride))
                        if n and idx[-1]!=n-1:idx.append(n-1)
                        cost+=len(idx)
                        predicted.append(dict(segment=segment,target=np.r_[goalpose.p,goalpose.q].tolist(),position=serial(rr['position']),executed_indices=idx))
                        if n:qq=np.r_[rr['position'][-1],qq[-2:]]
                    self.plans.append(dict(phase='FULL_SUFFIX_SEARCH_'+label,arm=1,step=len(self.rows),planning_wall_s=time.monotonic()-started,status='Success' if valid else 'FAIL',position=None,predicted_segments=predicted,predicted_total_steps=cost,remaining_steps=100-len(self.rows),settling_reserve_steps=6,model='nominal rigid-grasp geometry, unverified actual contact/settling'))
                    if not valid:continue
                candidates.append((cost,label,target,path))
            except Failure:pass
        if not candidates:raise Failure('RIGHT_SUFFIX_NO_FEASIBLE_PLAN' if axis_objective=='full-suffix' else 'RIGHT_GRASP_NO_IK')
        _,chosen,g,path=min(candidates,key=lambda x:x[0])
        self.decisions.append(dict(skill='RIGHT_PICK_TO_GOAL',boundary_step=len(self.rows),parameters=dict(grasp_axis=chosen),observation=self.measure(),previous_skill_result=self.skill_results[-1],selection_objective=axis_objective,candidate_costs=[dict(axis=x[1],approach_steps=len(x[3]),objective_steps=x[0],budget_feasible=x[0]<=100-len(self.rows)) for x in candidates],next_skill_requirement='native goal within25mm and right arm static by100'))
        self.follow({1:path},'RIGHT_REACH');self.move(1,g,'RIGHT_DESCEND')
        self.targets[self.keys[1]][-1]=-1;self.hold(6,'RIGHT_CLOSE')
        if not bool(right.is_grasping(self.u.cube)[0]):raise Failure('RIGHT_NOT_GRASPED')
        self.move(1,sapien.Pose(pose(right.tcp).p+np.array([0,0,.06]),pose(right.tcp).q),'RIGHT_LIFT')
        attachment=pose(right.tcp).inv()*pose(self.u.cube)
        desired=sapien.Pose(pose(self.u.goal_site).p,pose(self.u.cube).q)
        self.move(1,desired*attachment.inv(),'RIGHT_TO_GOAL')
        # Terminal correction uses observed object displacement and attachment.
        for _ in range(2):
            if self.measure()['cube_goal_distance_m']<=.015:break
            attachment=pose(right.tcp).inv()*pose(self.u.cube)
            self.move(1,desired*attachment.inv(),'RIGHT_GOAL_FEEDBACK')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=742000);ap.add_argument('--push-y',type=float,default=.05);ap.add_argument('--grasp-axis',choices=['geometry','cube_x','cube_y'],default='geometry');ap.add_argument('--axis-objective',choices=['pregrasp','full-suffix'],default='pregrasp');ap.add_argument('--align-grasp-sign',action='store_true');ap.add_argument('--controller',choices=['original','continuous-clear'],default='original');ap.add_argument('--stride',type=int,choices=[1,2,3],default=1);a=ap.parse_args()
    if a.axis_objective=='full-suffix':a.align_grasp_sign=True
    if not 0<=a.push_y<=.12:ap.error('push target must be within native central/right tabletop [0,.12]m')
    a.out.mkdir(parents=True,exist_ok=False);random.seed(31101);np.random.seed(31101);torch.manual_seed(31101)
    source=Path(__file__).read_bytes();(a.out/Path(__file__).name).write_bytes(source)
    manifest=dict(resource_context=os.environ.get('ARES_RESOURCE_CONTEXT','UNSPECIFIED'),task='TwoRobotPickCube-v1',native_horizon=100,initial_distribution='NATIVE_UNCHANGED',source_kind='ACTUAL_PHYSX_EXECUTION',optimizer_updates=0,reset_count=1,seed=a.seed,parameters=dict(push_y=a.push_y,grasp_axis=a.grasp_axis,path_stride=a.stride,controller=a.controller,align_grasp_sign=a.align_grasp_sign,axis_objective=a.axis_objective),source_sha256=hashlib.sha256(source).hexdigest(),baseline_status='DEVELOPMENT_SCOUT_NOT_YET_VALIDATED_STRONG',collision_scope='native robot self collision planner; other robot and cube NOT registered as planning obstacles; actual contacts logged',timing='full native TOPP positions' if a.stride==1 else 'subsampled absolute targets, not native TOPP timing; actual tracking recorded')
    pinned=Path('/home/andyee/robot-research/ARES-Skilllab/external/benchmark_scout_20261027/ManiSkill')
    dependencies=[Path(__file__).with_name('handoff_physics.py'),Path(__file__).with_name('method_stack_physics.py'),pinned/'mani_skill/envs/tasks/tabletop/two_robot_pick_cube.py',pinned/'mani_skill/utils/scene_builder/table/scene_builder.py',pinned/'mani_skill/agents/multi_agent.py',pinned/'mani_skill/examples/motionplanning/panda/motionplanner.py',pinned/'mani_skill/examples/motionplanning/base_motionplanner/motionplanner.py']
    manifest['dependencies_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    manifest['maniskill_revision']='62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3'
    manifest['maniskill_license_sha256']=hashlib.sha256((pinned/'LICENSE').read_bytes()).hexdigest()
    for i,p in enumerate(dependencies):(a.out/f'source_{i}_{p.name}').write_bytes(p.read_bytes())
    (a.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));s=None;t=time.monotonic();error=None
    try:
        s=Scout(a.seed,a.stride)
        try:s.run(a.push_y,a.grasp_axis,a.controller,a.align_grasp_sign,a.axis_objective)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        if len(s.rows)<100:s.hold(100-len(s.rows),'TERMINAL_HOLD')
        final=s.measure();row=dict(seed=a.seed,success=final['native_success'],error=error,steps=len(s.rows),decisions=len(s.decisions),initial_sha256=s.initial_sha,wall_s=time.monotonic()-t,planning_wall_s=sum(p['planning_wall_s'] for p in s.plans),final=final)
        if len(s.decisions)==2:s.skill_results.append(dict(skill='RIGHT_PICK_TO_GOAL',boundary_step=100,physical_success=final['native_success'],measurement=final))
        data=serial(dict(summary=row,initial=s.initial,initial_measure=s.initial_measure,steps=s.rows,plans=s.plans,decisions=s.decisions,skill_results=s.skill_results))
        payload=json.dumps(data,separators=(',',':'))
        path=a.out/'TRAJECTORY.json.gz'
        with gzip.open(path,'wt') as f:f.write(payload)
        row.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());(a.out/'RESULTS.json').write_text(json.dumps(serial([row]),indent=2));print(json.dumps(serial(row)),flush=True)
    except Exception as exc:
        crash=dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),wall_s=time.monotonic()-t,steps=len(s.rows) if s is not None else None,accounting='known recorded physics only; construction failure may precede recorder, never count as completed episode')
        (a.out/'CRASH_ACCOUNTING.json').write_text(json.dumps(crash,indent=2))
        if s is not None:
            payload=json.dumps(serial(dict(initial=s.initial,initial_measure=s.initial_measure,steps=s.rows,plans=s.plans,decisions=s.decisions,skill_results=s.skill_results,crash=crash)),separators=(',',':'))
            with gzip.open(a.out/'PARTIAL_PHYSICS.json.gz','wt') as f:f.write(payload)
        raise
    finally:
        if s:s.env.close()

if __name__=='__main__':main()
