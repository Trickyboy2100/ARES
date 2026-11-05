"""Conservative standard IK waypoint splice, fixed original time grid.
Rejects infeasible corrections rather than stretching native budget. Planning
model checks are not a real-hardware safety certificate.
"""
import inspect,textwrap,time
import numpy as np
from scipy.spatial.transform import Rotation
import program_peg_feedback as base
from program_peg_native import val
class ProgressRuntime(base.FeedbackRuntime):
 def __init__(self,*args,**kw):
  kw['correction_frame']='remaining_target';super().__init__(*args,**kw);self.controller_id+='__progress_splice_v1'
 def splice(self,planner,queue,cursor,target,active,record):
  tic=time.monotonic();p=planner.planner;u=self.env.unwrapped;dt=u.control_timestep
  old=np.asarray(queue[cursor:]).copy();current=np.asarray(val(u.agent.robot.get_qpos()[0]));v=np.asarray(val(u.agent.robot.get_qvel()[0]))[:7];reason=None;new=[];count=0
  detail=dict(validation_scope='inherited planning-world collision model; sampled joint dynamics, not a certified contact-safety guarantee',old_cursor=cursor,old_remaining=len(old),dt=dt,requested_shift=(target.p-active.p).tolist(),native_remaining=100-len(self.env.records))
  if len(old)<2:reason='insufficient_remaining_knots'
  if not bool(u.agent.is_grasping(u.peg)[0]):reason='grasp_not_confirmed'
  if np.linalg.norm(target.p-active.p)>.003001:reason='outside_local_translation_domain'
  idx=p.link_name_2_idx[p.move_group];shift=np.asarray(target.p-active.p);seed=current.copy()
  try:
   if not reason:
    # Apply a smooth displacement on the original future FK poses.
    # Measured q/qvel anchor feasibility checks, not a tracking-error bridge.
    for i,q in enumerate(old):
     full=current.copy();full[:7]=q;p.pinocchio_model.compute_forward_kinematics(full);pose=np.asarray(p.pinocchio_model.get_link_pose(idx)).copy()
     f=min((i+1)/max(4,len(old)),1.);blend=f*f*(3-2*f)
     # Planner FK is base-frame; rotate the world displacement accordingly.
     delta=p.transform_goal_to_wrt_base(np.r_[target.p,target.q])[:3]-p.transform_goal_to_wrt_base(np.r_[active.p,active.q])[:3]
     pose[:3]+=blend*delta
     status,qs=p.IK(pose,seed,n_init_qpos=1,threshold=1e-4);count+=1
     if status!='Success':reason='IK_or_collision';break
     candidate=np.asarray(qs[0])[:7]
     # Do not force a large lagging measured start onto a new schedule.
     new.append(candidate);seed=current.copy();seed[:7]=candidate
    if not reason:
     new=np.asarray(new);vel=np.diff(np.vstack([current[:7],new]),axis=0)/dt;acc=np.diff(np.vstack([v,vel]),axis=0)/dt
     detail.update(max_velocity=np.abs(vel).max(0).tolist(),max_acceleration=np.abs(acc).max(0).tolist(),velocity_limits=np.asarray(p.joint_vel_limits).tolist(),acceleration_limits=np.asarray(p.joint_acc_limits).tolist())
     if np.any(np.abs(vel)>np.asarray(p.joint_vel_limits)+1e-6):reason='velocity_limit'
     elif np.any(np.abs(acc)>np.asarray(p.joint_acc_limits)+1e-6):reason='acceleration_limit'
     else:
      previous=current[:7]
      for q in new:
       for blend in [.25,.5,.75,1.]:
        probe=previous+(q-previous)*blend
        if p.check_for_self_collision(qpos=probe) or p.check_for_env_collision(qpos=probe,with_point_cloud=planner.use_point_cloud):reason='collision';break
       if reason:break
       previous=q
  except Exception as ex:reason=type(ex).__name__+':'+str(ex)
  detail.update(accepted=reason is None,rejection=reason,ik_queries=count,wall_s=time.monotonic()-tic)
  record['splice']=detail;self.queries.append(dict(kind='progress_splice',**detail))
  return np.asarray(new)if reason is None else None
# Keep unchanged surrounding execution and logging; save this exact expanded
# method in each physical run for source-level reproducibility.
source=textwrap.dedent(inspect.getsource(base.FeedbackRuntime.execute_feedback))
needle="record['replanned']=True;new=plan(target);record['target']"
assert needle in source
source=source.replace(needle,"record['replanned']=True;new=self.splice(planner,queue,cursor,target,active_target,record) if action in (2,3,4,5) else plan(target);record['target']")
ns={};exec(compile(source,__file__,'exec'),dict(base.__dict__),ns);ProgressRuntime.execute_feedback=ns['execute_feedback']
