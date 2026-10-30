"""EXP-001 bounded geometric servo, same native grasp prefix and task."""
from charger_probe import *
from transforms3d.quaternions import qinverse,quat2axangle,axangle2quat
def strong(env,seed,angle,standoff,Planner,variant):
    policy=None
    env.reset(seed=seed);u=env.unwrapped;p=Planner(env,vis=False,base_pose=u.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=1.,joint_acc_limits=1.);p.manual=True;p.stride=1;env.active_planner=p
    o=pack_observation(env.measure(),0)
    if policy:idx,meta=policy(o,0);angle=ANGLES[idx]
    else:meta={}
    env.decisions.append(dict(stage=0,observation=o.tolist(),value=angle,step=0,**meta))
    def move(target,label):
        env.phase=label;p.stride=2 if label in ['REACH','LIFT','ALIGN'] else 1
        if p.move_to_pose_with_screw(target,refine_steps=4)==-1:raise Failure('PLAN_'+label)
    try:
        box=trimesh.primitives.Box(extents=np.array(u._base_size)*2,transform=u.charger_base_pose.sp.to_transformation_matrix());g=compute_grasp_info_by_obb(box,approaching=np.array([0,0,-1]),target_closing=sp(u.agent.tcp).to_transformation_matrix()[:3,1],depth=.025);pose=u.agent.build_grasp_pose(np.array([0,0,-1]),g['closing'],g['center'])*sapien.Pose(q=euler2quat(0,np.deg2rad(angle),0))
        move(pose*sapien.Pose([0,0,-.05]),'REACH');move(pose,'DESCEND');p.close_gripper()
        if not bool(u.agent.is_grasping(u.charger)[0]):raise Failure('CHARGER_NOT_GRASPED')
        move(sapien.Pose(sp(u.agent.tcp).p+[0,0,.04],sp(u.agent.tcp).q),'LIFT')
        if not bool(u.agent.is_grasping(u.charger)[0]):raise Failure('CHARGER_LOST_ON_LIFT')
        o=pack_observation(env.measure(),len(env.records),angle)
        if policy:idx,meta=policy(o,1);standoff=DISTANCES[idx]
        else:meta={}
        env.decisions.append(dict(stage=1,observation=o.tolist(),value=standoff,step=len(env.records),**meta))
        attached=sp(u.agent.tcp).inv()*sp(u.charger)
        goal=u.goal_pose.sp
        move(goal*sapien.Pose([-standoff,0,0])*attached.inv(),'ALIGN')
        # Bounded measured-object servo; no learned parameters or outcome lookup.
        bias_p=np.zeros(3);hold_x=float((goal.inv()*sp(u.charger)).p[0])
        while len(env.records)<200:
            env.phase='CONTACT_SERVO';actual=sp(u.charger);tcp=sp(u.agent.tcp);local=goal.inv()*actual
            orient=2*np.arccos(np.clip(abs(local.q[0]),0,1));lateral=np.linalg.norm(local.p[1:])
            aligned=lateral<.0004 and orient<.03
            x=min(0.,float(local.p[0])+.002) if aligned else min(hold_x,-.020)
            if variant=='servo_local_hold_stop' and not aligned:x=float(local.p[0])
            if variant=='servo_direct':x=0.
            desired=goal*sapien.Pose([x,0,0]);bias_p=np.clip(bias_p+.2*(desired.p-actual.p),-.008,.008);attached=tcp.inv()*actual;target=desired*attached.inv()
            dp=target.p+bias_p-tcp.p;dp*=min(1.,.004/max(np.linalg.norm(dp),1e-9))
            dq=qmult(target.q,qinverse(tcp.q));axis,ang=quat2axangle(dq)
            if ang>np.pi:ang-=2*np.pi
            q=qmult(axangle2quat(axis,np.clip(ang,-.04,.04)),tcp.q);target=sapien.Pose(tcp.p+dp,q)
            qnow=np.asarray(serial(u.agent.robot.get_qpos()[0]));status,solutions=p.planner.IK(p.planner.transform_goal_to_wrt_base(np.r_[target.p,target.q]),qnow,n_init_qpos=1,threshold=1e-4)
            if status!='Success':raise Failure('SERVO_IK_FAILED')
            command=solutions[0][:7]
            command=qnow[:7]+np.clip(command-qnow[:7],-.05,.05)
            env.step(np.r_[command,-1.]);env.records[-1]['servo']=dict(lateral_m=float(lateral),orientation_rad=float(orient),advance=bool(aligned),desired_local_x=float(x))
    finally:p.close()
