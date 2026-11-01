"""Shared execution diagnostic: shortest valid native-timed transfer plan.
No trajectory decimation, no parameter RL; screw contact/pull unchanged.
"""
import time
import numpy as np
from program_stap_tool import TracePlanner

class TransferPlanner(TracePlanner):
    def move_to_pose_with_screw(self,pose,**kwargs):
        if self.env.phase!='TRANSFER':return super().move_to_pose_with_screw(pose,**kwargs)
        start=time.monotonic();q=self.robot.get_qpos()[0].cpu().numpy();target=np.r_[pose.p,pose.q]
        screw=self.planner.plan_screw(target,q,time_step=self.base_env.control_timestep,use_point_cloud=False)
        joint=self.planner.plan_qpos_to_pose(target,q,time_step=self.base_env.control_timestep,planning_time=.5,use_point_cloud=False,wrt_world=True)
        valid=[(n,r) for n,r in [('screw',screw),('joint',joint)] if r['status']=='Success' and len(r['position'])]
        costs={n:len(r['position']) if r['status']=='Success' else None for n,r in [('screw',screw),('joint',joint)]}
        if not valid:return -1
        name,r=min(valid,key=lambda z:len(z[1]['position']))
        self.env.plans.append(dict(phase='TRANSFER',step=len(self.env.records),target=target.tolist(),planning_wall_s=time.monotonic()-start,status='Success',selected_planner=name,candidate_steps=costs,position=r['position'].tolist()))
        return self.follow_path(r,refine_steps=0)

if __name__=='__main__':
    import program_stap_tool as task
    task.TracePlanner=TransferPlanner
    task.main()
