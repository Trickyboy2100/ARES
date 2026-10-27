"""Third physical Peg probe: compare legal reactive correction after tracked approach.

Only native controller commands; no rigid-body set_pose, reward/physics edits,
or extra horizon. State observations are explicitly privileged baselines.
"""
import argparse,hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
from method_peg_physics import Recorded,BudgetEnd,solver
from method_stack_physics import serial

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=8);p.add_argument('--seed',type=int,default=110000);p.add_argument('--video',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply,quaternion_invert,quaternion_to_axis_angle,quaternion_apply
    env=Recorded(gym.make('PegInsertionSide-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));rows=[]
    for case in range(a.cases):
        digest=None
        for mode in ['tracked_nominal','reactive','align_then_advance']:
            # Re-enter the original control interface on every independent reset.
            env.unwrapped.agent.set_control_mode('pd_joint_pos')
            solve,original,modified=solver(.95,1,True,[0,0,0],stride=3);failure=None
            if a.video and case<2:env.writer=imageio.get_writer(a.output/f'case{case:03d}_{mode}.mp4',fps=20,codec='libx264')
            try:
                solve(env,seed=a.seed+case,debug=False,vis=False)
                if mode!='tracked_nominal':
                    u=env.unwrapped;env.stage=100
                    while len(env.rows)<100:
                        if bool(u.evaluate()['success'][0]):break
                        target=u.goal_pose*u.peg.pose.inv()*u.agent.tcp.pose
                        dp=target.p-u.agent.tcp.pose.p
                        rotation=quaternion_multiply(target.q,quaternion_invert(u.agent.tcp.pose.q));dr=quaternion_to_axis_angle(rotation)
                        gap=u.evaluate()['peg_head_pos_at_hole'];orientation=quaternion_to_axis_angle(quaternion_multiply(u.goal_pose.q,quaternion_invert(u.peg.pose.q)))
                        aligned=(torch.linalg.vector_norm(gap[:,1:],dim=1)<.0015)&(torch.linalg.vector_norm(orientation,dim=1)<.02)
                        if mode=='align_then_advance' and not bool(aligned[0]):
                            axis=quaternion_apply(u.goal_pose.q,torch.tensor([[1.,0,0]],device=u.device));dp=dp-axis*(dp*axis).sum(-1,keepdim=True)-axis*.002
                        command=torch.cat(((dp/.1).clamp(-.1,.1),(dr/.1).clamp(-.3,.3),torch.full((1,1),-1.,device=u.device)),-1)
                        env.step({'control_mode':'pd_ee_delta_pose','action':command})
            except BudgetEnd:failure='NATIVE_HORIZON'
            except Exception:
                import traceback;failure=traceback.format_exc()
            if env.writer:env.writer.close();env.writer=None
            initial=hashlib.sha256(json.dumps(env.initial,sort_keys=True).encode()).hexdigest()
            if digest is None:digest=initial
            assert initial==digest,'Cold-reset initial mismatch'
            final=serial(env.unwrapped.evaluate());row=dict(seed=a.seed+case,policy=mode,steps=len(env.rows),success=final['success'][0],success_ever=any(r['info']['success'][0] for r in env.rows),terminal=final,failure=failure,initial_sha256=digest,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='NATIVE_100_STEP_NO_TASK_EDITS_STATE_BASELINE',upstream_solver_sha256=original)
            file=a.output/f'case{case:03d}_{mode}.json';file.write_text(json.dumps(dict(summary=row,initial=env.initial,transitions=env.rows),separators=(',',':')));row['trajectory_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();rows.append(row);print(json.dumps(row),flush=True)
            (a.output/'RESULTS.json').write_text(json.dumps(dict(source_kind='NEW_PHYSICAL_SIMULATION',task='PegInsertionSide-v1',native_horizon=100,rows=rows,physical_interactions=sum(x['steps'] for x in rows),sealed_test_used=False),indent=2)+'\n')
    env.close()

if __name__=='__main__':main()
