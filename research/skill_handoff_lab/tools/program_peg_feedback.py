"""Bounded feedback inside inherited Peg insertion; native100, no pose setters."""
import json,time
import numpy as np
import sapien
from scipy.spatial.transform import Rotation
from program_peg_option_runtime import Runtime
from program_peg_native import val
from program_paired_skill_value import peg_features
ACTIONS=['continue_plan','measured_target','target_y_plus3mm','target_y_minus3mm','target_z_plus3mm','target_z_minus3mm','withdraw6mm','hold_phase_bounded']
NOBS=64

def features(m,remaining,stage,queue=None,cursor=0,last=0):
    base=peg_features(m,remaining,stage);tail=np.zeros(10,dtype=np.float32)
    if queue is not None:
        tail[:7]=queue[min(cursor,len(queue)-1)]-np.array(m['q'][:7])
        tail[7:]=[min(max(len(queue)-cursor,0),100)/100,cursor/100,last/7]
    return np.r_[base,tail].astype(np.float32)

def potential(m):
    x=peg_features(m,0,1)
    return float(-np.clip(np.linalg.norm(x[45:48])/.15,0,3)-.2*x[48])

class FeedbackRuntime(Runtime):
    def __init__(self,out,chunk=4,feedback_start=3):
        self.chunk=chunk;super().__init__(out,feedback=self.execute_feedback,feedback_start=feedback_start);self.controller_id+=f'__feedback_v4_persistent_start{feedback_start}'
        (self.out/'FEEDBACK_CONTRACT.json').write_text(json.dumps(dict(id=self.controller_id,actions=ACTIONS,chunk=chunk,feedback_start=feedback_start,horizon=100,observation='shared privileged physical state plus active joint queue',zero='original refinement and insertion queues then actual final-joint hold',dense='0.2 potential difference; terminal potential zero; terminal native success',no_pose_reset=True,hold_semantics='ID7 replaces queue by current q: refinement ends after one step; INSERT holds up to4steps. Durations are measured, not assumed.',parameter_hold='IDs1..5 repeated while active continue queue; change or exhausted queue triggers measured replan'),indent=2))
    def decide(self,obs):
        if len(obs)==54:obs=np.r_[obs,np.zeros(10)].astype(np.float32)
        start=time.monotonic();a=int(self.callback(obs));stage=int(obs[44]);assert 0<=a<(3 if stage==0 else len(ACTIONS))
        self.actions.append(dict(stage=stage,step=len(self.env.records),id=a,obs=obs.tolist(),wall_s=time.monotonic()-start));return a
    def execute_feedback(self,planner,pose,*args,**kw):
        from mani_skill.utils.structs.pose import to_sapien_pose
        pose=to_sapien_pose(pose)
        e=self.env;u=e.unwrapped;phase=['REACH','GRASP','ALIGN','REFINE1','REFINE2','REFINE3','INSERT'][min(len(e.calls),6)];e.phase=phase;call=dict(phase=phase,start=len(e.records),target=np.r_[pose.p,pose.q].tolist());e.calls.append(call)
        def plan(target):
            for _ in range(2):
                result=planner.planner.plan_screw(np.r_[target.p,target.q],u.agent.robot.get_qpos().cpu().numpy()[0],time_step=u.control_timestep,use_point_cloud=planner.use_point_cloud)
                if result['status']=='Success':return np.array(result['position'])
            return None
        queue=plan(pose);nominal_failed=queue is None;cursor=0;last=0;holdq=None
        if queue is None:queue=np.array([val(u.agent.robot.get_qpos()[0,:7])])
        while len(e.records)<100:
            obs=features(e.measure(),100-len(e.records),1,queue,cursor,last);obs[49]=min(len(e.calls),7)/7
            action=self.decide(obs);record=self.actions[-1];record['plan_failed']=False;record['replanned']=False
            if nominal_failed and action==0:
                call.update(end=len(e.records),failed=True);return -1
            nominal_failed=False
            if action==7:queue=np.array([val(u.agent.robot.get_qpos()[0,:7])]);cursor=0;holdq=None
            elif action!=0 and not(action in (1,2,3,4,5) and action==last and cursor<len(queue)):
                shift=np.zeros(3)
                if action in (2,3):shift[1]=.003*(1 if action==2 else -1)
                if action in (4,5):shift[2]=.003*(1 if action==4 else -1)
                if phase=='INSERT':target=u.goal_pose.sp*sapien.Pose(shift)*u.peg.pose.sp.inv()*u.agent.tcp.pose.sp*sapien.Pose([.05,0,0])
                else:
                    shift[0]=-float(u.peg_half_sizes[0,0])-.01
                    target=u.goal_pose.sp*sapien.Pose(shift)*u.peg.pose.sp.inv()*u.agent.tcp.pose.sp
                if action==6:
                    world=Rotation.from_quat(np.r_[u.goal_pose.sp.q[1:],u.goal_pose.sp.q[0]]).apply([-.006,0,0]);target=sapien.Pose(u.agent.tcp.pose.sp.p+world,u.agent.tcp.pose.sp.q)
                record['replanned']=True;new=plan(target);record['target']=np.r_[target.p,target.q].tolist()
                if new is not None:queue=new;cursor=0;holdq=None
                else:record['plan_failed']=True
            record['queue_length']=len(queue);record['queue_cursor']=cursor
            for _ in range(min(self.chunk,100-len(e.records),len(queue)-cursor if phase!='INSERT' else 100)):
                if cursor<len(queue):q=queue[cursor];cursor+=1
                else:
                    if holdq is None:holdq=np.array(val(u.agent.robot.get_qpos()[0,:7]))
                    q=holdq
                e.phase=phase+'_FEEDBACK';out=e.step(np.r_[q,-1]);planner.elapsed_steps+=1
            last=action
            if phase!='INSERT' and cursor>=len(queue):break
        call['end']=len(e.records);call['failed']=False
        return out
    def feedback_transitions(self,d,source):
        out=[];actions=d['summary']['decisions'];y=float(d['summary']['success'])
        for i,a in enumerate(actions):
            end=actions[i+1]['step'] if i+1<len(actions)else 100
            nxt=actions[i+1]['obs'] if i+1<len(actions)else features(d['steps'][-1],0,1).tolist()
            m=d['initial']if a['step']==0 else d['steps'][a['step']-1];n=d['steps'][end-1];done=i+1==len(actions)
            shaped=.2*((0 if done else potential(n))-potential(m))
            out.append(dict(s=a['obs'],next=nxt,a=a['id'],r=(y if done else 0)+shaped,done=done,duration=end-a['step'],stage=a['stage'],outcome=y,scene=d['summary']['seed'],source=source,controller=self.controller_id,step=a['step']))
        assert sum(t['duration']for t in out)==100
        return out

def rule(obs):
    if obs[44]<.5:return 1
    if obs[-3]>0:return 0
    if obs[43]<.025:return 7
    return 1
