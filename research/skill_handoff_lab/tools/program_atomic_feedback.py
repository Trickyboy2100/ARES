"""Explicit reference/action family and observable semantic-phase Options.
Standard interface foundations; no claim of novel SAC or termination theory.
"""
import numpy as np,time,json
import sapien
from scipy.spatial.transform import Rotation
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption
PHASES=['REACH','GRASP','ALIGN','REFINE1','REFINE2','REFINE3','INSERT']
ACTIONS=['continue','refresh','local_y+','local_y-','local_z+','local_z-','withdraw','hold','refresh_y+','refresh_y-','refresh_z+','refresh_z-']
NOBS=83
class AtomicRuntime(FeedbackRuntime):
 def __init__(self,*a,**kw):
  kw['correction_frame']='remaining_target';super().__init__(*a,**kw)
  self.controller_id+='__atomic83_v1'
  (self.out/'ATOMIC_CONTRACT.json').write_text(json.dumps(dict(actions=ACTIONS,nobs=NOBS,phase='real current planner semantic phase, not capped query count',features='legacy64 + phase onehot7 + active-minus-TCP position/rotation + refreshed-minus-active position/rotation',native=100,novelty='standard action/state augmentation',future_success_input=False),indent=2))
 def execution_observation(self,obs,phase,active,u):
  shift=np.zeros(3)
  if phase=='INSERT':refresh=u.goal_pose.sp*u.peg.pose.sp.inv()*u.agent.tcp.pose.sp*sapien.Pose([.05,0,0])
  else:
   shift[0]=-float(u.peg_half_sizes[0,0])-.01
   refresh=u.goal_pose.sp*sapien.Pose(shift)*u.peg.pose.sp.inv()*u.agent.tcp.pose.sp
  def rot(q):return Rotation.from_quat(np.r_[q[1:],q[0]])
  tcp=u.agent.tcp.pose.sp;rg=rot(u.goal_pose.sp.q)
  tail=np.r_[np.eye(7)[PHASES.index(phase)],rg.inv().apply(active.p-tcp.p)/.1,(rot(tcp.q).inv()*rot(active.q)).as_rotvec()/np.pi,rg.inv().apply(refresh.p-active.p)/.05,(rot(active.q).inv()*rot(refresh.q)).as_rotvec()/np.pi]
  return np.r_[obs,tail].astype(np.float32)
 def execution_action(self,a):return (a-6,True) if a>=8 else (a,False)
 def decide(self,obs):
  if len(obs)<NOBS:obs=np.r_[obs,np.zeros(NOBS-len(obs))].astype(np.float32)
  tic=time.monotonic();a=int(self.callback(obs));stage=int(obs[44]);assert 0<=a<(3 if stage==0 else 12)
  self.actions.append(dict(stage=stage,step=len(self.env.records),id=a,obs=obs.tolist(),wall_s=time.monotonic()-tic));return a
 def feedback_transitions(self,d,source):
  ts=super().feedback_transitions(d,source)
  for t in ts:
   if len(t['next'])<NOBS:t['next']=list(t['next'])+[0.]*(NOBS-len(t['next']))
  return ts
class SemanticOption(ContinuationOption):
 def __call__(self,s):
  phase=int(np.argmax(s[64:71]))+1 if s[44]>=.5 else 0
  reason='grasp_lost' if s[42]<.5 else 'phase_end' if self.phase!=phase else None
  if self.termination=='chunk' or (self.termination=='phase' and reason) or (self.termination=='insertion_queue' and (reason=='grasp_lost' or (phase==7 and s[61]<=0))):self.active=False
  boundary=s[44]<.5 or not self.active
  a=int(self.actor(s) if boundary else self.reference(s))
  if boundary and s[44]>=.5:self.active=a!=0
  self.phase=phase;self.events.append(dict(boundary=bool(boundary),action=a,phase=phase,reason=reason,remaining=float(s[41])*100));return a
