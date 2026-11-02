"""Numerical ranking regression: probability saturation must not choose ties."""
import json
import numpy as np
import torch
from program_stap_kernel import Agent,Box,Critic,Primitive,Constant,port_plan
from program_skillresult_logkernel import stable_port_plan
class D:
 def rollout(self,obs,sk,policies,batch_size=None,time_index=False):
  n=batch_size or 1;s=obs[None].repeat(n,1);a=policies[0].actor.predict(s);ss=torch.stack([s,s],1);aa=a[:,None,:];return (ss,aa) if batch_size else (ss[0],aa[0])
 def decode(self,s,primitive=None):return s
class Actor:
 def predict(self,s):return torch.zeros((len(s),1))
def run():
 results=[]
 for bias in [0.,20.,100.]:
  logits=lambda s,a:bias+a[:,0]
  p=Agent(Box([-1],[1]),Actor(),Critic(lambda s,a:torch.sigmoid(logits(s,a))))
  lp=Agent(Box([-1],[1]),Actor(),Critic(lambda s,a:torch.nn.functional.logsigmoid(logits(s,a).double())))
  torch.manual_seed(731);ordinary=port_plan(np.array([0.],np.float32),[Primitive(0)],[p],D(),num_iterations=4,num_samples=32,num_elites=4)
  torch.manual_seed(731);stable=stable_port_plan(np.array([0.],np.float32),[Primitive(0)],[lp],D(),num_iterations=4,num_samples=32,num_elites=4)
  assert float(stable['actions'][0,0])==1.;results.append(dict(logit_bias=bias,probability_rank_action=float(ordinary['actions'][0,0]),log_rank_action=float(stable['actions'][0,0])))
 print(json.dumps(dict(status='PASS',cases=results,scope='Synthetic numerical fixture only, zero physical evidence'),indent=2))
if __name__=='__main__':run()
