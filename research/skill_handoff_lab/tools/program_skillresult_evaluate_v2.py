"""Actual native-horizon evaluation; shared frozen controller and first Q policy."""
import sys,json,time
from pathlib import Path
import numpy as np
import torch
from program_stap_learning import Selector,Net,features
import program_stap_evaluate as runner
import program_stap_learning as learning
from program_skillresult_logkernel import stable_port_plan
learning.port_plan=stable_port_plan

class ResultSelector(Selector):
 def __init__(self,path,method,samples=16):
  super().__init__(path,method if method.startswith('stap') or method=='foresight' else 'fq',samples)
  self.requested_method=method;data=torch.load(path.parent/'SUCCESSOR.pt',map_location='cpu',weights_only=False);self.heads={};self.temps={};self.ensemble=[]
  for name,p in data['models'].items():
   n=Net(46,5);n.load_state_dict(p['weights']);n.eval();self.heads[name]=n;self.temps[name]=p['temperature']
  for w in data['ensemble']:
   n=Net(46,5);n.load_state_dict(w);n.eval();self.ensemble.append(n)
  oldq=self.model.q
  def q(stage,s,a):
   if stage==0:return torch.nn.functional.logsigmoid(self.model.value('q0',s,a)[:,0].double())
   x=torch.cat([self.model.norm(s),a],-1)
   if method=='stap_uq':
    probs=torch.stack([torch.sigmoid(n(x)[:,0].double()/self.temps['ordinary']) for n in self.ensemble]);return torch.log((probs.mean(0)-probs.std(0)).clamp(1e-15,1-1e-15))
   return torch.nn.functional.logsigmoid(self.heads['ordinary'](x)[:,0].double()/self.temps['ordinary'])
  if method.startswith('stap'):self.model.q=q
 def choose(self,env,stage):
  decision_start=time.monotonic();name=self.requested_method
  if name=='fixed90':
   value=.09 if stage==0 else -.065;self.logs.append(dict(method=name,stage=stage,step=len(env.records),skill_model_queries=0,wall_s=0.,selection='new-training-data strong fixed90/-65 baseline'));return value
  if stage==0 and name=='geometry_fixed':
   self.logs.append(dict(method=name,stage=0,step=0,normalized_action=(.1-.0875)/.0175,skill_model_queries=0,wall_s=0.,selection='predeclared strong fixed100mm grasp'));return .1
  if stage==0 and not name.startswith('stap'):
   obs=features(env.measure(),100-len(env.records),env.unwrapped.density_scale,env.target_y,stage);grid=torch.linspace(-1,1,33).reshape(-1,1);states=torch.from_numpy(obs).repeat(33,1)
   with torch.no_grad():scores=self.model.value('foresight' if name=='foresight' else 'fq',states,grid)[:,0];idx=int(scores.argmax());action=float(grid[idx,0])
   self.logs.append(dict(stage=0,step=0,method=name,normalized_action=action,skill_model_queries=33,wall_s=time.monotonic()-decision_start,selection='logit ranking at first boundary',all_logits=scores.tolist()));return .0875+.0175*action
  if name.startswith('stap') or name=='foresight':
   value=super().choose(env,stage)
   if stage==1 and name=='stap_retain' and self.logs[-1]['p_success']<.05:
    action=self.logs[0]['planned_actions'][1][0];self.logs[-1]['confidence_abstention']=dict(threshold=.05,proposed_hook=value,retained_action=action);value=-.065+.01*action
   return value
  start=time.monotonic();obs=features(env.measure(),100-len(env.records),env.unwrapped.density_scale,env.target_y,stage)
  if name in ['geometry','geometry_fixed']:action=None;values=[]
  else:
   grid=torch.linspace(-1,1,33).reshape(-1,1);states=torch.from_numpy(obs).repeat(33,1)
   with torch.no_grad():scores=self.heads[name](torch.cat([self.model.norm(states),grid],-1))[:,0];values=torch.sigmoid(scores.double()/self.temps[name]).tolist()
   action=float(grid[int(scores.argmax()),0])
  self.logs.append(dict(method=name,stage=1,step=len(env.records),observation=obs.tolist(),normalized_action=action,all_values=values,all_logits=[] if action is None else scores.tolist(),skill_model_queries=len(values),wall_s=time.monotonic()-start,selection='measured SkillResult successor head; no future labels'))
  return None if action is None else -.065+.01*action

if __name__=='__main__':
 runner.Selector=ResultSelector;runner.main()
 out=Path(sys.argv[sys.argv.index('--out')+1]);(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 (out/'SELECTOR_SCOPE.json').write_text(json.dumps(dict(uq='Three scene-bootstrap critics, mean minus one standard deviation. STAP uncertainty-aware analogue, NOT original SCOD estimator or exact filter reproduction. 3x critic evaluations for uq.',calibration='Temperature fitted only on scenes 743046/47; no test tuning.',fairness='ordinary/generic/contract/relative/relative_contract/geometry share identical learned first-stage FQ. Full STAP has its own learned sequence optimizer. Terminal ordinary is also supervised scoring.'),indent=2))
