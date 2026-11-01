"""CPU interface review: no physics, no GPU, no optimizer updates."""
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.tools))
 from program_completion_contract import CompletionGate
 from program_skill_baselines import BaselineEnv
 from program_adaptive_rl import SkillEnv
 results=[]
 def check(name,condition):
  results.append(dict(name=name,pass_=bool(condition)))
 gate=CompletionGate();r=[gate.update(1,.003,.02,.001,.01) for _ in range(5)];check('same_physical_frame_counted_once',gate.count==1 and not any(r))
 gate.update(2,.0043,.14,.001,.01);check('outer_hysteresis_retains_active',gate.active and gate.count==2)
 check('third_distinct_stable_frame_reaches',gate.update(3,.0043,.14,.001,.01))
 gate=CompletionGate();gate.update(1,.003,.02,.001,.01);gate.update(2,.0047,.02,.001,.01);check('distance_exit_resets_dwell',gate.count==0 and not gate.active)
 gate=CompletionGate();gate.update(1,.003,.02,.001,.01);gate.update(2,.003,.17,.001,.01);check('angle_exit_resets_dwell',gate.count==0 and not gate.active)
 gate=CompletionGate();gate.update(1,.003,.02,.001,.01);gate.update(2,.003,.02,.03,.01);check('motion_resets_dwell',gate.count==0 and not gate.reached)
 gate.update(3,.003,.02,.001,.01);gate.update(4,.003,.02,.001,.01);check('new_dwell_after_motion_not_early',not gate.reached);check('new_dwell_after_motion_completes',gate.update(5,.003,.02,.001,.01))
 source=(a.tools/'program_completion_contract.py').read_text();check('pure_gate_has_no_evaluate_or_bias_access','evaluate(' not in source and '.bias' not in source and 'goal_pose' not in source)
 # Record rather than conceal nonfinite/out-of-order edge behavior.
 gate=CompletionGate();gate.update(1,.003,.02,.001,.01)
 try:gate.update(2,float('nan'),.02,.001,.01);bad=gate.update(3,float('nan'),.02,.001,.01);nan_safe=not bad
 except (ValueError,AssertionError):nan_safe=True
 check('invalid_distance_cannot_confirm',nan_safe)
 gate=CompletionGate();gate.update(3,.003,.02,.001,.01)
 try:gate.update(2,.003,.02,.001,.01);retro_safe=gate.count==1
 except (ValueError,AssertionError):retro_safe=True
 check('reverse_step_not_counted',retro_safe)
 for authority,family,expect_z in [('full','Y',.7),('y_only','YZ',0.),('task','Y',0.),('task','YZ',.7)]:
  obj=BaselineEnv.__new__(BaselineEnv);obj.authority=authority;obj.family=family;obj.episode=SimpleNamespace();seen=[]
  def fake(self,action):seen.append(np.array(action));return np.arange(90,dtype=np.float32),0.,False,False,{}
  requested=np.array([.2,.7,-.1],dtype=np.float32)
  with patch.object(SkillEnv,'step',fake):obs,*_=obj.step(requested)
  check(f'projection_{authority}_{family}',len(seen)==1 and np.isclose(seen[0][1],expect_z) and requested[1]==np.float32(.7) and np.array_equal(obj.episode.policy_requested,requested))
  check(f'context92_{authority}_{family}',obs.shape==(92,) and np.array_equal(obs[:90],np.arange(90)) and np.array_equal(obs[-2:],[family=='Y',family=='YZ']))
 data=dict(physics_steps=0,optimizer_steps=0,device='CPU',checks=results,all_pass=all(x['pass_'] for x in results),sources={n:hashlib.sha256((a.tools/n).read_bytes()).hexdigest() for n in ['program_completion_contract.py','program_skill_baselines.py','program_adaptive_contact.py']})
 a.out.write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
if __name__=='__main__':main()
