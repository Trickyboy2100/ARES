"""Execute exact settling-hook AST with fake physical-step counter; no simulation."""
import argparse,ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 path=a.tools/'program_skill_baselines.py';tree=ast.parse(path.read_text())
 fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='completion_reached')
 ns={'np':np,'serial':lambda x:x};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
 class Stub:
  completion_reached=ns['completion_reached']
  def __init__(self,k,confirm=None,exit_after=None):
   self.completion_mode='dwell';self.gate=SimpleNamespace(distance_enter_m=.004,angle_enter_rad=.12,distance_exit_m=.0046,angle_exit_rad=.16);self.last_probe=-100;self.probe_steps=0;self.hold_entry=None;self.k0=k;self.confirm=confirm;self.exit_after=exit_after;self.calls=[]
   robot=SimpleNamespace(get_qpos=lambda:np.zeros((1,9)))
   self.env=SimpleNamespace(records=[{} for _ in range(k)],unwrapped=SimpleNamespace(agent=SimpleNamespace(robot=robot)),step=self.step)
  def step(self,a):
   assert len(self.env.records)<200,'native budget overrun';self.calls.append(np.array(a));self.env.records.append({})
  def completion_measure(self):
   n=len(self.env.records)-self.k0;return (self.confirm is not None and n>=self.confirm,.005 if self.exit_after is not None and n>=self.exit_after else .003,.02)
 checks=[]
 def check(n,x):checks.append(dict(name=n,pass_=bool(x)))
 s=Stub(100);check('unconfirmed_probe_bounded6',not s.completion_reached() and s.probe_steps==6);s.completion_reached();check('duplicate_completion_call_no_new_probe',s.probe_steps==6)
 s.step(np.zeros(8));s.step(np.zeros(8));s.completion_reached();check('cooldown8_from_probe_start',s.probe_steps==12)
 s=Stub(197);s.completion_reached();check('probe_capped_at_native200',len(s.env.records)==200 and s.probe_steps==3);s.completion_reached();check('duplicate_at_horizon_no_steps',len(s.env.records)==200)
 s=Stub(100,confirm=3);check('confirmation_breaks_probe',s.completion_reached() and s.probe_steps==3 and s.hold_entry==103);s.completion_reached();check('confirmed_duplicate_no_extra_motion',s.probe_steps==3)
 s=Stub(100,exit_after=1);s.completion_reached();check('outer_exit_aborts_probe',s.probe_steps==1)
 check('probe_holds_current_q_and_closed_gripper',all(np.array_equal(x,np.r_[np.zeros(7),-1.]) for x in s.calls))
 contact=a.tools/'program_adaptive_contact.py';t=ast.parse(contact.read_text());step=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='step' and any(isinstance(z,ast.Name) and z.id=='authorities' for z in ast.walk(n)))
 loops=[n for n in ast.walk(step) if isinstance(n,ast.For)]
 # Exact source guard is static evidence, separate from the hook execution.
 guarded=any(isinstance(loop.body[0],ast.If) and '200' in ast.unparse(loop.body[0].test) and any(isinstance(x,ast.Break) for x in ast.walk(loop.body[0])) for loop in loops)
 check('outer_servo_loop_rechecks_remaining_budget_static',guarded)
 out=dict(device='CPU',physics_steps=0,checks=checks,all_pass=all(c['pass_'] for c in checks),sources={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [path,contact]},limitation='Exact hook executed against stub step counter, not PhysX or full Episode.step; outer-loop guard is AST inspection.')
 a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
