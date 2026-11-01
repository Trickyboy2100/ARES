"""Exact completion_measure hook on synthetic CPU frame pairs; not physics data."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.tools));from program_completion_contract import CompletionGate
 path=a.tools/'program_skill_baselines.py';tree=ast.parse(path.read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='completion_measure');ns=dict(np=np,serial=lambda x:x,sp=lambda actor:actor.pose);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
 class Goal:
  def inv(self):return self
  def __mul__(self,p):return p
 def instance(prev,now,dt=.05):
  charger=SimpleNamespace(pose=SimpleNamespace(p=np.array(now[:3]),q=np.array(now[3:])),linear_velocity=np.array([[.4,0,0]]),angular_velocity=np.array([[18.,0,0]]))
  return SimpleNamespace(goal=Goal(),gate=CompletionGate(),env=SimpleNamespace(unwrapped=SimpleNamespace(charger=charger,control_timestep=dt),records=[{'tool':prev},{'tool':now}]))
 def measure(o):ns['completion_measure'](o);return o.env.records[-1]['observable_completion']
 checks=[]
 def check(name,v):checks.append(dict(name=name,pass_=bool(v)))
 base=[.003,0,0,1,0,0,0];o=instance(base,base);v=measure(o);check('stationary_frames_ignore_instantaneous_jitter',v['speed_m_s']==0 and v['measured_angular_speed_rad_s']==0 and o.gate.count==1);check('raw_instantaneous_velocity_preserved',v['instantaneous_speed_m_s']==.4 and v['instantaneous_angular_speed_rad_s']==18)
 measure(o);check('same_frame_repeat_not_new_dwell',o.gate.count==1)
 signed=base[:3]+[-1,0,0,0];v=measure(instance(base,signed));check('quaternion_double_cover_invariant',v['measured_angular_speed_rad_s']==0)
 moved=[.0035,0,0,1,0,0,0];v=measure(instance(base,moved));check('translation_divided_by_control_dt',np.isclose(v['speed_m_s'],.01));v2=measure(instance(base,moved,.025));check('control_dt_scaling',np.isclose(v2['speed_m_s'],.02))
 theta=.01;rot=base[:3]+[np.cos(theta/2),0,0,np.sin(theta/2)];v=measure(instance(base,rot));check('rotation_geodesic_rate',np.isclose(v['measured_angular_speed_rad_s'],.2,atol=1e-10))
 o=instance(base,base)
 for _ in range(3):measure(o);o.env.records.append({'tool':base})
 check('three_distinct_stable_frames_reach',o.gate.reached)
 out=dict(device='CPU',physics_steps=0,optimizer_updates=0,synthetic_unit_inputs_not_research_data=True,checks=checks,all_pass=all(x['pass_'] for x in checks),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),limitation='Observed control-rate pose stability can alias substep oscillation; not proof of zero instantaneous velocity or hardware sensor validation.')
 a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
