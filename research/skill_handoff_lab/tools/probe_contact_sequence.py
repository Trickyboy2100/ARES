"""Read-only force instrumentation in two complete native witness executions."""
import argparse,json
from pathlib import Path
from program_atomic_feedback import AtomicRuntime,SemanticOption
from program_peg_native import val
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
for mode in ['phase','insertion_queue']:
 rt=AtomicRuntime(a.out/mode);old=rt.env.measure
 def measure():
  d=old();u=rt.env.unwrapped;d['peg_fixture_force_N']=val(u.scene.get_pairwise_contact_forces(u.peg,u.box)[0]);return d
 rt.env.measure=measure
 def actor(s):return 1 if s[44]<.5 else 10 if s[49]<=6/7+1e-5 and s[41]<=.28 else 0
 try:
  p=SemanticOption(actor,lambda s:0,mode);r,t,d=rt.run(760213,p,mode);rows.append(dict(termination=mode,**r));(a.out/f'{mode}_EVENTS.json').write_text(json.dumps(p.events));print(mode,r['success'],flush=True)
 finally:rt.close()
(a.out/'RESULTS.json').write_text(json.dumps(rows))
