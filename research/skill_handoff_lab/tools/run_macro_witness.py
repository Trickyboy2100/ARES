import json,argparse
from pathlib import Path
from program_atomic_feedback import AtomicRuntime
from peg_atomic_macro import AtomicMacro
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rt=AtomicRuntime(a.out/'physics')
def actor(s):return 1 if s[44]<.5 else 10 if s[49]<=6/7+1e-5 and s[41]<=.28 else 0
try:
 p=AtomicMacro(actor,lambda s:0);r,t,d=rt.run(760213,p,'atomic_macro');(a.out/'EVENTS.json').write_text(json.dumps(p.events));assert r['success'],'physical macro witness failed';print('MACRO_WITNESS_PASS',flush=True)
finally:rt.close()
