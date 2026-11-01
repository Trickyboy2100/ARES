"""Execute exact original SearchControl AST on synthetic observation sequences."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.tools));from program_authority_search import AuthoritySearchControl
 source=a.tools/'program_adaptive_contact.py';cls=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='SearchControl');ns={'np':np};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns);Original=ns['SearchControl']
 checks=[]
 def check(n,b):checks.append(dict(name=n,pass_=bool(b)))
 sequence=[-.025]*5+[-.018]*14+[-.016]*12+[-.015]*11
 for auth,family,n in [('full','Y',9),('full','YZ',9),('y_only','Y',3),('y_only','YZ',3),('task','Y',3),('task','YZ',9)]:
  candidate=AuthoritySearchControl(auth,family,Original);check(f'{auth}_{family}_unique_count{n}',len(candidate.points)==n==len(set(candidate.points)))
  reference=Original();reference.points=list(candidate.points)
  if n==3:check(f'{auth}_{family}_preserves_first_order',candidate.points==((0,0.),(.8,0.),(-.8,0.)))
  equal=True;advance=True;observations_unmodified=True
  for x in sequence:
   obs=np.zeros(21);obs[0]=x/.025;before=obs.copy();aa=candidate(obs);bb=reference(obs)
   equal &= bool(np.array_equal(aa,bb) and all(getattr(candidate.base,k)==getattr(reference,k) for k in ['index','age','backoff','best','stall']))
   advance &= aa[2]==bb[2];observations_unmodified &= np.array_equal(obs,before)
  check(f'{auth}_{family}_original_state_transition_exact',equal and advance and observations_unmodified)
 baseline=Original();check('original_search_still9',len(baseline.points)==9)
 rejected=False
 try:AuthoritySearchControl('invalid','Y',Original)
 except ValueError:rejected=True
 check('invalid_authority_rejected',rejected)
 out=dict(physics_steps=0,optimizer_updates=0,synthetic_CPU_test_not_experiment=True,checks=checks,all_pass=all(x['pass_'] for x in checks),sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,a.tools/'program_authority_search.py']})
 a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
