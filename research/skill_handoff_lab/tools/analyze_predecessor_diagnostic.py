import json,gzip
from pathlib import Path
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/decoupled_interface_20261105');out=[]
for g in range(3):
 a=json.loads((r/f'diag_measured_g{g}/RESULTS.json').read_text());b=json.loads((r/f'diag_relative_g{g}/RESULTS.json').read_text())
 for x,y in zip(a,b):
  assert x['seed']==y['seed'];dx=json.load(gzip.open(x['path'],'rt'));dy=json.load(gzip.open(y['path'],'rt'))
  da=[d for d in dx['summary']['decisions']if d['stage']==1];db=[d for d in dy['summary']['decisions']if d['stage']==1];n=min(da[0]['step'],db[0]['step']) if da and db else 100
  exact=dx['initial']==dy['initial'] and all(xx[k]==yy[k]for xx,yy in zip(dx['steps'][:n],dy['steps'][:n])for k in ['q','qvel','tcp','peg','action']);assert exact
  out.append(dict(seed=x['seed'],grasp=g,measured_success=x['success'],relative_success=y['success'],identical_prefix_steps=n,exact_prefix=exact,measured_actions=[(d['step'],d['id'])for d in da],relative_actions=[(d['step'],d['id'])for d in db],measured_source=x['sha256'],relative_source=y['sha256']))
(r/'PREDECESSOR_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
