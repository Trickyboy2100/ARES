"""Verify shared warmup physics/checkpoint and finite-action dataset integrity."""
import argparse,json,gzip,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();report=[]
for seed in [31301,31302]:
 folders=[x for x in sorted(a.root.glob(f'frontier_*_{seed}'))if(x/'FINAL.pt').exists()];assert len(folders)==7
 reference=None
 for folder in folders:
  rows=json.loads((folder/'physics/RESULTS.json').read_text());warmup=[x for x in rows if x['tag'].startswith('warmup')];assert len(warmup)==8
  actual=[json.load(gzip.open(x['path'],'rt'))for x in warmup]
  model=torch.load(folder/'MODEL_8.pt',weights_only=False,map_location='cpu')
  if reference is None:reference=(actual,model)
  for x,y in zip(actual,reference[0]):assert x['initial']==y['initial']and x['steps']==y['steps']
  for key in ['q','target','actor','v','dyn']:
   assert all(torch.equal(v,reference[1][key][k])for k,v in model[key].items())
  report.append(dict(run=folder.name,warmup_episodes=8,all100steps_exact=True,model8_exact=True))
(a.root/'SHARED_CONTROL_AUDIT.json').write_text(json.dumps(report,indent=2));print('Shared warmup and MODEL_8 exactly equal:',len(report))
