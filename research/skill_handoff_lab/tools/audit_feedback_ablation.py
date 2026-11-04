import argparse,gzip,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out={}
for mode in ['ordinary','budget']:
 z=json.loads((r/f'zero_intervention_{mode}/RESULTS.json').read_text());full={x['seed']:x for x in json.loads((r/f'local_{mode}_31501/EVAL_FINAL.json').read_text())};pairs=[]
 for row in z:
  f=full[row['seed']];aa,bb=[json.load(gzip.open(x['path'],'rt'))for x in [f,row]];n=f['decisions'][1]['step'];assert row['decisions'][1]['step']==n
  err=max(float(np.max(np.abs(np.array(aa['steps'][i][k])-bb['steps'][i][k])))for i in range(n)for k in ['action','q','qvel','peg','tcp']);assert err==0
  pairs.append(dict(seed=row['seed'],grasp=f['decisions'][0]['id'],prefix_steps=n,prefix_max_difference=err,full_success=f['success'],zero_suffix_success=row['success']))
 out[mode]=dict(scope='Targeted seen regression cases; not an overall success estimate.',pairs=pairs,full_success=sum(x['full_success']for x in pairs),zero_suffix_success=sum(x['zero_suffix_success']for x in pairs),n=len(pairs))
(r/'ZERO_SUFFIX_ABLATION.json').write_text(json.dumps(out,indent=2));print({m:(x['full_success'],x['zero_suffix_success'],x['n'])for m,x in out.items()})
