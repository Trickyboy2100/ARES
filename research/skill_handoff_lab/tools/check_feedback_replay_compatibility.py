import argparse,json,gzip
from pathlib import Path
import numpy as np
from program_peg_feedback import FeedbackRuntime
p=argparse.ArgumentParser();p.add_argument('--adapted',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);manifest=json.loads((a.adapted/'COMPATIBILITY.json').read_text());rt=FeedbackRuntime(a.out/'physics');checks=[]
for entry in manifest['accepted'][:2]:
 d=json.load(gzip.open(entry['path'],'rt'));acts=d['summary']['decisions'];i=[0]
 def callback(obs):
  old=acts[i[0]];assert np.max(np.abs(obs-np.array(old['obs'])))<1e-6; i[0]+=1;return old['id']
 row,ts,new=rt.run(d['summary']['seed'],callback,'actual_reexecution')
 assert i[0]==len(acts)
 errors={k:max(np.max(np.abs(np.array(x[k])-np.array(y[k])))for x,y in zip(d['steps'],new['steps']))for k in ['action','q','qvel','peg','tcp']}
 assert max(errors.values())<1e-6,errors
 checks.append(dict(source=entry,actual=row['path'],max_error=errors))
rt.close();(a.out/'CHECKS.json').write_text(json.dumps(checks,indent=2));print('PASS',len(checks),'full native100 reexecutions')
