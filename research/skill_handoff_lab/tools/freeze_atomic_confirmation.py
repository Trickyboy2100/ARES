"""Execute the declared equal checkpoint rule BEFORE reading new outcomes."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'FROZEN_SELECTION.json';assert not out.exists();rows=[]
for seed in [31701,31702]:
 for term in ['queue','macro']:
  name=f'{term}_bc64'if seed==31701 else f'{term}_seed{seed}';run=a.root/name;h=json.loads((run/'HISTORY.json').read_text());assert sum(v['job']['kind']=='train'for v in h)==64
  scores={}
  for c in [0,32,64]:
   z=[v for v in h if v['job']['kind']=='eval'and v['job']['checkpoint']==c];assert len(z)==16;scores[c]=sum(v['row']['success']for v in z)
  k=max(scores,key=lambda c:(scores[c],-c));model=a.root/f'warm{seed}/INITIAL.pt'if k==0 else run/f'CHECKPOINT_{k}.pt'
  rows.append(dict(seed=seed,run=name,termination='insertion_queue'if term=='queue'else'macro',checkpoint=k,scores=scores,model=str(model),sha256=hashlib.sha256(model.read_bytes()).hexdigest(),budget=64))
out.write_text(json.dumps(dict(rule='predeclared in REPEAT_PROTOCOL; max0/32/64 earliest tie',conditions=list(range(762200,762224)),models=rows,sealed=False),indent=2));print(rows)
