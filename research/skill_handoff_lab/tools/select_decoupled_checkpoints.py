"""Freeze declared development selection before fresh physical confirmation."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,choices=[31601,31602],required=True);a=p.parse_args();selected={}
for frame,prefix in [('measured','measured'),('remaining_target','relative')]:
 folder=a.root/(prefix+'_bc_extend64' if a.seed==31601 else prefix+'_bc_repeat128');h=json.loads((folder/'HISTORY.json').read_text());manifest=json.loads((folder/'MANIFEST.json').read_text());marks=[0,32,64]if a.seed==31601 else[64,128];scores=[]
 for k in marks:
  rows=[x['row']for x in h if x['job']['kind']=='eval' and x['job']['checkpoint']==k];assert len(rows)==16
  if k==0:cp=Path(manifest['initialize_run']['path'])/'FINAL.pt'
  else:cp=folder/f'CHECKPOINT_{k}.pt'
  assert cp.exists();scores.append(dict(cumulative=k+64 if a.seed==31601 else k,success=sum(x['success']for x in rows),path=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
 selected[frame]=dict(candidates=scores,chosen=sorted(scores,key=lambda x:(-x['success'],x['cumulative']))[0])
p=a.root/f'FROZEN_SELECTION_{a.seed}.json';assert not p.exists();p.write_text(json.dumps(dict(seed=a.seed,selection=selected,confirmation_seeds=list(range(760200,760216)),rule='highest dev success, earliest tie; frozen before new outcomes',scope='common inherited initialization; independent exploration/replay seed'),indent=2));print(p.read_text())
