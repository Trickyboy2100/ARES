"""Frozen legacy-Q gaps versus complete measured single-deviation outcomes."""
import argparse,json,torch
from pathlib import Path
from program_rule_option import Assets
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();assets=Assets();rows=[];torch.set_num_threads(2)
with torch.no_grad():
 for v in json.loads((a.root/'pairs/PAIRS.json').read_text()):
  m=assets.models[v['proposal']];q=torch.stack([n(m.x([v['z']]))for n in m.q]).mean(0)[0]
  rows.append(dict(seed=v['seed'],delta=v['delta'],old_Q_gap=float(q[v['a_c']]-q[v['a_b']]),baseline=v['a_b'],candidate=v['a_c']))
(a.root/'OLD_Q_CALIBRATION.json').write_text(json.dumps(rows,indent=2))
for k in [-1,0,1]:
 s=[v for v in rows if v['delta']==k];print(k,len(s),'positive old gap',sum(v['old_Q_gap']>0 for v in s))
