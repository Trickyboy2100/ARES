"""Freeze actual design transitions; all contain the new complete executor semantics."""
import argparse,torch,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=[];sources={}
for name in ['design_a','design_b']:
 f=a.root/name/'REPLAY.pt';z=torch.load(f,weights_only=False);assert len(z)==128;rows+=z;sources[name]=hashlib.sha256(f.read_bytes()).hexdigest()
assert len(set(r['row']['seed']for r in rows))==16
data=[]
for r in rows:
 assert sum(t['duration']for t in r['ts'])==100
 if r['row']['entry']is not None:assert r['row']['prefix_equal']
 data.extend(r['ts'])
f=a.root/'SHARED_FRONTIER.pt';assert not f.exists();torch.save(dict(data=data,episodes=256,sources=sources),f)
(a.root/'SHARED_FRONTIER.json').write_text(json.dumps(dict(episodes=256,transitions=len(data),sources=sources,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),scope='new controller catalogue transitions, no old relabeling; charged to all learners'),indent=2));print(len(data))
