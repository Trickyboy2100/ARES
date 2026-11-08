import argparse,json,hashlib
from pathlib import Path
import torch
from continuation_data import build
from continuation_learner import ContinuationLearner
from program_rule_option import Assets
from train_option_online import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
folders=[a.root/'design_a',a.root/'design_b'];episodes=[]
for f in folders:
 z=torch.load(f/'REPLAY.pt',weights_only=False);assert len(set(e['row']['seed']for e in z))==32;episodes+=z
assert set(e['row']['seed']for e in episodes)==set(range(768000,768064))
m=ContinuationLearner(Assets(),32301,'sac');d=build(folders,m);out=a.root/'SHARED.pt';assert not out.exists();atomic_torch(out,d)
atomic_json(a.root/'SHARED.json',dict(episodes=d['episodes'],transitions=len(d['data']),pairs=len(d['pairs']),non_ties=sum(p['delta']!=0 for p in d['pairs']),anchors=len(d['anchors']),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),sources=d['sources'],cost='charged in full to every consuming arm; unique physical executions counted once globally'))
print(json.dumps(json.loads((a.root/'SHARED.json').read_text()),indent=2))
