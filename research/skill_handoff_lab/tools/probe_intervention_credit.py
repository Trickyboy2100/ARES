"""Frozen actor/critic probe at measured paired training states; not physics."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from program_budget_feedback_sac import FeedbackSAC
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);out=[]
for folder in a.root.iterdir():
 if not folder.is_dir() or not (folder/'FINAL.pt').exists():continue
 pairs_path=folder/'PAIRS.json'
 if not pairs_path.exists():pairs_path=a.root/'zero_data/PAIRS.json'
 pairs=json.loads(pairs_path.read_text());
 if not pairs:continue
 for cp in sorted(folder.glob('*.pt')):
  m=FeedbackSAC.load(cp);x=m.x([t['s'] for t in pairs]);idx=torch.arange(len(pairs));act=torch.tensor([t['a'] for t in pairs])
  with torch.no_grad():
   p=m.probs(x);q=torch.stack([v(x) for v in m.q]).mean(0);gap=(q[idx,act]-q[:,0]).numpy()
  out.append(dict(model=str(cp.relative_to(a.root)),states=[dict(scene=t['scene'],a=t['a'],measured_gap=t['delta'],q_gap=float(gap[i]),p_action=float(p[i,t['a']]),p_zero=float(p[i,0]),argmax=int(p[i].argmax())) for i,t in enumerate(pairs)]))
(a.root/'POLICY_PROBE.json').write_text(json.dumps(out,indent=2));print('PROBED',len(out))
