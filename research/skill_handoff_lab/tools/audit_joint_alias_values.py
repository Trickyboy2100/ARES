"""Recompute execution-equivalent proposal value disagreements from frozen Q."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from train_joint_continuation import Assets,Learner,rule,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);assets=Assets();m=Learner(assets,31901);cp=a.root/'sac_31901/CHECKPOINT_64.pt';m.load(torch.load(cp,weights_only=False));data=torch.load(a.root/'sac_31901/CURRENT.pt',weights_only=False)['data'];aliases=[];maps={1:{},2:{}}
for t in data:
 if t['stage']==0:continue
 cs=assets.proposals(np.array(t['s']));ab=rule(t['s'])
 for i,c in enumerate(cs):maps[i+1][c]=maps[i+1].get(c,0)+1
 if cs[0]==cs[1]and cs[0]!=ab:
  with torch.no_grad():q=torch.stack([q(m.norm([t['s']]))for q in m.q]).mean(0)[0]
  aliases.append(dict(scene=t['scene'],step=t['step'],physical=cs[0],q_gap=float(q[1]-q[2])))
result=dict(checkpoint=str(cp),sha=sha(cp),proposal_physical_frequencies=maps,actual_equivalent_proposal_states=aliases,interpretation='A representation consistency issue; not proof of task-level improvement or cause of all failures')
old=a.root/'EXECUTION_IDENTITY_DIAGNOSIS.json'
if old.exists():assert json.loads(old.read_text())['actual_equivalent_proposal_states']==aliases
(a.root/'ALIAS_VALUES_VERIFIED.json').write_text(json.dumps(result,indent=2));print(len(aliases),'alias value gaps verified against frozen final checkpoint')
