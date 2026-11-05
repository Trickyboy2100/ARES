"""Observed replay support, actual freedom and recurrence, not future labels."""
import argparse,json,collections
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out={}
for folder in a.root.iterdir():
 if not (folder/'CURRENT.pt').exists():continue
 d=torch.load(folder/'CURRENT.pt',weights_only=False);data=[t for t in d['data']if t.get('episode',0)>=0];front=[t for t in data if t['stage']==0];rear=[t for t in data if t['stage']==1]
 h=d['history'];training=[v for v in h if v['kind']=='train'];updates=d['model']['updates']
 own_grasp={str(g):dict(n=sum(t['a']==g for t in front),success=sum(t['outcome']for t in front if t['a']==g))for g in range(3)}
 sequences=[[e for e in v['events']if e['boundary']and e.get('stage',1)==1]for v in training]
 out[folder.name]=dict(own_episodes=len(front),own_grasp=own_grasp,own_option_boundaries=len(data),own_free_decisions=sum(sum(t['base_mask'])>1 for t in data),own_free_rear_decisions=sum(sum(t['base_mask'])>1 for t in rear),episodes_multiple_rear_adoptions=sum(sum(e['a']>0 for e in seq)>=2 for seq in sequences),episodes_post_adoption_decision=sum(any(e['a']>0 for e in seq[:-1])for seq in sequences),physical_action_by_selector={str(k):dict(collections.Counter(t['physical_a']for t in rear if t['a']==k))for k in range(3)},gradient_iterations=updates,shared_transitions=len(d['data'])-len(data),claim='real current-policy collection, sequential permission; not evidence of independent performance gain')
(a.root/'SUPPORT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
