"""Frozen-reference path entropy diagnostic, NOT stochastic-policy Q truth."""
import sys,json,torch,numpy as np
from pathlib import Path
from continuation_learner import ContinuationLearner
from program_rule_option import Assets
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/continuation_retention_20261108');torch.set_num_threads(2);out=[]
for folder in sorted(r.glob('*/refresh64')):
 if not(folder/'COMPLETE.json').exists():continue
 state=torch.load(folder.parent/'REFERENCE_64.pt',weights_only=False);m=ContinuationLearner(Assets(),32301,state['arm']);m.load(state);episodes={x['row']['path']:x for x in torch.load(folder/'REPLAY.pt',weights_only=False)}
 for pair in json.loads((folder/'PAIRS.json').read_text()):
  result=dict(arm=folder.parent.name,seed=pair['seed'],step=pair['step'],baseline_action=pair['baseline'],branch_action=pair['action'],reference_sha=pair['reference_sha'],baseline_path=pair['baseline_path'],branch_path=pair['branch_path'],success_delta=pair['delta'],scope='Entropy evaluated on actual deterministic-reference paths. Not an on-policy stochastic Q estimate; entropy not training input.')
  for key,path in [('reference',pair['baseline_path']),('branch',pair['branch_path'])]:
   ts=episodes[path]['ts'];idx=next(i for i,t in enumerate(ts)if t['step']==pair['step']and t['s']==pair['s']);tail=ts[idx:];future=[t['s']for t in tail[1:]];bonus=0.
   if future:
    b,mask=m.context(future)
    with torch.no_grad():p=m.probs(m.norm(future),b,mask,future);bonus=float(-.01*(p*p.clamp_min(1e-9).log()).sum())
   result[key]=dict(reward=sum(t['r']for t in tail),future_decisions=len(future),path_entropy_bonus=bonus,remaining=100-pair['step'])
  result['reward_delta']=result['branch']['reward']-result['reference']['reward'];result['entropy_delta']=result['branch']['path_entropy_bonus']-result['reference']['path_entropy_bonus'];result['path_soft_delta']=result['reward_delta']+result['entropy_delta'];out.append(result)
(r/'OPTION_ENTROPY_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2));non=[x for x in out if x['success_delta']];print(dict(pairs=len(out),non_ties=len(non),sign_reversals=sum(x['reward_delta']*x['path_soft_delta']<0 for x in non),max_abs_entropy_delta=max(abs(x['entropy_delta'])for x in out),max_non_tie_entropy_delta=max(abs(x['entropy_delta'])for x in non)))
