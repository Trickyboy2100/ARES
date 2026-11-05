"""Physical regression of early-contact rejection and true post-event adoption."""
import argparse,gzip,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
old=json.loads((a.root/'tool_early/RESULTS.json').read_text());new=json.loads((a.root/'tool_early_v2/RESULTS.json').read_text())
def key(v):return v['seed'],v['grasp'],v['mode']
oo={key(v):v for v in old};nn={key(v):v for v in new}
def prefix(x,y):
    dx=json.load(gzip.open(x['path'],'rt'));dy=json.load(gzip.open(y['path'],'rt'));assert dx['initial']==dy['initial'];n=0
    for u,v in zip(dx['steps'],dy['steps']):
        if any(u[k]!=v[k] for k in ['q','qvel','tcp','tool','cube','action']):break
        n+=1
    return n
checks=[]
for v in new:
    k=key(v);base=nn[(v['seed'],v['grasp'],'none')]
    if v['mode']=='none':
        n=prefix(oo[k],v);assert n==100
        checks.append(dict(seed=v['seed'],grasp=v['grasp'],kind='original_nominal_preserved',identical_steps=n))
    elif v['early_resolution']=='none':
        n=prefix(base,v);assert n==100
        checks.append(dict(seed=v['seed'],grasp=v['grasp'],kind='declined_intervention_preserves_full_nominal',identical_steps=n))
    elif v['early_event']:
        n=prefix(base,v);assert n>=v['early_event']['step']
        checks.append(dict(seed=v['seed'],grasp=v['grasp'],kind='correction_only_after_measured_event',identical_steps=n,event_step=v['early_event']['step']))
out=dict(checks=checks,passed=True,scope='actual complete original-reset executions; no mid-episode state restore; shared runtime repair, not RL gain')
(a.root/'TOOL_CONTINUATION_CHECKS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
