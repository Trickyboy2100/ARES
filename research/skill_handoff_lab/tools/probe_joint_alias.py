"""Native reexecution of two equivalent proposal labels under common suffix.
This is a training mechanism check, not a policy/generalization score.
"""
import argparse,json,torch,gzip
from pathlib import Path
from train_joint_continuation import Assets,Joint,AtomicRuntime,atomic_json
from program_rule_option import prefix_hash
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
h=json.loads((a.run/'HISTORY.json').read_text());assets=Assets();rt=AtomicRuntime(a.out/'physics');rows=[]
try:
 for seed,step in [(764124,86),(764126,78)]:
  original=next(v for v in h if v['kind']=='train'and v['row']['seed']==seed);recorded={e['step']:e['a']for e in original['events']if e['boundary']}
  for branch in [1,2]:
   checked=[]
   def actor(s):
    now=len(rt.env.records)
    if now<step:
     assert now in recorded,'Prefix schedule mismatch';return recorded[now]
    if now==step:
     cs=assets.proposals(s);assert cs[0]==cs[1];checked.append(dict(step=now,physical=cs[0]));return branch
    return 0
   policy=Joint(assets,actor,lambda:len(rt.env.records));row,ts,d=rt.run(seed,policy,f'alias_{branch}');assert checked;source=json.load(gzip.open(original['row']['path'],'rt'));assert prefix_hash(d,step)==prefix_hash(source,step);row.update(alias=checked,branch=branch,source=original['row']['path']);rows.append(row);atomic_json(a.out/'RESULTS.json',rows);print(seed,branch,row['success'],flush=True)
 for seed in [764124,764126]:
  aa=[json.load(gzip.open(r['path'],'rt'))for r in rows if r['seed']==seed]
  keys=['q','qvel','tcp','peg','action'];assert all(all(x[k]==y[k]for k in keys)for x,y in zip(aa[0]['steps'],aa[1]['steps']))
 atomic_json(a.out/'CHECKS.json',dict(native_episodes=4,full100step_physical_equal=True,scope='selected training aliases, not RL improvement'))
finally:rt.close()
