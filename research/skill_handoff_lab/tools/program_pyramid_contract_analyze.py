"""Casewise physical opportunity, matched prefixes, and native predicates."""
import argparse,gzip,json,hashlib,itertools,collections
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 groups={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 for f in sorted(a.root.glob('*/RESULTS.json')):
  rows=json.loads(f.read_text());data=[]
  for r in rows:
   path=Path(r['path']);assert sha(path)==r['sha256'];d=json.load(gzip.open(path,'rt'));assert len(d['transitions'])==250
   last=d['transitions'][-1];assert bool(last['info']['success'][0])==r['native_success']
   r['category']='success' if r['contract_success'] else 'predecessor_failure' if not r['boundaries'] or not r['boundaries'][0]['valid'] else 'successor_failure' if len(r['boundaries'])<2 or not r['boundaries'][1]['valid'] else 'terminal_contract_failure'
   data.append(d)
  assert all(len({r['initial_sha256'] for r in rows if r['seed']==seed})==1 for seed in {r['seed'] for r in rows})
  contexts=collections.defaultdict(list)
  for r in rows:contexts[r['seed'],r['yaw']].append(r)
  fixed=collections.Counter();reversals=[];matched=[]
  for key,rr in contexts.items():
   for r in rr:fixed[r['gap'],r['offset']]+=r['contract_success']
  for seed in sorted({r['seed'] for r in rows}):
   left={(r['gap'],r['offset']):r for r in rows if r['seed']==seed and r['yaw']==0};right={(r['gap'],r['offset']):r for r in rows if r['seed']==seed and r['yaw']==45}
   for x,y in itertools.combinations(left,2):
    if (int(left[x]['contract_success'])-int(left[y]['contract_success']))*(int(right[x]['contract_success'])-int(right[y]['contract_success']))<0:reversals.append(dict(seed=seed,actions=[x,y]))
   for gap in [-1,0,1]:
    ds=[d for d in data if d['summary']['seed']==seed and d['summary']['gap']==gap];prefixes=[]
    for d in ds:
     if len(d['decisions'])>1:
      step=d['decisions'][1]['boundary_step'];prefixes.append(hashlib.sha256(json.dumps(d['transitions'][:step],sort_keys=True).encode()).hexdigest())
    matched.append(dict(seed=seed,gap=gap,count=len(prefixes),exact=len(set(prefixes))<=1))
  groups[f.parent.name]=dict(episodes=len(rows),steps=250*len(rows),native_success=sum(r['native_success'] for r in rows),contract_success=sum(r['contract_success'] for r in rows),categories=dict(collections.Counter(r['category'] for r in rows)),errors=dict(collections.Counter(str(r['error']) for r in rows if not r['contract_success'])),context_success_hist=dict(collections.Counter(sum(r['contract_success'] for r in rr) for rr in contexts.values())),fixed={str(k):v for k,v in fixed.items()},best_fixed=max(fixed.values()),oracle=sum(any(r['contract_success'] for r in rr) for rr in contexts.values()),strict_reversals=reversals,prefix_matching=matched,total_wall_s=sum(r['wall_s'] for r in rows),execution_planning_wall_s=sum(r['planning_wall_s'] for r in rows),rows=rows)
 paired=[]
 if 'symmetry' in groups and 'feasible' in groups:
  for old in groups['symmetry']['rows']:
   new=next(r for r in groups['feasible']['rows'] if (r['seed'],r['gap'],r['yaw'],r['offset'])==(old['seed'],old['gap'],old['yaw'],old['offset']))
   left=json.load(gzip.open(old['path'],'rt'));right=json.load(gzip.open(new['path'],'rt'));end=left['decisions'][1]['boundary_step']
   same=left['transitions'][:end]==right['transitions'][:end]
   assert same and old['initial_sha256']==new['initial_sha256']
   paired.append(dict(seed=old['seed'],gap=old['gap'],yaw=old['yaw'],offset=old['offset'],prefix_exact=same,through_step=end,rescue=bool(new['contract_success'] and not old['contract_success']),regression=bool(old['contract_success'] and not new['contract_success'])))
 (a.out/'MATCHED_CONTROL.json').write_text(json.dumps(paired,indent=2))
 groups={k:groups[k] for k in ['pilot','symmetry','feasible'] if k in groups}
 (a.out/'SUMMARY.json').write_text(json.dumps(groups,indent=2))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,len(groups),figsize=(6*len(groups),5),squeeze=False)
 for ax,(label,g) in zip(axs[0],groups.items()):
  rr=g['rows'];contexts=sorted({(r['seed'],r['yaw']) for r in rr});matrix=np.array([[next(int(r['contract_success']) for r in rr if (r['seed'],r['yaw'],r['gap'],r['offset'])==(*ctx,gap,off)) for gap in [-1,0,1] for off in [-1,0,1]] for ctx in contexts]);ax.imshow(matrix,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto');
  for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='white',fontweight='bold')
  ax.set_yticks(range(len(contexts)),[f'{s} / yaw {y}' for s,y in contexts]);ax.set_xticks(range(9),[f'{g}/{o}' for g in [-1,0,1] for o in [-1,0,1]],rotation=45);ax.set_xlabel('Gap correction / top shift (units: 6 mm / 8 mm)');ax.set_title(f'{label}\nNative {g["native_success"]}/{g["episodes"]}; contract {g["contract_success"]}/{g["episodes"]}')
 fig.suptitle('Measured two-skill physical outcomes — NO NEW RL CLAIM (green 1 = pass, red 0 = fail)');fig.tight_layout();fig.savefig(a.out/'KEY_FIGURE.png',dpi=300);fig.savefig(a.out/'KEY_FIGURE.pdf');print(json.dumps({k:{n:v for n,v in g.items() if n!='rows'} for k,g in groups.items()},indent=2))
if __name__=='__main__':main()
