"""New natural training scenes; complete physical reexecution of paired suffixes."""
import argparse,json,gzip,shutil,torch,numpy as np
from pathlib import Path
from program_atomic_feedback import AtomicRuntime
from program_rule_option import Assets,SingleDeviation,rule,prefix_hash,gate_features
from train_option_online import atomic_json
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,default=763000);p.add_argument('--count',type=int,default=64);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);assets=Assets();rt=AtomicRuntime(a.out/'physics');pairs=[];baselines=[]
for f in ['program_rule_option.py','collect_rule_pairs.py']:shutil.copy2(Path(__file__).with_name(f),a.out/f)
try:
 for seed in range(a.start,a.start+a.count):
  base=SingleDeviation(assets,lambda *x:None,lambda:len(rt.env.records));b,ts,bd=rt.run(seed,base,'baseline');baselines.append(b);atomic_json(a.out/'BASELINES.json',baselines)
  opportunities=[]
  for d in b['decisions']:
   if not d['stage']:continue
   s=np.asarray(d['obs']);ab=rule(s)
   for j,ac in enumerate(assets.proposals(s)):
    if ac!=ab:opportunities.append((d['step'],j,ac,s,ab))
  if not opportunities:print(seed,'no_disagreement',flush=True);continue
  rng=np.random.default_rng(seed);step,j,ac,s,ab=opportunities[int(rng.integers(len(opportunities)))];pol=SingleDeviation(assets,lambda s,b,c,t:j if t==step else None,lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,'candidate');assert pol.entry and pol.entry['ac']==ac
  assert prefix_hash(bd,step)==prefix_hash(d,step),'recorded physical prefix mismatch'
  pair=dict(scene_group='new_natural_train',seed=seed,physical_prefix_hash=prefix_hash(bd,step),z=s.tolist(),features=gate_features(s,ab,ac).tolist(),a_b=ab,a_c=ac,proposal=j,execution_interface='atomic83_single_deviation_v1',front_grasp=b['decisions'][0]['id'],option_duration=pol.metadata()['option_duration'],baseline_suffix_policy='frozen_rule_9a30507',G_b=int(b['success']),G_candidate=int(row['success']),delta=int(row['success'])-int(b['success']),budget=100-step,baseline=b,candidate=row,option=pol.metadata())
  pairs.append(pair);atomic_json(a.out/'PAIRS.json',pairs);print(seed,'pair',step,ab,ac,pair['delta'],flush=True)
finally:rt.close()
