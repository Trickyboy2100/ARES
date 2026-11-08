"""Native-reset, version-bound full-option branches; no state restoration.
Queue is evidence only, never added to the deployed actor observation.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from program_full_option import FullJoint, Warm, legal
from program_rule_option import Assets, prefix_hash
from program_atomic_feedback import AtomicRuntime
from train_full_option import Learner
from train_option_online import atomic_json, atomic_torch

def digest(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=32);p.add_argument('--reference',type=Path);p.add_argument('--freeze-front',action='store_true');a=p.parse_args(argv)
 torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
 assets=Assets();warm=Warm(assets);ref=warm
 if a.reference:
  state=torch.load(a.reference,weights_only=False);ref=Learner(assets,32301,state['arm']);ref.load(state)
 version=hashlib.sha256(a.reference.read_bytes()).hexdigest()if a.reference else digest(warm.model.actor.state_dict().__repr__())
 # Hash actual warm parameter bytes, not a truncated tensor representation.
 if not a.reference:
  version=hashlib.sha256(b''.join(v.detach().numpy().tobytes()for v in warm.model.actor.state_dict().values())).hexdigest()
 atomic_json(a.out/'MANIFEST.json',dict(scenes=list(range(a.start,a.start+a.count)),reference_sha=version,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection='scene-index cyclic reachable rear boundary and legal alternatives, before outcomes',suffix='same frozen recurrent policy after complete Option',native_steps=100,scope='training only',prefix='original reset; measured state/action plus queue/cursor/active reference equality; not hidden PhysX equality'))
 rt=AtomicRuntime(a.out/'physics');episodes=[];pairs=[]
 try:
  for seed in range(a.start,a.start+a.count):
   visited=[]
   def baseline(s):
    selected=warm.choose(s)if a.freeze_front and s[44]<.5 else ref.choose(s);visited.append(dict(step=len(rt.env.records),s=s.tolist(),a=selected));return selected
   pol=FullJoint(assets,baseline,lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,'reference');ts=pol.transitions(ts)
   episodes.append(dict(row=row,ts=ts,events=pol.events));candidates=[v for v in visited if v['s'][44]>=.5 and sum(legal(v['s']))>1]
   if candidates:
    v=candidates[(seed-a.start)%len(candidates)];n=v['step'];acts=[int(k)for k in np.flatnonzero(legal(v['s']))if k!=v['a']];offset=(seed-a.start)%len(acts);acts=(acts[offset:]+acts[:offset])[:2]
    original=next(x for x in d['summary']['decisions']if x['step']==n)
    for ac in acts:
     hit=[]
     def branch(s):
      if len(rt.env.records)==n:
       assert np.array_equal(s,np.asarray(v['s']));hit.append(True);return ac
      return warm.choose(s)if a.freeze_front and s[44]<.5 else ref.choose(s)
     bp=FullJoint(assets,branch,lambda:len(rt.env.records));br,bt,bd=rt.run(seed,bp,f'branch_{ac}');bt=bp.transitions(bt)
     assert len(hit)==1
     decision=next(x for x in bd['summary']['decisions']if x['step']==n)
     same=prefix_hash(d,n)==prefix_hash(bd,n)
     queue=all(original[k]==decision[k]for k in ['queue_before','active_target_before'])
     assert same and queue,(seed,n,'prefix mismatch')
     t=next(t for t in bt if t['step']==n)
     pair=dict(seed=seed,step=n,s=v['s'],baseline=v['a'],action=ac,baseline_success=bool(row['success']),success=bool(br['success']),delta=int(br['success'])-int(row['success']),duration=t['duration'],reference_sha=version,prefix_sha=prefix_hash(d,n),queue_sha=digest({k:original[k]for k in ['queue_before','active_target_before']}),baseline_path=row['path'],branch_path=br['path'])
     pairs.append(pair);episodes.append(dict(row=br,ts=bt,events=bp.events));print(seed,n,v['a'],ac,pair['delta'],flush=True)
   atomic_json(a.out/'PAIRS.json',pairs);atomic_torch(a.out/'REPLAY.pt',episodes);atomic_json(a.out/'RESULTS.json',[e['row']for e in episodes])
 finally:rt.close()
if __name__=='__main__':main()
