"""Check prefix identity and zero-action equivalence before timing interpretation."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;design=json.load(open(root/'design/RESULTS.json'));out={}
for folder in ['first_contact','pull_rejoin']:
 if not(root/folder/'RESULTS.json').exists():continue
 rows=json.load(open(root/folder/'RESULTS.json'));groups=collections.defaultdict(list)
 for r in rows:groups[(r['seed'],r['grasp'])].append(r)
 results=[]
 for k,rs in groups.items():
  zero=next(r for r in rs if r['action']==[0,0]);z=json.load(gzip.open(zero['path'],'rt'));reference=next(r for r in design if(r['seed'],r['grasp'])==k and r['action']==[0,0]);old=json.load(gzip.open(reference['path'],'rt'));same_zero=z['steps']==old['steps'];checks=[]
  for r in rs:
   d=json.load(gzip.open(r['path'],'rt'));assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'];at=zero['contact_step'];same=at is not None and z['steps'][:at]==d['steps'][:at];checks.append(dict(action=r['action'],same_prefix=same,success=r['success'],distance=r['final_distance'],error=r['error']))
  results.append(dict(seed=k[0],grasp=k[1],zero_matches_full_original_100step_trace=same_zero,contact_step=zero['contact_step'],contact_phase=z['contact_event']['phase']if z['contact_event']else None,comparisons=checks))
 out[folder]=dict(episodes=len(rows),success=sum(r['success']for r in rows),all_zero_traces_match=all(r['zero_matches_full_original_100step_trace']for r in results),strict_prefix_pairs=sum(c['same_prefix']for r in results for c in r['comparisons']if c['action']!=[0,0]),contexts=results)
(root/'analysis/FIRST_CONTACT_PAIRS.json').write_text(json.dumps(out,indent=2));print({k:{x:v[x]for x in ['episodes','success','all_zero_traces_match','strict_prefix_pairs']}for k,v in out.items()});assert all(v['all_zero_traces_match']for v in out.values())
