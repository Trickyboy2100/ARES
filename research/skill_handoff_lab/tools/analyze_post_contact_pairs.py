"""Exact-prefix physical pair audit and continuous parameter effects."""
import json,gzip,argparse,hashlib,collections
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result={}
for folder in ['pilot','local','relative','pitch','near','design']:
 f=a.root/folder/'RESULTS.json'
 if not f.exists():continue
 rows=json.loads(f.read_text());groups=collections.defaultdict(list)
 for r in rows:groups[(r['seed'],r['grasp'])].append(r)
 summary=[]
 for key,rs in groups.items():
  ref=next(r for r in rs if r['action']==[0,0]);rd=json.load(gzip.open(ref['path'],'rt'));at=ref['contact_step'];checks=[]
  for r in rs:
   path=Path(r['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(path,'rt'));checks.append(dict(action=r['action'],same_initial=r['initial_sha256']==ref['initial_sha256'],same_prefix=at is not None and d['steps'][:at]==rd['steps'][:at],success=r['success'],terminal_distance_m=r['final_distance'],error=r['error']))
  relative=None
  if at is not None:
   z=rd['contact_event']['measurement'];q=z['tool'][3:];R=Rotation.from_quat([*q[1:],q[0]]);relative=R.inv().apply(np.array(z['cube'][:3])-z['tool'][:3]).tolist()
  summary.append(dict(seed=key[0],grasp=key[1],contact_step=at,remaining=None if at is None else 100-at,cube_in_tool_frame=relative,nominal_success=ref['success'],any_success=any(r['success']for r in rs),mixed=len({r['success']for r in rs})>1,comparisons=checks))
 result[folder]=dict(episodes=len(rows),success=sum(r['success']for r in rows),contexts=summary,strict_prefix_checks=sum(z['same_prefix']for x in summary for z in x['comparisons']if z['action']!=[0,0]),rescued_contexts=sum(not x['nominal_success']and x['any_success']for x in summary))
a.out.write_text(json.dumps(result,indent=2));print({k:{x:v[x]for x in ['episodes','success','strict_prefix_checks','rescued_contexts']}for k,v in result.items()})
