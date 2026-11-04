"""Conservative exact-program-branch compatibility, never relabel outcomes.
V4 differs only when a repeated target parameter interrupts a still-active path.
An entire V3 episode is excluded if it ever executes that changed branch.
"""
import argparse,json,gzip,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--extra',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
data=json.loads((a.source/'TRANSITIONS.json').read_text());labels=json.loads((a.source/'LABELS.json').read_text());rows=json.loads((a.source/'physics/RESULTS.json').read_text());accepted=[];excluded=[];target='Peg100_GPU_OBB_speed2_original_three_refines_v1__feedback_v4_persistent_start3'
for row in rows:
 assert row['controller']=='Peg100_GPU_OBB_speed2_original_three_refines_v1__feedback_v3_start3'
 assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
 changed=[x['step']for x in row['decisions']if x['stage']==1 and x['id']in (1,2,3,4,5) and round(x['obs'][-1]*7)==x['id'] and x['obs'][-3]>0]
 (excluded if changed else accepted).append(dict(path=row['path'],sha256=row['sha256'],changed_branch_steps=changed))
allowed={x['path']for x in accepted};adapted=[]
for t in data:
 if t['source']in allowed:
  t=dict(t,source_controller=t['controller'],controller=target,compatibility='V4 changed branch never executed; raw physical outcomes unmodified');adapted.append(t)
labels=[t for t in labels if t['source']in allowed]
if a.extra:
 extra=json.loads((a.extra/'TRANSITIONS.json').read_text());assert all(t['controller']==target for t in extra);adapted+=extra;labels+=json.loads((a.extra/'LABELS.json').read_text())
(a.out/'TRANSITIONS.json').write_text(json.dumps(adapted));(a.out/'LABELS.json').write_text(json.dumps(labels));(a.out/'COMPATIBILITY.json').write_text(json.dumps(dict(source=str(a.source),extra=str(a.extra),target=target,accepted=accepted,excluded=excluded,physical_new_queries=0,note='Logical branch equivalence; separate real replay checks must be reported as such, not all accepted episodes re-executed.'),indent=2));print('compatible episodes',len(accepted),'excluded',len(excluded),'transitions',len(adapted))
