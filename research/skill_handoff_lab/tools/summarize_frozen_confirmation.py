"""Paired frozen outcomes, cohorts separate; adaptive extension disclosed."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();names=['measured_31601','measured_31602','remaining_target_31601','remaining_target_31602','measured_frozen_rule','remaining_target_frozen_rule','measured_nominal'];cohorts={};pooled={n:[]for n in names}
for suffix in ['', '_batch2']:
 data={}
 for n in names:
  path=a.root/f'confirm_{n}{suffix}';rows=json.loads((path/'RESULTS.json').read_text());assert len(rows)==16;assert len({r['seed']for r in rows})==16;data[n]=rows;pooled[n]+=rows
 baseline={r['seed']:r['success']for r in data['measured_nominal']}
 cohorts[suffix or 'batch1']={n:dict(success=sum(r['success']for r in rows),n=len(rows),rescue=[r['seed']for r in rows if r['success'] and not baseline[r['seed']]],regression=[r['seed']for r in rows if not r['success'] and baseline[r['seed']]])for n,rows in data.items()}
for n in names:
 x=json.loads((a.root/f'confirm_{n}/MANIFEST.json').read_text());y=json.loads((a.root/f'confirm_{n}_batch2/MANIFEST.json').read_text())
 for key in ['checkpoint_sha','runtime_sha','frame','control','reference_sha']:assert x[key]==y[key],(n,key)
 assert not set(x['seeds'])&set(y['seeds'])
base={r['seed']:r['success']for r in pooled['measured_nominal']};assert len(base)==32
out={}
for n,rows in pooled.items():
 assert {r['seed']for r in rows}==base.keys()
 out[n]=dict(success=sum(r['success']for r in rows),n=32,rescue=[r['seed']for r in rows if r['success']and not base[r['seed']]],regression=[r['seed']for r in rows if not r['success']and base[r['seed']]],planning_queries=sum(len(r['queries'])for r in rows),physical_steps=sum(r['steps']for r in rows),front_counts={str(g):sum(r['decisions'][0]['id']==g for r in rows)for g in range(3)})
comparisons={}
for seed in [31601,31602]:
 old={r['seed']:r['success']for r in pooled[f'measured_{seed}']};new={r['seed']:r['success']for r in pooled[f'remaining_target_{seed}']};comparisons[str(seed)]=dict(new_only=[s for s in old if new[s]and not old[s]],old_only=[s for s in old if old[s]and not new[s]])
s=dict(cohorts=cohorts,pooled=out,interface_pairs=comparisons,scope='32 new natural initial states; second cohort adaptively added after first; frozen models/rules unchanged; descriptive, not sealed or fixed-sample significance evidence; seeds share inherited neural weights')
(a.root/'CONFIRMATION_SUMMARY.json').write_text(json.dumps(s,indent=2));print(json.dumps(out,indent=2))
