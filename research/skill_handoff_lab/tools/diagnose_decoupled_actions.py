"""Descriptive executed-policy diagnostics; no causal success attribution."""
import json,collections,gzip
from pathlib import Path
root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004');out={}
for folder in ['online_integrity_20261105/online_sac','online_integrity_20261105/online_success_bc','decoupled_interface_20261105/relative_sac64','decoupled_interface_20261105/relative_bc64']:
 h=json.loads((root/folder/'HISTORY.json').read_text());rows=[v['row']for v in h if v['job']['kind']=='eval' and v['job']['checkpoint']==64];cases=[]
 for r in rows:
  d=json.load(gzip.open(r['path'],'rt'));rear=[v for v in d['summary']['decisions']if v['stage']==1]
  cases.append(dict(seed=r['seed'],success=r['success'],grasp=r['decisions'][0]['id'],action_counts=dict(collections.Counter(v['id']for v in rear)),replans=sum(v.get('replanned',False)for v in rear),failed_replans=sum(v.get('plan_failed',False)for v in rear),requested_deltas_mm=[v['requested_target_delta_m']*1000 for v in rear if v.get('replanned') and v['id']in [2,3,4,5]],plans=len(r['queries']),source=r['sha256']))
 out[folder.split('/')[-1]]=cases
(root/'decoupled_interface_20261105/ACTION_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2))
for n,cs in out.items():print(n,dict(success=sum(c['success']for c in cs),replans=sum(c['replans']for c in cs),failed=sum(c['failed_replans']for c in cs),grasp=dict(collections.Counter(c['grasp']for c in cs))))
