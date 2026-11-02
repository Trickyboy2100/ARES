"""Audit model-predicted native path counts against observed task execution."""
import argparse,gzip,json
from pathlib import Path
from collections import Counter
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
 for path in sorted(a.root.rglob('TRAJECTORY.json.gz')):
  if 'datasets' in path.parts:continue
  d=json.load(gzip.open(path,'rt'));r=d['summary']
  search=next((x for x in d['plans'] if x.get('phase')=='SEARCH'),None)
  if not search:continue
  selected=search['selected'];cs=[c for c in search['candidates'] if c['valid'] and abs(c['angle']-selected[0])<1e-6 and abs(c['hook_y']-selected[1])<1e-6]
  if not cs:continue
  predicted=min(c['predicted_steps'] for c in cs);remaining=100-search['step'];used=sum(s['phase'] in ['TRANSFER','HOOK','PULL'] for s in d['steps']);finished=any(s['phase']=='PULL' for s in d['steps']) and not (r['error'] and 'HORIZON' in r['error'] and d['steps'][-1]['phase'] in ['TRANSFER','HOOK','PULL'])
  rows.append(dict(path=str(path),group=path.relative_to(a.root).parts[0],predicted=predicted,remaining=remaining,observed_primary_steps=used,complete_primary=finished,task_success=r['task_success'],phase_at_horizon=d['steps'][-1]['phase']))
 report={}
 for group in sorted(set(r['group'] for r in rows)):
  rs=[r for r in rows if r['group']==group];feasible=[r for r in rs if r['predicted']<=r['remaining']];completed=[r for r in rs if r['complete_primary']]
  report[group]=dict(cases=len(rs),predicted_within_budget=len(feasible),actual_success_among_predicted_feasible=sum(r['task_success'] for r in feasible),primary_not_completed_among_predicted_feasible=sum(not r['complete_primary'] for r in feasible),median_primary_count_error=None if not completed else float(np.median([r['observed_primary_steps']-r['predicted'] for r in completed])))
 a.out.write_text(json.dumps(dict(summary=report,rows=rows,scope='Predictor omits target feedback and physical attachment drift. Counts are not hard feasibility certificates; only observed completed primary paths have uncensored errors.'),indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
