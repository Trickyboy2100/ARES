"""Two bounded CPU processes for independent audits of completed GPU runs."""
import argparse,concurrent.futures,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--grid',type=Path,required=True);p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);a=p.parse_args()
directories=[d for d in a.grid.glob('multicontract_v*') if d.is_dir() and d.name.endswith(('_val','_test','_validation','_fullchain'))]
def audit(directory):
    argv=[sys.executable,str(Path(__file__).with_name('validate_process_fcl.py')),'--ares',str(a.ares),'--ares-r',str(a.ares_r),'--asset',str(a.asset),'--experiment',str(directory),'--self-check']
    result=subprocess.run(argv,capture_output=True,text=True,timeout=120)
    if result.returncode:raise RuntimeError(directory.name+': '+result.stderr)
    data=json.loads((directory/'independent_process_fcl.json').read_text());cases={r['case_id']:r for r in data['cases']}
    episodes=json.loads((directory/'frozen_evaluation.json').read_text())['episodes'];qualified=0;review=[]
    for episode in episodes:
        if not episode['success']:continue
        row=cases[episode['case_id']]
        if row['collision_samples'] or row.get('self_collision_samples') or row.get('soft_limit_violating_samples') or row.get('pose_violating_samples') or row.get('max_workspace_excess_m',0)>.001:review.append(episode['case_id'])
        else:qualified+=1
    return {'experiment':directory.name,'raw_successes':sum(r['success'] for r in episodes),'episodes':len(episodes),
        'independently_qualified_noncontact_successes':qualified,'success_cases_requiring_review_NOT_IMPUTED_FAILURE':review,
        'checked_physical_states':data['checked_states'],'collision_samples':sum(r['collision_samples'] for r in cases.values()),
        'self_collision_samples':sum(r.get('self_collision_samples',0) for r in cases.values()),
        'soft_limit_samples':sum(r.get('soft_limit_violating_samples',0) for r in cases.values()),
        'max_workspace_excess_m':max(r.get('max_workspace_excess_m',0) for r in cases.values()),
        'max_pose_error_rad':max(r.get('max_pose_error_rad',0) for r in cases.values()),
        'max_penetration_m':max(r['max_penetration_m'] for r in cases.values())}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(audit,sorted(directories)))
(a.grid/'independent_grid_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
