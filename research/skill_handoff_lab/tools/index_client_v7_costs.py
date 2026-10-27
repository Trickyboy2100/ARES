"""Account actual simulation/training work once per completed own synthetic run."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
rows=[]
for path in sorted(root.rglob('manifest.json')):
    if not (path.parent/'RUN_COMPLETE.json').exists():continue
    m=json.loads(path.read_text());formal='formal' in path.relative_to(root).parts
    rows.append({'run':str(path.parent),'cohort':'formal' if formal else 'software_diagnostic_or_physical_smoke',
        'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'online_training_interactions':m['online_training_interactions'],
        'training_env_physics_steps':m['training_physics_steps']*m['num_envs'],
        'all_env_physics_steps':m['env_physics_steps'],
        'evaluation_env_control_steps_including_padding':(m['env_physics_steps']-m['training_physics_steps']*m['num_envs'])//2,
        'completed_training_episodes':m['training_episodes'],'completed_evaluation_episodes':m['test_episodes'],
        'measured_run_wall_s_excluding_app_setup':m['wall_s']})
keys=['online_training_interactions','training_env_physics_steps','all_env_physics_steps',
      'evaluation_env_control_steps_including_padding','completed_training_episodes','completed_evaluation_episodes']
totals={key:sum(r[key] for r in rows) for key in keys}
report={'scope':'This turn own SYNTHETIC runs only; historical901120 excluded; evaluation counts are workload, not IID sceneN',
    'totals':totals,'run_count':len(rows),'runs':rows,
    'caution':'physics counters cover instrumented120Hz loop, not app warm-up; eval padding slots are real physical workload but not additional named testcases'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps({'run_count':len(rows),'totals':totals},indent=2))
