"""Assemble measured figures, compact raw records and models, not synthetic data.

Full 343MB trace archive remains on 190 and local runs; Git gets compact
episode-level raw data, 12 models, exact source snapshots and hash manifest.
"""
import argparse,json,hashlib,shutil,gzip
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace
grid=w/'runs/multicontract_20261014/grid_complete';repo=w/'repos/ARES/research/skill_handoff_lab/results/mentor_package';stage=repo/'03_PRELIMINARY_RESULTS/MULTICONTRACT';stage.mkdir(parents=True,exist_ok=True)
for path in (w/'runs/mentor_package/03_PRELIMINARY_RESULTS/MULTICONTRACT').iterdir():
    if path.is_file():shutil.copy2(path,stage/path.name)
compact=stage/'raw_compact';compact.mkdir(exist_ok=True)
names={'manifest.json','frozen_cases.json','frozen_evaluation.json','industrial_episodes.json','scene_specs.json',
    'actual_scene_bounds.json','controller_jacobian_audit.json','video_frame_records.json','independent_process_fcl.json','preplace_execution.json','ppo_physics.zip'}
for directory in sorted((grid/'r1_20261014').glob('multicontract_v*')):
    if not directory.is_dir():continue
    target=compact/directory.name;target.mkdir(exist_ok=True)
    for path in directory.iterdir():
        if path.name in names:shutil.copy2(path,target/path.name)
    if (directory/'ppo/progress.csv').exists():shutil.copy2(directory/'ppo/progress.csv',target/'training_progress.csv')
for name in ['multitask_v19.py','fullchain_v20.py','run_multitask_v17.py','run_fullchain_v20.py','batch_multicontract_v19.py',
    'final_frozen_test_v19.py','extra_physics_checks_v20.py','multicontract_final_export_manifest.json','export_integrity.json']:
    shutil.copy2(grid/name,compact/name)
for name in ['independent_grid_audit.json']:
    if (grid/'r1_20261014'/name).exists():shutil.copy2(grid/'r1_20261014'/name,stage/name)
for source in [w/'runs/multicontract_20261014/oracle_prefix',w/'runs/multicontract_20261014/offline_prefix_mesh_qualification_v2',w/'runs/multicontract_20261014/directional_preflight_v18']:
    target=compact/source.name;target.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix=='.json':shutil.copy2(path,target/path.name)
for source in sorted((w/'runs/multicontract_20261014').glob('strict_guard_v21*'))+[w/'runs/multicontract_20261014/fullchain_capture_v22']:
    if source.is_dir():
        target=compact/source.name;target.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.name in names:shutil.copy2(path,target/path.name)
    elif source.suffix=='.py':shutil.copy2(source,compact/source.name)
for source in (w/'runs/multicontract_20261014').glob('fullchain_capture_v22.py'):shutil.copy2(source,compact/source.name)
for path in (w/'runs/multicontract_20261014/asset_mesh_colliders_v5').glob('*.json'):shutil.copy2(path,compact/path.name)
files=[{'path':str(path.relative_to(stage)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    for path in stage.rglob('*') if path.is_file() and path.name!='data_manifest.json']
manifest={'base_commit':'3c20dcd','host':'190','full_trace_archive':str(w/'runs/multicontract_20261014/multicontract_final_v19_v20.tar.gz'),
    'full_trace_archive_sha256':'a855d39ccc875d2889aece79f2b93c65f4798aeb3c249e67c7ac67c1e1c8996d',
    'full_trace_archive_in_git':False,'full_trace_archive_preserved_on_190_and_local':True,
    'scope':'independent held right arm; welded source-AABB tray; no contact grasp/full robot safety certification','files':files}
(stage/'data_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'compact_files':len(files),'bytes':sum(x['bytes'] for x in files)}))
