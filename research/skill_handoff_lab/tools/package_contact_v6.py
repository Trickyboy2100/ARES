"""Publish an allowlisted SYNTHETIC task subset; never traverse site data.

Full raw archive stays outside Git. Gzip preserves original bytes; the release
manifest distinguishes compressed hashes from full archive original hashes.
"""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root,dest):
    if root.name!='contact_handoff_20261016_r1':raise ValueError('explicit synthetic archive only')
    dest.mkdir(parents=True,exist_ok=True);index=[]
    # Exact runtime sources are deduplicated by their BYTE hash, not Git HEAD.
    snapshots={};runs=[]
    selected={'v6b_residual_seed31_validation','v6b_recover_seed31_validation','v6b_recover_force_resume31'}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith(('v6b_','handoff_v')):continue
        if not (d/'manifest.json').exists():
            runs.append({'run':d.name,'status':'SOFTWARE_INCOMPLETE_NO_MANIFEST'});continue
        m=json.loads((d/'manifest.json').read_text())
        runs.append({'run':d.name,'complete':(d/'RUN_COMPLETE.json').exists(),'training_interactions':m['online_training_interactions'],
                     'source_sha256':m['task_source_sha256'],'raw_successes':m['test_successes']})
        for p in sorted(d.rglob('*')):
            if not p.is_file():continue
            rel=p.relative_to(root);target=dest/rel
            if p.name in ['task_source.py','boundary_credit_source.py','hybrid_policy_source.py']:
                h=sha(p);out=dest/'source_snapshots'/f'{h}.py';out.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,out);snapshots[str(rel)]=str(out.relative_to(dest));continue
            allowed=(p.suffix=='.json' and p.name!='action_log.json') or p.suffix=='.zip' or p.suffix=='.csv' or p.suffix=='.tfevents'
            if p.name=='action_log.json' and d.name in selected:allowed=True
            if p.name in ['actual_physics_trace.npz'] and d.name in selected:allowed=True
            if p.name in ['actual_physics.gif','physical_scene.png','post_step_contact_history.npz']:allowed=True
            if p.name=='exception.txt':allowed=True
            if not allowed:continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.json' and p.name not in ['manifest.json','run_config.json','RUN_COMPLETE.json']:
                out=target.with_suffix(target.suffix+'.gz');out.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
            else:out=target;shutil.copyfile(p,out)
            index.append({'original':str(rel),'original_bytes':p.stat().st_size,'original_sha256':sha(p),
                          'published':str(out.relative_to(dest)),'published_sha256':sha(out)})
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in ['.json','.png']:
            out=dest/p.name
            if p.suffix=='.json':out=out.with_suffix('.json.gz');out.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
            else:shutil.copyfile(p,out)
            index.append({'original':p.name,'original_sha256':sha(p),'published':out.name,'published_sha256':sha(out)})
    meta=root/'asset_narrow/asset_manifest.json';shutil.copyfile(meta,dest/'asset_manifest.json')
    (dest/'RELEASE_INDEX.json').write_text(json.dumps({'scope':'ARES SYNTHETIC PHYSX TASKS ONLY; no site scans/configs/readbacks',
        'full_archive':str(root),'runs':runs,'files':index,'exact_source_snapshot_mapping':snapshots},indent=2))
    files=[{'path':str(p.relative_to(dest)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json']
    (dest/'SHA256_MANIFEST.json').write_text(json.dumps(files,indent=2))
    print(json.dumps({'runs':len(runs),'files':len(files),'release_bytes':sum(x['bytes'] for x in files)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.dest.resolve())
