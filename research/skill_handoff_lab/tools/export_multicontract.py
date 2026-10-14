"""Export only scoped completed research artifacts, with per-file hashes."""
import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run=a.root/'runs';out=run/'r1_20261014'
archive=run/'multicontract_final_v19_v20.tar.gz'
if archive.exists():raise FileExistsError(archive)
names=[f'multicontract_v19_{m}_seed{s}{suffix}' for m in ['ppo','residual','contract_blind','contract'] for s in [31,47,59] for suffix in ['','_val','_test']]
names+=['multicontract_v20_nominal_validation','multicontract_v20_nominal_test','multicontract_v20_fullchain']
files=[]
for name in names:
    directory=out/name
    if not (directory/'manifest.json').exists():raise FileNotFoundError(directory)
    proc=out/(name+'_process.json');status=json.loads(proc.read_text())
    if status['returncode'] or status['timeout']:raise ValueError(f'Invalid completed run {name}')
    files.extend(x for x in directory.rglob('*') if x.is_file());files+=[proc,out/(name+'.log')]
for name in ['multitask_v19.py','fullchain_v20.py','run_multitask_v17.py','run_fullchain_v20.py',
    'batch_multicontract_v19.py','final_frozen_test_v19.py','extra_physics_checks_v20.py','multicontract_v5.json',
    'multicontract_v19_gpu0_queue.json','multicontract_v19_gpu1_queue.json',
    'multicontract_v19_gpu0_test_queue.json','multicontract_v19_gpu1_test_queue.json','multicontract_v20_extra_queue.json']:
    files.append(run/name)
manifest={'base_commit':'3c20dcd','host':'190','schema':1,'completed_experiments':names,
    'files':[{'path':str(x.relative_to(run)),'size':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in files]}
manifest_path=run/'multicontract_final_export_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
with tarfile.open(archive,'w:gz') as tar:
    for file in files+[manifest_path]:tar.add(file,arcname=str(file.relative_to(run)))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'experiments':len(names)}))
