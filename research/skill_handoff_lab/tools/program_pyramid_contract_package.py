"""Package public synthetic-physics evidence with per-file archive verification."""
import argparse,json,tarfile,hashlib,shutil,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 for name in ['RAW_TOOL_AUDIT.json','RAW_PYRAMID_AUDIT.json']:
  shutil.copy2(a.root/name,a.out/name)
 shutil.copy2(a.root/'history_audit/OPPORTUNITY.json',a.out/'HISTORICAL_OPPORTUNITY.json')
 files=[f for folder in ['pilot','symmetry','feasible'] for f in (a.root/folder).rglob('*') if f.is_file()]+[a.root/name for name in ['PILOT.log','SYMMETRY.log','FEASIBLE.log','ANALYSIS.log']]
 manifest={str(f.relative_to(a.root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 target=a.out/'physics.tar.gz'
 with tarfile.open(target,'w:gz') as tar:
  for f in files:tar.add(f,arcname=str(f.relative_to(a.root)))
 with tarfile.open(target,'r:gz') as tar:
  for name,digest in manifest.items():assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest
 (a.out/'PHYSICS_FILES.json').write_text(json.dumps(manifest,indent=2))
 (a.out/'PROVENANCE.json').write_text(json.dumps(dict(base_commit='b9f5713a9d3f48d3149f0a9bb7eba11d6dd6af86',working_source='per-run source snapshots in archive; subsequent probes do not overwrite earlier code',archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),archive_bytes=target.stat().st_size,verified_files=len(files),task='ARES-PyramidOrientation-adaptation',native_task='StackPyramid-v1',native_horizon=250,restricted_field_data=False,hardware_commands=False,optimizer_updates=0,maniskill_revision='62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3',limitation='Inherited stride3 execution is not certified native timing or hardware dynamics. No claimed physical contact-force measurement, sealed test, RL novelty or simulator benchmark reproduction.'),indent=2))
 print('Verified archive',len(files),target.stat().st_size)
if __name__=='__main__':main()
