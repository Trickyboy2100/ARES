"""Package closed-loop simulation evidence, including negative and interrupted runs."""
import argparse,json,tarfile,hashlib,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
sha=lambda b:hashlib.sha256(b).hexdigest();archives=[]
files=[f for f in sorted(a.root.rglob('*'))if f.is_file()and not any(x in f.relative_to(a.root).parts for x in ['prior','prior_code','__pycache__','figures','video_regression'])]
def flush(chunk):
 path=a.out/f'RAW_{len(archives):03d}.tar.gz';entries=[]
 with tarfile.open(path,'w:gz')as t:
  for f in chunk:
   b=f.read_bytes();name=str(f.relative_to(a.root));t.add(f,arcname=name,recursive=False);entries.append(dict(path=name,size=len(b),sha256=sha(b)))
 assert path.stat().st_size<95_000_000
 with tarfile.open(path)as t:
  for e in entries:assert sha(t.extractfile(e['path']).read())==e['sha256']
 archives.append(dict(archive=path.name,size=path.stat().st_size,sha256=sha(path.read_bytes()),entries=entries))
chunk=[];size=0
for f in files:
 if chunk and size+f.stat().st_size>80_000_000:flush(chunk);chunk=[];size=0
 chunk.append(f);size+=f.stat().st_size
if chunk:flush(chunk)
for folder in ['figures','video_regression']:shutil.copytree(a.root/folder,a.out/folder)
for name in ['README.md','REVIEW.html','RESULTS_SUMMARY.json','ZERO_SUFFIX_ABLATION.json','PARENT_CALIBRATION.json','DEPLOYMENT_CALIBRATION.json','INTERRUPTIONS.json','FINAL_SOURCE_MANIFEST.json','PRIOR_ART.md','PRIOR_CODE.json','ENVIRONMENT.json','MANISKILL_APACHE_LICENSE.txt','THIRD_PARTY_NOTICES.md','REPRODUCE.sh']:
 if (a.root/name).exists():shutil.copy2(a.root/name,a.out/name)
assets=[dict(path=str(f.relative_to(a.out)),size=f.stat().st_size,sha256=sha(f.read_bytes()))for f in sorted(a.out.rglob('*'))if f.is_file()and not f.name.startswith('RAW_')]
(a.out/'RAW_MANIFEST.json').write_text(json.dumps(dict(original_root=str(a.root),archives=archives,visible_assets=assets,scope='Public ManiSkill simulation only; all run-specific trajectories, models and source snapshots included; downloaded foreign repositories excluded',verification='Every archive member decompressed and SHA verified'),indent=2));print(dict(raw_members=sum(len(x['entries'])for x in archives),compressed_bytes=sum(x['size']for x in archives),visible_assets=len(assets)))
