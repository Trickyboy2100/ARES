"""Package completed public simulation evidence with per-member SHA checks."""
import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
assert not list(a.out.glob('RAW_*.tar.gz')),'Never overwrite an evidence release'
files=[f for f in sorted(a.root.rglob('*'))if f.is_file()and '__pycache__'not in f.parts and not f.name.endswith('.tmp') and f.name!='TASKBOOK.md']
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();groups=[];group=[];size=0
for f in files:
 if group and size+f.stat().st_size>32*1024*1024:groups.append(group);group=[];size=0
 group.append(f);size+=f.stat().st_size
if group:groups.append(group)
archives=[]
for i,fs in enumerate(groups):
 path=a.out/f'RAW_{i:03d}.tar.gz';members=[]
 with tarfile.open(path,'w:gz')as tar:
  for f in fs:
   rel=str(f.relative_to(a.root));members.append(dict(path=rel,bytes=f.stat().st_size,sha256=sha(f)));tar.add(f,arcname=rel)
 with tarfile.open(path)as tar:
  for m in members:assert hashlib.sha256(tar.extractfile(m['path']).read()).hexdigest()==m['sha256']
 archives.append(dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),members=members));print(path.name,path.stat().st_size,len(members),flush=True)
(a.out/'RAW_MANIFEST.json').write_text(json.dumps(dict(root=str(a.root),source='public native simulation only; no sealed or restricted site data',archives=archives,verified_all_members=True),indent=2))
print('VERIFIED',len(archives),len(files),sum(v['bytes']for v in archives))
