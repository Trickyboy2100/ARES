"""Public simulation evidence only: source snapshots, models, real traces, SHA verification."""
import argparse,json,hashlib,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--exclude',action='append',default=[]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
files=[p for p in sorted(a.root.rglob('*'))if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and not any(x in p.relative_to(a.root).parts for x in a.exclude)];groups=[];group=[];size=0
for path in files:
 if group and size+path.stat().st_size>40_000_000:groups.append(group);group=[];size=0
 group.append(path);size+=path.stat().st_size
audit=[]
if group:groups.append(group)
for i,group in enumerate(groups):
 dest=a.out/f'RAW_{i:03}.tar.gz';assert not dest.exists(),dest
 members={}
 with tarfile.open(dest,'w:gz',compresslevel=5)as t:
  for path in group:
   rel=str(path.relative_to(a.root));members[rel]=hashlib.sha256(path.read_bytes()).hexdigest();t.add(path,arcname=rel,recursive=False)
 with tarfile.open(dest,'r:gz')as t:
  for member in t.getmembers():assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==members[member.name]
 audit.append(dict(file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size,members=members));print(dest.name,len(members),flush=True)
(a.out/'RAW_MANIFEST.json').write_text(json.dumps(dict(source=str(a.root),verified=True,excluded=a.exclude,archives=audit),indent=2))
