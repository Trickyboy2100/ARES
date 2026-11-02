"""Lossless simulator-only evidence packaging; verify every stored entry."""
import argparse,json,hashlib,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();visible=['analysis','figures','video_rescue','video_regression','README.md','REVIEW.html','THIRD_PARTY_SB3_LICENSE.txt'];assets=[]
for name in visible:
 src=a.root/name;dst=a.out/name
 if src.is_dir():shutil.copytree(src,dst)
 else:shutil.copy2(src,dst)
for f in sorted(a.out.rglob('*')):
 if f.is_file():assets.append(dict(path=str(f.relative_to(a.out)),size=f.stat().st_size,sha256=sha(f)))
files=[f for f in sorted(a.root.rglob('*'))if f.is_file()and f.relative_to(a.root).parts[0]not in visible];chunks=[];batch=[];size=0
for f in files:
 if size+f.stat().st_size>32*1024**2 and batch:chunks.append(batch);batch=[];size=0
 batch.append(f);size+=f.stat().st_size
if batch:chunks.append(batch)
archives=[]
for i,batch in enumerate(chunks):
 path=a.out/f'RAW_{i:02d}.tar.gz';entries=[]
 with tarfile.open(path,'w:gz')as t:
  for f in batch:
   rel=str(f.relative_to(a.root));assert not f.is_symlink();t.add(f,arcname=rel,recursive=False);entries.append(dict(path=rel,size=f.stat().st_size,sha256=sha(f)))
 with tarfile.open(path,'r:gz')as t:
  for e in entries:assert hashlib.sha256(t.extractfile(e['path']).read()).hexdigest()==e['sha256']
 archives.append(dict(archive=path.name,size=path.stat().st_size,sha256=sha(path),entries=entries))
(a.out/'RAW_MANIFEST.json').write_text(json.dumps(dict(original_root=str(a.root.resolve()),scope='Public native ManiSkill physics and derived research only; no site/hardware data',archives=archives,visible_assets=assets,all_entry_sha_verified=True,source_provenance='Per-run start snapshots where present; near wrapper/dependency copy made after launch, not claimed as start-time attestation; raw scripts/logs and all negative versions retained'),indent=2));print(dict(raw_files=len(files),archives=len(archives),archive_bytes=sum(x['size']for x in archives),visible_files=len(assets)))
