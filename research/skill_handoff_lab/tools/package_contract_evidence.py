"""Lossless public-simulator evidence archive, bounded Git blobs and entry SHA verification."""
import argparse,json,hashlib,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['analysis','figures','video_stroke']:shutil.copytree(a.root/name,a.out/name,dirs_exist_ok=True)
files=[f for f in sorted(a.root.rglob('*'))if f.is_file() and f.relative_to(a.root).parts[0]not in ['analysis','figures','video_stroke']];manifest=[];chunks=[];batch=[];size=0
for f in files:
 if size+f.stat().st_size>32*1024**2 and batch:chunks.append(batch);batch=[];size=0
 batch.append(f);size+=f.stat().st_size
if batch:chunks.append(batch)
for i,batch in enumerate(chunks):
 name=f'RAW_{i:02d}.tar.gz';archive=a.out/name;entries=[]
 with tarfile.open(archive,'w:gz')as t:
  for f in batch:
   rel=str(f.relative_to(a.root));t.add(f,arcname=rel,recursive=False);entries.append(dict(path=rel,size=f.stat().st_size,sha256=sha(f)))
 with tarfile.open(archive,'r:gz')as t:
  for e in entries:assert hashlib.sha256(t.extractfile(e['path']).read()).hexdigest()==e['sha256']
 manifest.append(dict(archive=name,size=archive.stat().st_size,sha256=sha(archive),entries=entries))
(a.out/'RAW_MANIFEST.json').write_text(json.dumps(dict(original_root=str(a.root),archives=manifest,source_scope='Public ManiSkill physical simulation only; no site data',source_copy_note='Training/grid source snapshots copied at start; pilot/effect wrapper and evaluation entry copied at end. Later unused optional flags exist in final source; do not infer exact earlier dependency hashes where not captured. All raw scripts/logs preserved.'),indent=2))
print(json.dumps(dict(files=len(files),archives=len(chunks),archive_bytes=sum(m['size']for m in manifest))))
