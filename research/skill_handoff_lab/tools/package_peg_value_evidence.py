"""Public-only archive with member SHA verification; exclude downloaded source licenses."""
import argparse,json,tarfile,hashlib,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
archives=[]
for folder in sorted(x for x in a.root.iterdir()if x.is_dir()and x.name not in ['figures','video_peg','video_peg_axial']):
 files=sorted(x for x in folder.rglob('*')if x.is_file()and (folder.name!='prior_code'or x.name.endswith('_PIN.json')));entries=[];arc=a.out/f'RAW_{folder.name}.tar.gz'
 with tarfile.open(arc,'w:gz')as t:
  for x in files:
   rel=str(x.relative_to(a.root));b=x.read_bytes();t.add(x,arcname=rel,recursive=False);entries.append(dict(path=rel,size=len(b),sha256=sha(b)))
 assert arc.stat().st_size<95_000_000
 archives.append(dict(archive=arc.name,size=arc.stat().st_size,sha256=sha(arc.read_bytes()),entries=entries))
arc=a.out/'RAW_metadata.tar.gz';entries=[]
with tarfile.open(arc,'w:gz')as t:
 for x in sorted(a.root.iterdir()):
  if x.is_file():b=x.read_bytes();t.add(x,arcname=x.name,recursive=False);entries.append(dict(path=x.name,size=len(b),sha256=sha(b)))
archives.append(dict(archive=arc.name,size=arc.stat().st_size,sha256=sha(arc.read_bytes()),entries=entries))
for name in ['figures','video_peg','video_peg_axial']:shutil.copytree(a.root/name,a.out/name,dirs_exist_ok=True)
for name in ['README.md','REVIEW.html','ANALYSIS.json','PRIOR_ART_COMPARISON.md','TARGET_SEMANTICS.json','RUN_SCOPE.json','VIDEO_QA.json','THIRD_PARTY_NOTICES.md','MANISKILL_APACHE_LICENSE.txt','BUFFER_AUDIT.json','ENVIRONMENT.json']:shutil.copy2(a.root/name,a.out/name)
for ar in archives:
 with tarfile.open(a.out/ar['archive'])as t:
  for e in ar['entries']:b=t.extractfile(e['path']).read();assert len(b)==e['size']and sha(b)==e['sha256']
assets=[dict(path=str(x.relative_to(a.out)),size=x.stat().st_size,sha256=sha(x.read_bytes()))for x in sorted(a.out.rglob('*'))if x.is_file()and x.name!='RAW_MANIFEST.json'and not x.name.startswith('RAW_')]
(a.out/'RAW_MANIFEST.json').write_text(json.dumps(dict(original_root=str(a.root),archives=archives,visible_assets=assets,scope='Public simulator evidence; downloaded prior-art source excluded, pin/hash indexes only',verification='All archive members decompressed and SHA-verified'),indent=2));print('Verified',sum(len(x['entries'])for x in archives),'members;',sum(x['size']for x in archives),'compressed bytes')
