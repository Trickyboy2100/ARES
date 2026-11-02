"""Verify public evidence and restore originals or a clearly marked portable copy.
Physical gzip trajectories and model weights are never rewritten.
"""
import argparse,hashlib,json,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path);p.add_argument('--portable',action='store_true');a=p.parse_args();m=json.loads((a.package/'RAW_MANIFEST.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
if a.portable and a.out is None:p.error('--portable requires --out')
if a.out:a.out.mkdir(parents=True,exist_ok=False)
def destination(name):
 rel=Path(name);assert not rel.is_absolute()and '..'not in rel.parts
 out=a.out/rel;out.parent.mkdir(parents=True,exist_ok=True);return out
count=0
for arc in m['archives']:
 path=a.package/arc['archive'];assert sha(path.read_bytes())==arc['sha256']
 with tarfile.open(path,'r:gz')as t:
  for e in arc['entries']:
   member=t.getmember(e['path']);assert member.isfile();b=t.extractfile(member).read();assert len(b)==e['size']and sha(b)==e['sha256'];count+=1
   if a.out:destination(e['path']).write_bytes(b)
for e in m['visible_assets']:
 b=(a.package/e['path']).read_bytes();assert sha(b)==e['sha256']
 if a.out:destination(e['path']).write_bytes(b)
if a.out:shutil.copy2(a.package/'RAW_MANIFEST.json',a.out/'RAW_MANIFEST.json')
changed=[]
if a.portable:
 old=m['original_root'];new=str(a.out.resolve())
 for f in list(a.out.rglob('*.json')):
  if f.name=='RAW_MANIFEST.json':continue
  source=f.read_text()
  if old not in source:continue
  obj=json.loads(source)
  def replace(v):
   if isinstance(v,str):return v.replace(old,new)
   if isinstance(v,list):return[replace(x)for x in v]
   if isinstance(v,dict):return{k:replace(x)for k,x in v.items()}
   return v
  original=a.out/'ORIGINAL_METADATA'/f.relative_to(a.out);original.parent.mkdir(parents=True,exist_ok=True);original.write_text(source);f.write_text(json.dumps(replace(obj),indent=2));changed.append(dict(path=str(f.relative_to(a.out)),original_sha256=sha(source.encode()),derived_sha256=sha(f.read_bytes())))
 (a.out/'PORTABLE_COPY.json').write_text(json.dumps(dict(scope='Derived metadata path relocation only; original JSON preserved in ORIGINAL_METADATA; all gzipped physics and weights unchanged',original_root=old,relocated_root=new,changed=changed),indent=2))
print(dict(status='PASS',verified_raw_entries=count,visible_assets=len(m['visible_assets']),restored=None if a.out is None else str(a.out),portable_metadata_files=len(changed)))
