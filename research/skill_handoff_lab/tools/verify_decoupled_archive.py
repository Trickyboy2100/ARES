"""Verify every archive/member; optional fresh-directory restoration, no overwrites."""
import argparse,json,hashlib,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args();m=json.loads((a.package/'RAW_MANIFEST.json').read_text());n=0
if a.out:a.out.mkdir(parents=True,exist_ok=False)
for arc in m['archives']:
 path=a.package/arc['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==arc['sha256']
 with tarfile.open(path,'r:gz')as t:
  assert {v.name for v in t.getmembers()}==arc['members'].keys()
  for v in t.getmembers():
   rel=Path(v.name);assert v.isfile()and not rel.is_absolute()and '..'not in rel.parts;b=t.extractfile(v).read();assert hashlib.sha256(b).hexdigest()==arc['members'][v.name];n+=1
   if a.out:
    dest=a.out/rel;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();dest.write_bytes(b)
print('VERIFIED',len(m['archives']),'archives',n,'members; raw bytes unchanged')
