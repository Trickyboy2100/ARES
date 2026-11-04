"""Stage public native-simulation evidence, preserving all local checkpoints."""
import argparse,json,hashlib,shutil,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);inventory={}
for f in sorted(a.root.rglob('*')):
 if not f.is_file() or '__pycache__'in f.parts or f.name=='LOCK':continue
 rel=f.relative_to(a.root);inventory[str(rel)]=dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
 keep=not f.name.startswith('RESUME_')
 if f.name.startswith('RESUME_'):
  c=f.parent/'CURRENT.json';keep=c.exists()and json.loads(c.read_text())['checkpoint']==f.name
 if keep:
  dst=a.out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
(a.out/'LOCAL_INVENTORY_SHA.json').write_text(json.dumps(inventory,indent=2))
print('All local files indexed:',len(inventory),'staged:',sum(p.is_file()for p in a.out.rglob('*')))
