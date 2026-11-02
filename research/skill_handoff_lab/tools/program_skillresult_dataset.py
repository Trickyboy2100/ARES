"""Build disjoint train/calibration manifests without copying private data."""
import argparse,json,shutil,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();manifest=[]
 for root in [a.old,a.new]:
  for path in sorted(root.rglob('TRAJECTORY.json.gz')):
   m=json.loads((path.parent/'MANIFEST.json').read_text());split='calibration' if m['seed'] in [743046,743047] else 'train';out=a.out/split/(root.name+'_'+path.parent.parent.name+'_'+path.parent.name);out.mkdir(parents=True)
   for name in ['TRAJECTORY.json.gz','MANIFEST.json']:shutil.copy2(path.parent/name,out/name)
   manifest.append(dict(split=split,seed=m['seed'],source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 (a.out/'SPLIT.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
