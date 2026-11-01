"""Package source snapshots and complete saved physical traces, CPU only."""
import argparse,hashlib,json,tarfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 files=[]
 for results in sorted(a.root.glob('*/RESULTS.json')):
  rows=json.loads(results.read_text())
  for row in rows:
   trace=results.parent/Path(row['path']).name
   if hashlib.sha256(trace.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(f'SHA mismatch: {trace}')
  files.extend(sorted(x for x in results.parent.iterdir() if x.is_file()))
 archive=a.out/'RAW_EVIDENCE.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for f in files:tar.add(f,arcname=str(f.relative_to(a.root)))
 manifest=dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,source_root=str(a.root),files=[dict(path=str(f.relative_to(a.root)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],scope='complete raw physics traces plus exact per-run source snapshots; no physical hardware data')
 (a.out/'RAW_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
if __name__=='__main__':main()
