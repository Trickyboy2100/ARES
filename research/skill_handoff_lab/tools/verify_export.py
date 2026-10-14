"""Verify every original exported file before adding derived audit results."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
manifest=json.loads((a.directory/'multicontract_final_export_manifest.json').read_text());failures=[]
for row in manifest['files']:
    path=a.directory/row['path']
    if not path.is_file() or path.stat().st_size!=row['size'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:failures.append(row['path'])
result={'checked_files':len(manifest['files']),'failures':failures,'original_file_integrity':'PASS' if not failures else 'FAIL'}
(a.directory/'export_integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(bool(failures))
