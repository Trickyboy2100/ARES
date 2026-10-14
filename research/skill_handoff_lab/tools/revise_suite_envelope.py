"""Freeze a new geometry revision from measured USD-hull wrist interference.

Never overwrites v1. This is model/geometry correction before any training,
not seed selection or changing a scored test result.
"""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
d=json.loads(a.source.read_text());d['parent_sha256']=hashlib.sha256(a.source.read_bytes()).hexdigest()
d['revision']='multicontract_v2_wrist_envelope_pretraining'
d['revision_reason']='Independent actual-USD FCL found wrist/side-wall overlap in T2, so research rack bay includes wrist clearance; limits/pose/success unchanged.'
for split in ['train','validation','test']:
    for g in d[split]:
        if g['family']=='T2':g['width']+=.20
a.output.write_text(json.dumps(d,indent=2));print(d['revision'])
