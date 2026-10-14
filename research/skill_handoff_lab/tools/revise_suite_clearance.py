"""Pretraining rack clearance revision after actual contact/FCL probe."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
d=json.loads(a.source.read_text());d['parent_sha256']=hashlib.sha256(a.source.read_bytes()).hexdigest()
d['revision']='multicontract_v3_actual_wrist_clearance_pretraining'
d['revision_reason']='Correctly placed rack produced actual Link5/left-wall contact >100N and FCL <=0.27mm penetration; add 40mm lateral wrist clearance per side before training, keep all prior failures.'
d['horizon_feasibility_rule']='max(sampled horizon, travel/allowed_step * (1/60) + 0.6 s); nominal uses full allowed step'
for split in ['train','validation','test']:
    for g in d[split]:
        if g['family']=='T2':g['width']+=.08
a.output.write_text(json.dumps(d,indent=2))
