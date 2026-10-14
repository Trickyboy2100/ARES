"""Freeze rack bay from actual USD wrist hull requirement, before training."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
d=json.loads(a.source.read_text());d['parent_sha256']=hashlib.sha256(a.source.read_bytes()).hexdigest()
d['revision']='multicontract_v4_USD_wrist_envelope_pretraining'
d['revision_reason']='Actual USD wrist hull at held T2 terminal extends to BODY X=0.120958m, tray center X=0.45475m; minimum rack width including 20mm side clearance is 0.707584m. Earlier 0.50/0.58m bays cause verified wrist contacts. Preserve failures and use >=0.73m research shelf bays.'
widths={'T2_rack_y':.74,'T2_rack_xyz':.76,'V_T2_285':.73,'X_T2_292':.75}
for split in ['train','validation','test']:
    for g in d[split]:
        if g['family']=='T2':g['width']=widths[g['id']]
a.output.write_text(json.dumps(d,indent=2))
