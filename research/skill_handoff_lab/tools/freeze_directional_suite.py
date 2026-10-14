"""Directional T2 with a real legal yaw-aligned Preplace configuration.

Preserves failed side-entry versions; no BODY calibration or limits change.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from research.skill_handoff_lab.bench.model import PinnedRobotModel
p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
m=PinnedRobotModel(a.ares,a.ares_r);old=json.loads((a.asset/'asset_manifest.json').read_text())['start_q']
normal=np.array([2**-.5,-2**-.5,0]);center=np.array([.45475,-.48,1.45])+.04*normal
tcp=center-.03475*normal
answers=m.solve(tcp,-np.pi/4,0,[old],seed_value=31,attempts=4)
if not answers:raise ValueError('No legal yaw-aligned Preplace')
start=min(answers,key=lambda x:np.linalg.norm(x[1]-old))[1]
d=json.loads(a.source.read_text());d['parent_sha256']=hashlib.sha256(a.source.read_bytes()).hexdigest()
d['revision']='multicontract_v5_yaw_aligned_directional_pretraining'
d['revision_reason']='90deg lateral entry with yaw=0 makes the forearm enter the rack. T2 uses genuinely yaw-aligned -45deg Preplace and -45deg slot/rack geometry; no installation transform or real joint limits changed.'
widths={'train':[.24,.26],'validation':[.25],'test':[.23]}
for split in ['train','validation','test']:
    index=0
    for g in d[split]:
        if g['family']!='T2':continue
        g['yaw_deg']=-45;g['tool_yaw_deg']=-45;g['width']=widths[split][index];index+=1
        g['height']=.20;g['depth']=.17;g['travel']=.11;g['mouth_offset']=.05;g['success_depth']=.055
        g['center_body_m']=center.tolist();g['start_q']=start.tolist()
        g['mask']=[1,1,0] if index==1 else [1,1,1]
        goals=m.solve(tcp+.11*normal,-np.pi/4,0,[start],seed_value=31,attempts=4)
        if not goals:raise ValueError('T2 endpoint has no legal IK')
d['preplace_evidence']={'seed':31,'attempts':4,'start_q':start.tolist(),'fk_tcp':m.fk(start).tolist(),
    'scope':'legal IK/soft limits, not yet real cuRobo transition proof'}
a.output.write_text(json.dumps(d,indent=2))
