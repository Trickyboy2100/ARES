"""Recompute native plus target success from all96 original physical terminal states."""
import csv,gzip,json,hashlib,math
from pathlib import Path
base=Path(__file__).resolve().parents[1]/'research_program/evidence/EXP-003/2026-11-02/skillresult'
rows=[x for x in csv.DictReader((base/'EPISODES.csv').open()) if x['group']=='train96'];out=[]
for r in rows:
 p=Path(r['path']);d=json.load(gzip.open(p,'rt'));s=d['steps'][-1];target=float(r['target']);dist=math.hypot(s['cube'][0]+.045,s['cube'][1]-target);y=bool(s['native_success'] and dist<=.04)
 assert int(y)==int(r['success']);assert len(d['steps'])==100
 out.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),success=y,distance_m=dist))
path=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-003/pyramid_contract_20261102/RAW_TOOL_AUDIT.json');path.write_text(json.dumps(dict(episodes=len(out),success=sum(x['success'] for x in out),new_physics_steps=0,rows=out),indent=2));print('RAW96 MATCH',sum(x['success'] for x in out))
