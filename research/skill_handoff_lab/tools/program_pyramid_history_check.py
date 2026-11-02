"""Verify historical outcome table against all 36 original vector traces."""
import json,hashlib
from pathlib import Path
workspace=Path('/home/andyee/robot-research/ARES-Skilllab')
root=workspace/'runs/single_paper_20261028/opportunity64'
rows=[]
for path in sorted(root.glob('*_A*_C*.json')):
 d=json.loads(path.read_text());assert len(d['transitions'])==250
 assert d['transitions'][-1]['info']['success']==d['summary']['success']
 rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),episodes=len(d['summary']['seeds']),success=sum(d['summary']['success'])))
assert len(rows)==36 and sum(x['episodes'] for x in rows)==576
out=workspace/'runs/research_program/EXP-003/pyramid_contract_20261102/RAW_PYRAMID_AUDIT.json'
out.write_text(json.dumps(dict(episodes=576,success=sum(x['success'] for x in rows),new_physics_steps=0,rows=rows),indent=2));print('Verified',len(rows),'raw batches; success',sum(x['success'] for x in rows))
