import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'research/skill_handoff_lab/tools'))
import program_post_contact as p
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[5]/'runs/research_program/EXP-003/post_contact_20261102');args=parser.parse_args();root=args.root;out=root/'near';out.mkdir(exist_ok=False)
rs=json.loads((root/'ordinary/episodes/RESULTS.json').read_text());cases=[]
for r in rs:
 if r['contact_step'] is not None and not r['success'] and r['final_distance']<.75:
  if r['seed'] not in [x['seed']for x in cases]:cases.append(r)
(out/'SELECTION.json').write_text(json.dumps(dict(rule='first encountered unique training scene with contact, failure, terminal distance<0.75; diagnostic only, not independent evaluation',cases=cases),indent=2));rows=[];e=p.make_env();e.correction='relative_yaw'
try:
 for r in cases:
  for action in [[0,0],[-1,0],[1,0],[0,-1],[0,1]]:
   t=time.monotonic();err=p.prefix(e,r['seed'],r['grasp'])
   if e.contact_event is not None:err=p.suffix(e,action)
   p.finish(e);row=p.save(e,out,r['seed'],r['grasp'],action,err,t,len(rows));rows.append(row);(out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(row['seed'],action,row['success'],row['final_distance'],err,flush=True)
finally:e.close()
