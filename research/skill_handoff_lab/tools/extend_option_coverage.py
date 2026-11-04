"""Add third shared successor option; reuse every existing physical branch.
Pilot episodes are reused, never silently re-counted as independent data.
"""
import argparse,json,gzip,time
from pathlib import Path
import torch
from program_peg_option_runtime import Runtime
from program_discrete_options import episode_transitions,OFFSETS

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 while True:
  try:
   rr=json.loads((a.reference/'ROWS.json').read_text())
   if len(rr)==192:break
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(5)
 ts=json.loads((a.reference/'TRANSITIONS.json').read_text());ref={(r['seed'],int(r['tag'][-2]),int(r['tag'][-1])):r for r in rr};scenes=sorted({r['seed']for r in rr});pilot={}
 if a.pilot:
  for r in json.loads((a.pilot/'physics/RESULTS.json').read_text()):
   if r['tag'].startswith('split'):pilot[r['seed'],int(r['tag'][-1])]=r
 rt=Runtime(a.out/'physics',successor_mode='split_refresh');pairs=[]
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(reference=str(a.reference),pilot=None if not a.pilot else str(a.pilot),scenes=scenes,actions='3x3 shared native100',claim='UNVERIFIED',repeat_prefix_cost='all actual executions charged; reused pilots not counted twice'),indent=2))
 try:
  for scene in scenes:
   for first in range(3):
    if (scene,first)in pilot:
     r=pilot[scene,first];d=json.load(gzip.open(r['path'],'rt'));t=episode_transitions(d,OFFSETS[first],True,r['path'])
     for v in t:v['controller']=rt.controller_id
     if len(t)>1:t[1]['a']=2
    else:r,t,d=rt.run(scene,lambda o:first if o[44]<.5 else 2,f'fixed{first}2')
    old=json.load(gzip.open(ref[scene,first,0]['path'],'rt'));n=t[0]['duration'];assert old['initial']==d['initial']and old['steps'][:n]==d['steps'][:n]
    rr.append(dict(r,tag=f'fixed{first}2'));ts+=t;pairs.append(dict(scene=scene,first=first,steps=n,exact=True))
   (a.out/'PAIRS.json').write_text(json.dumps(pairs));(a.out/'ROWS.json').write_text(json.dumps(sorted(rr,key=lambda r:(r['seed'],r['tag'])),indent=2))
   order={r['path']:i for i,r in enumerate(sorted(rr,key=lambda r:(r['seed'],r['tag'])))}
   (a.out/'TRANSITIONS.json').write_text(json.dumps(sorted(ts,key=lambda t:(order[t['source']],t['stage']))))
 finally:rt.close()
if __name__=='__main__':main()
