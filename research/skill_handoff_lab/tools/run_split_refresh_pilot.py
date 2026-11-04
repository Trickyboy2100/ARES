"""Bounded shared-option intervention; not an RL gain or new task criterion."""
import argparse,gzip,json
from pathlib import Path
import torch
from program_peg_option_runtime import Runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False);torch.set_num_threads(2);rows=json.loads((a.reference/'ROWS.json').read_text());ref={(r['seed'],int(r['tag'][-2]),int(r['tag'][-1])):r for r in rows};rt=Runtime(a.out/'physics',successor_mode='split_refresh');out=[]
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(scenes=list(range(752000,752008)),scope='first8 consecutive training-design scenes, not selected failures or independent evaluation',parameter='one halfway insertion waypoint plus refreshed attachment target',physical_budget=100,original_options_unchanged=True,expected_new_episodes=28,claim='shared controller/action-authority study, NOT RL'),indent=2))
 try:
  for scene in [752000,752001]:
   for b in [0,1]:
    r,t,d=rt.run(scene,lambda o:1 if o[44]<.5 else b,f'compat1{b}');old=json.load(gzip.open(ref[scene,1,b]['path'],'rt'));assert d['initial']==old['initial'] and d['steps']==old['steps'];out.append(dict(scene=scene,first=1,second=b,compat_full100=True,success=r['success'],path=r['path']))
  for scene in range(752000,752008):
   for aa in range(3):
    r,t,d=rt.run(scene,lambda o:aa if o[44]<.5 else 2,f'split{aa}');old=json.load(gzip.open(ref[scene,aa,0]['path'],'rt'));n=t[0]['duration'];assert d['initial']==old['initial'] and d['steps'][:n]==old['steps'][:n]
    out.append(dict(scene=scene,first=aa,second=2,success=r['success'],old_success=[ref[scene,aa,b]['success']for b in range(2)],prefix_steps=n,exact_prefix=True,path=r['path']));(a.out/'COMPARISON.json').write_text(json.dumps(out,indent=2))
 finally:rt.close()
if __name__=='__main__':main()
