"""Predeclared task-condition intervention queue; same native goal and100 steps.
Extra successor goal: cube within40mm of declared target AND native success.
Target y does not move/reset any object; it changes commanded successor skill.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',default='743010,743011');p.add_argument('--hooks',default='auto');p.add_argument('--targets',default='-.08,.08');p.add_argument('--feedback',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
 for target in map(float,a.targets.split(',')):
  for hook in a.hooks.split(','):
   name=f'target{target}_hook{hook}';cmd=[sys.executable,str(Path(__file__).with_name('program_stap_tool.py')),'--out',str(a.out/name),'--seeds',a.seeds,'--target-y',str(target),'--stability-max','10','--acceleration','6']
   if a.feedback:cmd+=['--target-feedback']
   if hook!='auto':cmd+=['--hook-y',hook]
   started=time.monotonic()
   with (a.out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
   rows.append(dict(command=cmd,exit_code=r.returncode,wall_s=time.monotonic()-started));(a.out/'QUEUE.json').write_text(json.dumps(rows,indent=2));print(name,r.returncode,flush=True)
   if r.returncode:raise RuntimeError('child failure; original logs retained')
if __name__=='__main__':main()
