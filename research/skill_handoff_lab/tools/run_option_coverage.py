"""Fresh natural Peg coverage with complete finite physical branches; no sealed scenes."""
import argparse,json,hashlib,subprocess,time
from pathlib import Path
import torch
from program_peg_option_runtime import Runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=32);p.add_argument('--split',required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('program_*option*.py')}
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scenes=list(range(a.start,a.start+a.count)),split=a.split,actions='unchanged3x2',controller='unchanged strong official Peg100',sealed=False,sources=sources),indent=2))
 rt=Runtime(a.out/'physics');rows=[];trans=[];pairs=[]
 try:
  for scene in range(a.start,a.start+a.count):
   for first in range(3):
    previous=None
    for second in range(2):
     r,t,d=rt.run(scene,lambda o:first if o[44]<.5 else second,f'fixed{first}{second}');rows.append(r);trans+=t
     if previous:
      n=t[0]['duration'];assert d['initial']==previous['initial'] and d['steps'][:n]==previous['steps'][:n]
      pairs.append(dict(scene=scene,first=first,prefix_steps=n,exact=True))
     previous=d
    (a.out/'TRANSITIONS.json').write_text(json.dumps(trans));(a.out/'PAIRS.json').write_text(json.dumps(pairs))
   (a.out/'ROWS.json').write_text(json.dumps(rows,indent=2))
 finally:rt.close()
if __name__=='__main__':main()
