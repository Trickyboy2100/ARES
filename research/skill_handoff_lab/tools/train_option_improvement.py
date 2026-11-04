"""Same-data chronological physical-learning comparison; shared real collection.
This is replay training as data become available, NOT each learner collecting
its own policy's experience. Online policy trials are a separate experiment.
"""
import argparse,json,hashlib,shutil,subprocess
from pathlib import Path
import torch
from program_discrete_options import Learner,prior,normalizer
from program_option_improvement import Improved,verify_physical_support

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=31401);p.add_argument('--successor-actions',type=int,default=2);p.add_argument('--methods',default='sac_min,sac_mean,vi_mean,vi_expectile,usac,handoff,lower_bound,fqi,iql,supervised,stap');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 old,_=prior(a.prior);new=json.loads(a.data.read_text());(a.out/'SUPPORT_AUDIT.json').write_text(json.dumps(verify_physical_support(old+new),indent=2));sources=list(dict.fromkeys(t['source']for t in new));hist=[]
 for f in ['program_option_improvement.py','train_option_improvement.py','program_discrete_options.py','program_peg_option_runtime.py']:shutil.copy2(Path(__file__).with_name(f),a.out/f)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in a.out.glob('*.py')},prior=str(a.prior),data=str(a.data),data_sha=hashlib.sha256(a.data.read_bytes()).hexdigest(),seed=a.seed,methods=a.methods,successor_actions=a.successor_actions,physical_episodes=len(sources),physical_steps=100*len(sources),initial_updates=800,updates_per_episode=25,checkpoints=[0,48,96,len(sources)],split='new training design, no development data',claim='UNVERIFIED',collection='shared complete natural-scene intervention, not own-policy online'),indent=2))
 for kind in a.methods.split(','):
  cls=Learner if kind in ('fqi','iql','supervised','stap')else Improved
  m=cls(kind,a.seed,normalizer(old),successor_actions=a.successor_actions);m.train(old,steps=800);m.save(a.out/f'{kind}_0.pt');buf=[]
  for k,path in enumerate(sources,1):
   buf +=[t for t in new if t['source']==path];m.train(old,buf,steps=25)
   if k in (48,96,192)or k==len(sources):m.save(a.out/f'{kind}_{k}.pt')
  hist.append(dict(method=kind,updates=m.updates,history=m.history));(a.out/'HISTORY.json').write_text(json.dumps(hist,indent=2));print('TRAINED',kind,a.seed,m.updates,flush=True)
if __name__=='__main__':main()
