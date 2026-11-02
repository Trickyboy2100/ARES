"""Validate actual SAC replay transitions against full saved physical episodes."""
import argparse,pickle,json,gzip,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'external/sb3-2.7.0'))
from program_stap_learning import features

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--buffer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();buf=pickle.load(a.buffer.open('rb'));expected=[]
 for r in json.loads((a.data/'RESULTS.json').read_text()):
  d=json.load(gzip.open(r['path'],'rt'));acts=d['policy_actions'];ds=d['decisions'];x=features(ds[0]['observation'],100,r['mass'],0,0);end=features(d['steps'][-1],0,r['mass'],0,1)
  if len(acts)>1:
   z=features(ds[1]['observation'],100-ds[1]['boundary_step'],r['mass'],0,1);expected.extend([(x,z,acts[0]['normalized'],0.,False),(z,end,acts[1]['normalized'],float(r['success']),True)])
  else:expected.append((x,end,acts[0]['normalized'],float(r['success']),True))
 assert len(expected)==buf.size(),(len(expected),buf.size());errors=[]
 for i,(x,z,u,r,done) in enumerate(expected):
  errors.append(max(float(np.abs(buf.observations[i,0]-x).max()),float(np.abs(buf.next_observations[i,0]-z).max()),abs(float(buf.actions[i,0,0])-u),abs(float(buf.rewards[i,0])-r),abs(float(buf.dones[i,0])-done)))
 assert max(errors)<1e-6
 result=dict(transitions=len(expected),max_transition_error=max(errors),nonterminal_count=sum(not x[-1]for x in expected),terminal_count=sum(x[-1]for x in expected),contract='gamma1 finite horizon: sparse terminal native success, state contains remaining budget, no cross-reset bootstrap',status='PASS');a.out.write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
