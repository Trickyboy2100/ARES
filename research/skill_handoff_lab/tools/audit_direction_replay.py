"""Recompute every saved replay transition from its actual source physical trace."""
import argparse,json,gzip,pickle,hashlib
from pathlib import Path
import numpy as np
from program_contact_direction import transitions,observation
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--checkpoint',default='OFFLINE');p.add_argument('--out',type=Path,required=True);a=p.parse_args();run=a.root/a.run;effect=json.loads((a.root/'effect_model_physical/EFFECT_MODEL.json').read_text());expected=[]
for r in effect['dataset']:
 assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'];expected.extend(transitions(json.load(gzip.open(r['path'],'rt'))))
shared=len(expected);buffer=pickle.load((run/('OFFLINE_BUFFER.pkl'if a.checkpoint=='OFFLINE'else f'BUFFER_{a.checkpoint}.pkl')).open('rb'))
if a.checkpoint!='OFFLINE':
 for r in json.loads((run/'episodes/RESULTS.json').read_text()):expected.extend(transitions(json.load(gzip.open(r['path'],'rt'))))
if len(expected)<buffer.size():
 partial=json.load(gzip.open(run/'PARTIAL.json.gz','rt'));assert len(partial['actions'])==1
 initial=None
 for r in json.loads((run/'episodes/RESULTS.json').read_text()):
  if r['seed']==partial['seed']:
   d=json.load(gzip.open(r['path'],'rt'))
   if d['initial']==partial['initial']:initial=d['initial_measure'];break
 assert initial is not None
 plan=next(v for v in partial['plans']if v['phase']=='PULL'and 'attempts'in v);before=partial['steps'][plan['step']-1];q=plan['target'][3:];goal=Rotation.from_quat([*q[1:],q[0]]).apply(before['tcp_T_tool'][:3])+plan['target'][:3]
 stable=next(z for z in reversed(partial['steps'])if z['phase']=='GRASP_STABILITY');last=partial['steps'][-1]
 expected.append((observation(initial,0,100,[0,0],[0,0,0]),observation(last,1,100-len(partial['steps']),np.array(last['cube'][:2])-stable['cube'][:2],goal),partial['actions'][0]['normalized'],0.,False))
maxerr=np.zeros(5)
for i,(x,z,u,reward,done)in enumerate(expected[:buffer.size()]):
 err=[np.max(abs(buffer.observations[i,0]-x)),np.max(abs(buffer.next_observations[i,0]-z)),np.max(abs(buffer.actions[i,0]-u)),abs(float(buffer.rewards[i,0])-reward),abs(float(buffer.dones[i,0])-done)];maxerr=np.maximum(maxerr,err)
assert len(expected)>=buffer.size();assert max(maxerr)<1e-5
result=dict(status='PASS',run=a.run,checkpoint=a.checkpoint,transitions=buffer.size(),shared_physical_transitions=shared,max_errors=dict(zip(['observation','next_observation','action','native_reward','terminal'],maxerr.tolist())),source='full physical initial state/contact boundary/final state and actual policy actions, not model predicted dynamics');a.out.write_text(json.dumps(result,indent=2));print(result)
