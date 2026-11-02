"""Audit actual two-boundary SAC obs/actions/durations, including initial double-reset consistency."""
import argparse,sys,json,gzip,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'external/sb3-2.7.0'))
import numpy as np
from program_stap_learning import features
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=int,default=128);p.add_argument('--out',type=Path,required=True);a=p.parse_args();buffer=pickle.load((a.root/f'BUFFER_{a.checkpoint}.pkl').open('rb'));rows=json.loads((a.root/'episodes/RESULTS.json').read_text());expected=[]
def obs(m,stage,remaining,progress,goal):return np.r_[features(m,remaining,1.,0.,stage),m['cube_linear_velocity'],np.log1p(m['tool_cube_force_N'])if stage else 0.,progress,goal].astype(np.float32)
for r in rows:
 d=json.load(gzip.open(r['path'],'rt'));actions=d['policy_actions'];x=obs(d['initial_measure'],0,100,[0,0],[0,0,0]);event=d['contact_event'];goal=np.zeros(3);z=None
 if event:
  # Recover actual PULL tool target from physical plan target and measured attachment at PULL start.
  from scipy.spatial.transform import Rotation
  plan=next(v for v in d['plans']if v['phase']=='PULL'and 'attempts'in v);before=d['steps'][plan['step']-1];q=plan['target'][3:];rot=Rotation.from_quat([*q[1:],q[0]]);goal=rot.apply(before['tcp_T_tool'][:3])+plan['target'][:3]
  start=np.array(d['skill_results'][0]['measurement']['cube'][:2]);z=obs(event['measurement'],1,event['remaining'],np.array(event['measurement']['cube'][:2])-start,goal)
  terminal=obs(d['steps'][-1],1,0,np.array(d['steps'][-1]['cube'][:2])-start,goal)
  expected.extend([(x,z,actions[0]['normalized'],0.,False),(z,terminal,actions[1]['normalized'],float(r['success']),True)])
 else:expected.append((x,obs(d['steps'][-1],1,0,[0,0],[0,0,0]),actions[0]['normalized'],float(r['success']),True))
 if len(expected)>=buffer.size():break
errors=[]
for i,(x,z,u,reward,done)in enumerate(expected[:buffer.size()]):errors.append(dict(i=i,obs=float(np.max(abs(buffer.observations[i,0]-x))),next=float(np.max(abs(buffer.next_observations[i,0]-z))),action=float(np.max(abs(buffer.actions[i,0]-u))),reward=abs(float(buffer.rewards[i,0])-reward),done=abs(float(buffer.dones[i,0])-done)))
maxerr={k:max(r[k]for r in errors)for k in ['obs','next','action','reward','done']};result=dict(transitions=buffer.size(),max_errors=maxerr,status='PASS'if max(maxerr.values())<1e-5 else 'FAIL',errors=errors);a.out.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items()if k!='errors'});assert result['status']=='PASS'
