"""Reconstruct ordinary SMDP replay from measured native trajectories (CPU only)."""
import argparse,json,pickle,gzip,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'external/sb3-2.7.0'))
import numpy as np
import sapien
from transforms3d.quaternions import quat2axangle
from transforms3d.euler import euler2quat
from program_stap_learning import features

def obs(m,step,stage):
 c=m['cube'];t=m['tool'];goal=sapien.Pose(c[:3],c[3:])*sapien.Pose(q=euler2quat(0,0,np.pi));local=goal.inv()*sapien.Pose(t[:3],t[3:]);axis,angle=quat2axangle(local.q)
 if angle>np.pi:angle-=2*np.pi
 force=np.array(m['tool_socket_force_vector_N'])if stage else np.zeros(3)
 return np.r_[features(m,(200-step)/2,1.,0.,stage),local.p/.025,axis*angle/.2,force/10].astype(np.float32)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=int,default=1024);a=p.parse_args();b=pickle.load((a.root/f'BUFFER_{a.checkpoint}.pkl').open('rb'));expected=[];cost=0
 rows=json.loads((a.root/'episodes/RESULTS.json').read_text());files=[Path(r['path'])for r in rows];partial=a.root/'PARTIAL.json.gz'
 if partial.exists():files.append(partial)
 for f in files:
  d=json.load(gzip.open(f,'rt'));acts=d.get('policy_actions',d.get('actions'));states={0:d['initial_measure'],**{z['step']:z for z in d['steps']}};cost+=len(d['steps'])
  for act in acts:
   start=act['step'];end=start+act['duration'];stage=act['stage'];done=end==200;rew=float(d['steps'][-1]['native_success'])if done else 0.
   expected.append((obs(states[start],start,stage),obs(states[end],end,1),act['action'],rew,done))
  if len(expected)>=b.size():break
 errors=[]
 for i,(x,z,action,reward,done)in enumerate(expected[:b.size()]):errors.append(dict(i=i,obs=float(np.max(abs(x-b.observations[i,0]))),next=float(np.max(abs(z-b.next_observations[i,0]))),action=float(np.max(abs(np.array(action)-b.actions[i,0]))),reward=abs(reward-float(b.rewards[i,0])),done=abs(float(done)-float(b.dones[i,0]))))
 maxima={k:max(e[k]for e in errors)for k in ['obs','next','action','reward','done']};result=dict(transitions=b.size(),reconstructed=len(errors),max_errors=maxima,status='PASS'if len(errors)==b.size()and max(maxima.values())<2e-4 else 'FAIL',source_files=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in files],errors=errors);(a.root/f'BUFFER_AUDIT_{a.checkpoint}.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items()if k not in ['errors','source_files']});assert result['status']=='PASS'
if __name__=='__main__':main()
