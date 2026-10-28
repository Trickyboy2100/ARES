"""Evidence regression checks, never a substitute for actual native simulation."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();n=0;steps=0;digests={}
 for f in sorted(a.root.glob('*/RESULTS.json')):
  for r in json.loads(f.read_text()):
   path=f.parent/f"{r['seed']}_{r['mode']}.json.gz"
   with gzip.open(path,'rt') as stream:t=json.load(stream)
   ss=t['steps'];charger='charger' in f.parent.name;limit=200 if charger else 100
   assert len(ss)==r['steps']==limit
   assert [s['step'] for s in ss]==list(range(1,limit+1))
   assert all(len(s['q'])==9 and len(s['tool'])==7 and len(s['tcp_T_tool'])==7 for s in ss)
   assert all(np.isfinite(np.r_[s['q'],s['qvel'],s['tool'],s['cube'],s['action']]).all() for s in ss)
   assert r['success']==ss[-1]['native_success']
   if charger:
    final=r['final'];native=final['obj_to_goal_dist'][0]<=.005 and final['obj_to_goal_angle'][0]<=.2
   else:native=ss[-1]['cube_base_distance_m']<.6
   assert bool(native)==r['success'],(f,r['seed'],r['mode'])
   sha=hashlib.sha256(json.dumps(t['initial'],sort_keys=True).encode()).hexdigest();assert sha==r['initial_sha256']
   key=(f.parent.name,r['seed']);assert key not in digests or digests[key]==sha;digests[key]=sha
   for d in r.get('decisions',[]):assert len(d['observation'])==54 and np.isfinite(d['observation']).all()
   assert not ('trajectory_sha256' in r and hashlib.sha256(path.read_bytes()).hexdigest()!=r['trajectory_sha256'])
   n+=1;steps+=limit
 print(json.dumps(dict(episodes=n,recorded_steps=steps,initial_pair_groups=len(digests),native_horizon_and_criterion='PASS',pose_action_finiteness='PASS',initial_state_hash_pairing='PASS',recorded_file_hashes='PASS')))
if __name__=='__main__':main()
