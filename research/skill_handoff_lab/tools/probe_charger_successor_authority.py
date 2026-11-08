"""Bounded legal pace interventions after every successful-but-unsolved nominal grasp."""
import argparse,json,random,shutil,hashlib
from pathlib import Path
import numpy as np,torch
import program_native_charger_pace as physical
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();source=json.loads((a.root/'charger_design/RESULTS.json').read_text());assert len(source)==72;selected=[r for r in source if r['policy']=='nominal'and not r['success']and r['prefix_steps']<200 and physical.prefix_failure(r)is None];policies=['advance_soft','rotate_soft','force_guard'];out=a.root/'charger_successor_probe';out.mkdir(exist_ok=False)
(out/'PROTOCOL.json').write_text(json.dumps(dict(scope='SEEN_DESIGN_ACTION_AUTHORITY, no learning or independent performance claim',selection='All nominal failures with an executable successful prefix; failed grasps excluded as absorbing',selected=[dict(seed=r['seed'],angle=r['angle'],source=r['path'])for r in selected],policies=policies,native200=True,observations='Existing actual contact/geometric feedback only; no extra action range',source_sha={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),Path(physical.__file__)]}),indent=2));shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(physical.__file__,out/Path(physical.__file__).name);ep=physical.make_native();rows=[]
try:
 for r in selected:
  for policy in policies:
   random.seed(31031);np.random.seed(31031);torch.manual_seed(31031);obs=physical.replay_prefix(ep,r['path']);assert ep.error is None
   while not ep.done:obs,reward,done=ep.step(physical.action_for(policy,obs))
   row=physical.save(ep,out,len(rows),r['angle'],policy);rows.append(row);(out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print('probe',r['seed'],r['angle'],policy,row['success'],row['error'],row['prefix_replay']['max_physical_error'],flush=True)
finally:ep.close()
