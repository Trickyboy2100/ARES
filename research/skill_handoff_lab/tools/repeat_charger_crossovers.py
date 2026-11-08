"""All disagreements between the two strongest predeclared grasp angles; seen repeats."""
import argparse,json,random,shutil,hashlib
from pathlib import Path
import numpy as np,torch
import program_native_charger_pace as physical
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=json.loads((a.root/'charger_design/RESULTS.json').read_text());assert len(rows)==72;by={(r['seed'],r['angle'],r['policy']):r for r in rows};selected=[s for s in sorted({r['seed']for r in rows})if by[s,15,'nominal']['success']!=by[s,30,'nominal']['success']];out=a.root/'charger_crossover_repeats';out.mkdir(exist_ok=False)
(out/'PROTOCOL.json').write_text(json.dumps(dict(scope='SEEN_MECHANISM_REPEAT, not fresh confirmation or training',selection='All observed nominal15-versus30 outcome disagreements; no selection on repeat outcomes',selected=selected,angles=[15,30],repeats=2,native200=True,policy='unchanged nominal pace, original native resets and complete replanning',source_sha={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),Path(physical.__file__)]}),indent=2));shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(physical.__file__,out/Path(physical.__file__).name);ep=physical.make_native();results=[]
try:
 for rep in range(2):
  for seed in selected:
   for angle in [15,30]:
    random.seed(31031);np.random.seed(31031);torch.manual_seed(31031);obs=ep.reset(seed,(0,0),angle=angle)
    while not ep.done:obs,reward,done=ep.step(physical.action_for('nominal',obs))
    row=physical.save(ep,out,len(results),angle,'repeated_nominal');row['repeat']=rep;results.append(row);(out/'RESULTS.json').write_text(json.dumps(results,indent=2));print('repeat',rep,seed,angle,row['success'],row['prefix_steps'],row['error'],flush=True)
finally:ep.close()
