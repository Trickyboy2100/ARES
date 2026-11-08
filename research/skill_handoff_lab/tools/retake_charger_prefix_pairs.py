"""Reexecute only mismatched-prefix controls; selection ignores outcomes."""
import argparse,gzip,json,hashlib,random,shutil,time
from pathlib import Path
import numpy as np,torch
import program_native_charger_pace as physical
from pull_tool_physics import serial
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();src=a.root/'charger_design';out=a.root/'charger_prefix_retakes';out.mkdir(exist_ok=False);rows=json.loads((src/'RESULTS.json').read_text());assert len(rows)==72;by={(r['seed'],r['angle'],r['policy']):r for r in rows};chosen=[];checks=[]
for seed in sorted({r['seed']for r in rows}):
 for angle in [15,30,45]:
  left=json.load(gzip.open(by[seed,angle,'nominal']['path'],'rt'));right=json.load(gzip.open(by[seed,angle,'feedback']['path'],'rt'));n=left['summary']['prefix_steps'];same=n==right['summary']['prefix_steps'];err=max((float(np.max(np.abs(np.asarray(x[k])-np.asarray(y[k]))))for x,y in zip(left['steps'][:n],right['steps'][:n])for k in ['q','qvel','tcp','tool','cube']),default=0.)
  checks.append(dict(seed=seed,angle=angle,same_length=same,max_recorded_prefix_error=err))
  if not same or err>1e-5:chosen.append(by[seed,angle,'nominal'])
protocol=dict(selection='Only recorded prefix length/error mismatch >1e-5; no selection on success/failure',initial_checks=checks,selected=[dict(seed=r['seed'],angle=r['angle'],source=r['path'])for r in chosen],native200=True,source_sha={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),Path(physical.__file__)]},no_state_setters=True)
(out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(physical.__file__,out/Path(physical.__file__).name);ep=physical.make_native();results=[];partial=[]
(out/'RESULTS.json').write_text('[]')
try:
 for r in chosen:
  random.seed(31031);np.random.seed(31031);torch.manual_seed(31031)
  try:obs=physical.replay_prefix(ep,r['path'])
  except RuntimeError as ex:
   path=out/f'UNMATCHED_{r["seed"]}_{r["angle"]}.json.gz'
   with gzip.open(path,'wt')as f:json.dump(dict(seed=r['seed'],angle=r['angle'],error=str(ex),steps=ep.env.records,initial=ep.env.initial,source=r['path']),f)
   partial.append(dict(seed=r['seed'],angle=r['angle'],physical_steps=len(ep.env.records),error=str(ex),path=str(path)));ep.planner.close();continue
  if ep.error is not None or len(ep.env.records)>=200:
   # Failed predecessor is absorbing: no fictitious successor decision.
   e=ep.env;q=np.asarray(serial(e.unwrapped.agent.robot.get_qpos()[0,:7]));e.phase='TERMINAL_HOLD'
   while len(e.records)<200:e.step(np.r_[q,-1.])
   ep.done=True;ep.planner.close()
  else:
   while not ep.done:obs,reward,done=ep.step(physical.action_for('feedback',obs))
  row=physical.save(ep,out,len(results),r['angle'],'feedback_replayed_prefix');results.append(row);(out/'RESULTS.json').write_text(json.dumps(results,indent=2));print('retake',r['seed'],r['angle'],row['success'],row['error'],row['prefix_replay']['max_physical_error'],flush=True)
finally:
 (out/'PARTIALS.json').write_text(json.dumps(partial,indent=2));ep.close()
