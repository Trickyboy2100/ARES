"""Predeclared 96 native-horizon physical interventions; no learned selection."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for i,seed in enumerate(range(743040,743048)):
        mass=.75 if i%2==0 else 1.25
        companion=.09 if i%4<2 else .105
        for target in [-.08,.08]:
            for hook in [-.055,-.065,-.075]:
                jobs.append(dict(seed=seed,mass=mass,grasps=[.1,companion],target=target,hook=hook))
    source=Path(__file__).with_name('program_stap_tool.py')
    manifest=dict(episodes=96,control_steps=9600,design=jobs,
      limitation='Mass is balanced between scenes, not a within-scene causal contrast. Goals, hooks and the two grasps are paired within scene.',
      controller_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='ACTUAL OFFLINE PHYSICS, unchanged native horizon and conjunctive declared target')
    (a.out/'PREDECLARATION.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for n,j in enumerate(jobs):
        name=f"case{n:02d}_s{j['seed']}_y{j['target']}_h{j['hook']}"
        cmd=[sys.executable,str(source),'--out',str(a.out/name),'--seeds',str(j['seed']),'--mass-scales',str(j['mass']),'--grasps',','.join(map(str,j['grasps'])),'--target-y',str(j['target']),'--hook-y',str(j['hook']),'--target-feedback','--stability-max','10','--acceleration','6']
        start=time.monotonic()
        with (a.out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        rows.append(dict(case=j,command=cmd,exit_code=r.returncode,wall_s=time.monotonic()-start))
        (a.out/'QUEUE.json').write_text(json.dumps(rows,indent=2));print(name,r.returncode,flush=True)
        if r.returncode:raise RuntimeError('Original failed run retained; stop queue')
if __name__=='__main__':main()
