"""Bounded sequential GPU0 experiment batch; never detaches or kills other jobs."""
import argparse,json,os,signal,subprocess,time,shutil,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
    base=a.workspace/'runs/method_evolution_20261028/training';source=base/'source';source.mkdir(parents=True,exist_ok=True)
    for name in ['method_train.py','method_stack_physics.py']:
        content=(a.root/'research/skill_handoff_lab/tools'/name).read_bytes();target=source/name
        if target.exists():assert target.read_bytes()==content,'Frozen training source differs'
        else:target.write_bytes(content)
    rows=[];jobdir=base/'runs';python=a.workspace/'envs/method-evolution-20261028/bin/python';upstream=a.workspace/'external/benchmark_scout_20261027/ManiSkill/examples/baselines/ppo/ppo.py'
    for seed in [31,37,43]:
        for mode in ['residual','parameter']:
            name=f'{mode}_{seed}_102400';out=jobdir/name
            command=['bash',str(a.root/'research/skill_handoff_lab/tools/method_offline.sh'),str(python),str(source/'method_train.py'),'--mode',mode,'--seed',str(seed),'--steps','102400','--name',name,'--run-root',str(base),'--upstream',str(upstream)]
            start=time.monotonic();timeout=False
            with (base/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=a.root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    timeout=True;os.killpg(child.pid,signal.SIGTERM)
                    try:code=child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            record=dict(mode=mode,seed=seed,requested_interactions=102400,returncode=code,timeout=timeout,wall_s=time.monotonic()-start,checkpoint=str(out/'final_ckpt.pt'),complete=(out/'final_ckpt.pt').exists(),command=command)
            if record['complete']:record['checkpoint_sha256']=hashlib.sha256((out/'final_ckpt.pt').read_bytes()).hexdigest()
            rows.append(record);(base/'BATCH.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(record),flush=True)
            if code!=0:raise RuntimeError('Stop batch on first error; fix before allocating more training')

if __name__=='__main__':main()
