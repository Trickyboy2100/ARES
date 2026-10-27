"""Equal-physics-budget global CEM baseline, trained only on train conditions.

16 candidates x96 cold resets x50 steps = 76800 native interactions, exactly
the paired selector dataset cost. Not a per-test oracle; publishes one fixed
height parameter for subsequent independent evaluation.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from method_stack_physics import make,serial

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(31);env=make(16);mean,std=0.,1.;rows=[];hashes={};best=None;at=time.monotonic()
    for iteration in range(2):
        requested=rng.normal(mean,std,8);requested[0]=mean
        if iteration==0:requested[1:3]=[-1.,1.]
        for j,raw in enumerate(requested):
            action=float(np.clip(raw,-1,1));successes=[];traces=[]
            for batch in range(6):
                seeds=list(range(100000+batch*16,100000+(batch+1)*16));obs,_=env.reset(seed=seeds);u=env.unwrapped
                initial=serial(u.get_state_dict());digest=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
                if batch not in hashes:hashes[batch]=digest
                assert hashes[batch]==digest
                states=[]
                for step in range(50):
                    obs,reward,_,_,info=env.step(torch.tensor([0.,0.,action],device=u.device).expand(16,3));states.append(serial(dict(step=step+1,cubeA=u.cubeA.pose.raw_pose,cubeB=u.cubeB.pose.raw_pose,q=u.agent.robot.get_qpos(),tcp=u.agent.tcp.pose.raw_pose,success=info['success'],authority=info['authority'])))
                ys=serial(u.evaluate()['success']);successes+=ys
                file=a.output/f'round{iteration}_candidate{j}_batch{batch}.json';file.write_text(json.dumps(dict(initial=initial,initial_sha256=digest,seeds=seeds,requested=raw,clipped=action,transitions=states),separators=(',',':')));traces.append(dict(path=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            row=dict(iteration=iteration,candidate=j,requested=float(raw),clipped=action,success=sum(successes),cases=96,interactions=4800,traces=traces);rows.append(row)
            if best is None or (row['success'],-abs(row['clipped']))>(best['success'],-abs(best['clipped'])):best=row
            print(json.dumps({k:row[k] for k in ['iteration','candidate','clipped','success']}),flush=True)
            (a.output/'RESULTS.json').write_text(json.dumps(dict(source_kind='NEW_PHYSICAL_SIMULATION',task='StackCube-v1',native_horizon=50,rows=rows,physical_interactions=len(rows)*4800,best=dict(parameter=[0,0,best['clipped']],train_success=best['success'],train_cases=96),wall_s=time.monotonic()-at,validation_or_test_used=False,equal_budget_to='512 x3 x50 paired selector data'),indent=2)+'\n')
        elite=sorted(rows[-8:],key=lambda r:(r['success'],-abs(r['clipped'])),reverse=True)[:2];mean=float(np.mean([r['clipped'] for r in elite]));std=max(.12,float(np.std([r['clipped'] for r in elite])))
    env.close()

if __name__=='__main__':main()
