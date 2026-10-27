"""Train ordinary pinned PPO on measured skill authority; no custom R3 likelihood."""
import argparse, hashlib, json, runpy, sys, os
from pathlib import Path
import numpy as np
import gymnasium as gym
from method_stack_physics import ParameterStack

def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--mode',choices=['parameter','residual'],required=True);p.add_argument('--name',required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=102400);a=p.parse_args()
    a.run_root.mkdir(parents=True,exist_ok=True);os.chdir(a.run_root)
    out=Path('runs')/a.name;out.mkdir(parents=True,exist_ok=False);original=gym.make
    class Seeded(gym.Wrapper):
        def __init__(self,env,validation):super().__init__(env);self.rng=np.random.default_rng(81000+a.seed+(10000 if validation else 0));self.pool=np.arange(200000,200128) if validation else np.arange(100000,100512)
        def reset(self,**kw):
            kw['seed']=self.rng.choice(self.pool,size=self.unwrapped.num_envs,replace=False).tolist()
            return self.env.reset(**kw)
    def make(task,**kw):
        assert task=='StackCube-v1';validation=kw.get('num_envs')==16;kw['control_mode']='pd_ee_delta_pos'
        return Seeded(ParameterStack(original(task,**kw),a.mode),validation)
    gym.make=make
    (out/'protocol.json').write_text(json.dumps(dict(mode=a.mode,seed=a.seed,steps=a.steps,upstream_ppo_sha256=hashlib.sha256(a.upstream.read_bytes()).hexdigest(),controller_sha256=hashlib.sha256(Path(__file__).with_name('method_stack_physics.py').read_bytes()).hexdigest(),task_unchanged=True,likelihood='ORDINARY_UPSTREAM_PPO',train_condition_pool=[100000,100511],validation_pool=[200000,200127],sealed_test_used=False,observation='privileged state baseline plus same phase/effect/applied/history/budget in both modes',interpretation='Parameter/action-interface study; not an original PPO loss'),indent=2)+'\n')
    sys.argv=[str(a.upstream),'--env-id','StackCube-v1','--exp-name',a.name,'--seed',str(a.seed),'--num-envs','64','--num-eval-envs','16','--total-timesteps',str(a.steps),'--num-steps','50','--num-eval-steps','50','--num-minibatches','8','--eval-freq','16','--no-capture-video','--save-model','--no-partial-reset','--reconfiguration-freq','1','--control-mode','pd_ee_delta_pos']
    runpy.run_path(str(a.upstream),run_name='__main__')

if __name__=='__main__':main()
