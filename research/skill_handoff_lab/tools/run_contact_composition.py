"""Reuse deployed Isaac Lab / SB3; launch a genuinely dynamic object task."""
import argparse,json,os,traceback
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=16);p.add_argument('--gpu',type=int,default=0)
p.add_argument('--steps',type=int,default=8192);p.add_argument('--seed',type=int,default=31)
p.add_argument('--family',choices=['push','slot','both'],default='both')
p.add_argument('--controller',choices=['ppo','nominal','goal_feedback','residual','random','fixed_push'],default='ppo')
p.add_argument('--credit-mode',choices=['standard','boundary','all'],default='standard')
p.add_argument('--credit-mix',type=float,default=.5);p.add_argument('--gamma',type=float,default=.99)
p.add_argument('--rollout-steps',type=int,default=128)
p.add_argument('--scenario-file',type=Path);p.add_argument('--eval-seed',type=int,default=1731)
p.add_argument('--action-intervention',choices=['none','phase0_feedback','phase1_feedback','zero'],default='none')
p.add_argument('--eval-episodes',type=int,default=64);p.add_argument('--load-policy',type=Path)
p.add_argument('--skip-capture',action='store_true');p.add_argument('--episode-seconds',type=float,default=8.)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device=f'cuda:{a.gpu}',enable_cameras=not a.skip_capture,
    kit_args=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false')
try:
    from contact_composition import run
    run(a)
except BaseException:
    (a.output/'exception.txt').write_text(traceback.format_exc());traceback.print_exc();raise
finally:launcher.app.close(wait_for_replicator=False,skip_cleanup=True)
