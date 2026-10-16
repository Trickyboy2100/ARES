"""Reuse deployed Isaac Lab / SB3; launch a genuinely dynamic object task."""
import argparse,json,os,traceback
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=16);p.add_argument('--gpu',type=int,default=0)
p.add_argument('--steps',type=int,default=8192);p.add_argument('--seed',type=int,default=31)
p.add_argument('--family',choices=['push','slot','both'],default='both')
p.add_argument('--controller',choices=['ppo','nominal','goal_feedback','damped','legacy','paired','residual','random','fixed_push'],default='ppo')
p.add_argument('--credit-mode',choices=['standard','boundary','all'],default='standard')
p.add_argument('--credit-mix',type=float,default=.5);p.add_argument('--gamma',type=float,default=.99)
p.add_argument('--rollout-steps',type=int,default=128)
p.add_argument('--scenario-file',type=Path);p.add_argument('--eval-seed',type=int,default=1731)
p.add_argument('--action-intervention',choices=['none','phase0_feedback','phase1_feedback','zero'],default='none')
p.add_argument('--eval-episodes',type=int,default=64);p.add_argument('--load-policy',type=Path)
p.add_argument('--skip-capture',action='store_true');p.add_argument('--episode-seconds',type=float,default=8.)
p.add_argument('--task-version',choices=['v5_replay','v6','v6b'],default='v6')
p.add_argument('--legacy-policy',type=Path);p.add_argument('--stochastic-eval',action='store_true')
p.add_argument('--switch-mode',choices=['fixed','learned','recover'],default='fixed');p.add_argument('--switch-dwell',type=int,default=0)
p.add_argument('--initial-log-std',type=float,default=-1.5);p.add_argument('--feedback-kp',type=float,default=1.5)
p.add_argument('--feedback-kd',type=float,default=2.);p.add_argument('--blind-next',action='store_true')
p.add_argument('--blind-rotation',action='store_true')
p.add_argument('--force-switch-signal',type=float,default=None)
p.add_argument('--rule-recovery',action='store_true',help='same-policy causal comparator: hold near the goal, resume otherwise')
p.add_argument('--policy-mode',choices=['gaussian','hybrid_joint','hybrid_conditional'],default='gaussian')
p.add_argument('--action-cost',choices=['raw','executed'],default='raw')
p.add_argument('--withdraw-rule',action='store_true',help='validation-only bounded contact release/recontact probe')
p.add_argument('--checkpoint-every',type=int,default=32768)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device=f'cuda:{a.gpu}',enable_cameras=not a.skip_capture,
    kit_args=f'--/renderer/activeGpu={a.gpu} --/renderer/multiGpu/enabled=false')
try:
    from contact_composition_v6 import run
    run(a)
    (a.output/'RUN_COMPLETE.json').write_text(json.dumps({'status':'COMPLETE'}))
except BaseException:
    (a.output/'exception.txt').write_text(traceback.format_exc());traceback.print_exc()
    # Kit fast-close may exit(0) even with an active Python exception. Preserve
    # the failure code; OS cleanup releases only this worker's GPU context.
    import sys
    sys.stdout.flush();sys.stderr.flush();os._exit(1)
launcher.app.close(wait_for_replicator=False,skip_cleanup=True)
