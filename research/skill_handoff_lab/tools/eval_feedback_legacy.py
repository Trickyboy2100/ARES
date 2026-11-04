"""Retained mature frozen skills on the new development scenes, no retraining."""
import argparse,json
from pathlib import Path
from program_peg_option_runtime import Runtime
from program_option_improvement import Improved
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
m=Improved.load(a.model);rt=Runtime(a.out/'physics',successor_mode='split_refresh')
for seed in range(753200,753216):
 rt.run(seed,lambda s:1,'fixed11')
 rt.run(seed,lambda s:m.choose(s),'retained_sac')
rt.close()
