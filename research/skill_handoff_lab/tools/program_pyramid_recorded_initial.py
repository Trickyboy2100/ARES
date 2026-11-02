"""Counterfactual physics from exact historical initial states, ONLY at reset.
Not a native RNG replay: batch size changes ManiSkill's sampled initialization.
No state setters after the first physical action or between skills.
"""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
import program_pyramid_joint_probe as run
from handoff_vector import VectorHandoff,serial
ROOT=Path('/home/andyee/robot-research/ARES-Skilllab/runs/single_paper_20261028/opportunity64')
sources=[ROOT/'000_A0_C0.json',ROOT/'001_A0_C0.json'];originals=[json.loads(p.read_text())['initial'] for p in sources]
state={g:{k:np.stack([np.asarray(originals[0][g][k][2]),np.asarray(originals[1][g][k][7])]) for k in data} for g,data in originals[0].items()}
original_reset=VectorHandoff.reset
errors=[]
def reset(self,seeds):
 original_reset(self,seeds);assert self.steps==0
 target={g:{k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in data.items()} for g,data in state.items()}
 self.u.set_state_dict(target);read=self.u.get_state_dict();err=max(float((read[g][k]-target[g][k]).abs().max()) for g in target for k in target[g]);assert err<1e-5;errors.append(err)
 self.initial=serial(read);self.initial_sha=hashlib.sha256(json.dumps(self.initial,sort_keys=True).encode()).hexdigest();self.start=[self.pose(x,i) for i in range(self.n) for x in [self.u.cubeA,self.u.cubeB,self.u.cubeC]]
VectorHandoff.reset=reset
if __name__=='__main__':
 try:run.main()
 finally:
  out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(exist_ok=True,parents=True);(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(out/'INITIAL_PROVENANCE.json').write_text(json.dumps(dict(scope='RECORDED_INITIAL_STATE_COUNTERFACTUAL_PHYSICS_NOT_RNG_REPLAY',state_setters='only before first action at reset; none between skills',max_readback_errors=errors,sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]),indent=2))
