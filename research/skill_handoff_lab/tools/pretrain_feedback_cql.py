"""Known conservative offline warmstart diagnostic; no physical simulation."""
import argparse,json,shutil
from pathlib import Path
import torch
from program_budget_feedback_sac import FeedbackSAC,normalization
p=argparse.ArgumentParser();p.add_argument('--warm',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
data=json.loads((a.warm/'TRANSITIONS.json').read_text());labels=json.loads((a.warm/'LABELS.json').read_text())
for mode in ['ordinary','budget','suffix']:
 m=FeedbackSAC(mode,31501,*normalization(data),recipe='cql_warm');m.train(data,labels,1000,bc=True);m.save(a.out/f'{mode}.pt');print(mode,m.history[-1],flush=True)
shutil.copy2(Path(__file__).parent/'program_budget_feedback_sac.py',a.out/'SOURCE.py')
