"""Recount finite-node constraints and real update acceptance. No task-gain proxy."""
import argparse,json,hashlib
from pathlib import Path
import torch
from continuation_learner import ContinuationLearner
from program_rule_option import Assets
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);out=[]
for arm,checkpoint in [('witness_soft',128),('witness_qp',128),('witness_margin',64),('witness_project',64)]:
 for seed in [32301,32302]:
  path=a.root/f'{arm}_{seed}'/f'CHECKPOINT_{checkpoint}.pt'
  if not path.exists():continue
  d=torch.load(path,weights_only=False);nodes=d['witness_graph']['nodes'];ss=[n['s']for n in nodes];m=ContinuationLearner(Assets(),seed,arm);m.load(d);base,mask=m.context(ss)
  with torch.no_grad():z=m.logits(m.norm(ss),base,mask,ss);choices=z.argmax(1).tolist()
  logs=d['projections'];out.append(dict(policy=f'{arm}_{seed}',checkpoint=checkpoint,model_sha=hashlib.sha256(path.read_bytes()).hexdigest(),nodes=len(nodes),choice_violations=sum(v not in n['allowed']for n,v in zip(nodes,choices)),chosen_verified_success=sum(v in n['verified']for n,v in zip(nodes,choices)),chosen_fallback_only=sum(v in n['fallback']and v not in n['verified']for n,v in zip(nodes,choices)),projection_attempts=len(logs),zero_steps=sum(x['scale']==0 for x in logs),backtracked_steps=sum(0<x['scale']<1 for x in logs),dual_not_converged=sum(not x.get('dual_converged',True)for x in logs),projection_seconds=sum(x['wall_s']for x in logs),scope='Finite stored nodes only; failed-reference fallback is not successful. No unseen-state safety or full-task guarantee.'))
(a.root/'FINAL_WITNESS_SUPPORT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
