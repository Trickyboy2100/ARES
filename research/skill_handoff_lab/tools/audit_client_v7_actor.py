"""Read trusted, own-run SB3 checkpoints: actor Adam-state evidence, not performance."""
import argparse,json,hashlib
from pathlib import Path
import torch
from stable_baselines3 import PPO
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for d in sorted(a.root.iterdir()):
    if not (d/'RUN_COMPLETE.json').exists() or not (d/'policy.zip').exists():continue
    model=PPO.load(d/'policy.zip',device='cpu');tensors=[]
    for name,parameter in model.policy.named_parameters():
        if not (name.startswith('mlp_extractor.policy_net') or name.startswith('action_net') or name=='log_std'):continue
        state=model.policy.optimizer.state.get(parameter,{})
        tensors.append({'name':name,'adam_steps':int(state.get('step',0)),
            'adam_first_moment_norm':float(state['exp_avg'].norm()) if 'exp_avg' in state else 0.,
            'parameter_norm':float(parameter.detach().norm())})
    rows.append({'run':d.name,'checkpoint_sha256':hashlib.sha256((d/'policy.zip').read_bytes()).hexdigest(),
        'actor_tensors':tensors,'actor_has_nonzero_adam_moments':any(t['adam_first_moment_norm']>0 for t in tensors),
        'log_std':model.policy.log_std.detach().cpu().tolist(),
        'caution':'Optimizer/Actor evidence only. Physical effect and qualified task outcome require separate trajectory and intervention audits.'})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2));print(json.dumps({'checkpoints':len(rows),'actor_updated':sum(r['actor_has_nonzero_adam_moments'] for r in rows)}))
