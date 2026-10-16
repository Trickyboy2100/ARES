"""Audit saved v5 facts; do not manufacture missing historical raw actions."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
import torch

def main(root,output):
    run=root/'contact_v5_boundary_seed31';m=json.loads((run/'manifest.json').read_text())
    episodes=json.loads((run/'training_episodes.json').read_text());audit=json.loads((run/'credit_audits.json').read_text())
    a=PPO.load(run/'policy.zip',device='cpu');b=PPO.load(root/'contact_v5_all_mc_seed31/policy.zip',device='cpu')
    out={'old_task_source_sha256':hashlib.sha256((run/'task_source.py').read_bytes()).hexdigest(),
         'source_matches_manifest':hashlib.sha256((run/'task_source.py').read_bytes()).hexdigest()==m['task_source_sha256'],
         'method_snapshot_present':(run/'boundary_credit_source.py').exists(),
         'complete_training_episodes':len(episodes),'training_successes':sum(e['success'] for e in episodes),
         'first16_final_phases':[e['phase'] for e in episodes[:16]],
         'actual_credit_audits':audit,'modified_samples':sum(x['mixed_actor_samples'] for x in audit),
         'total_training_samples':m['online_training_interactions'],
         'mix_coverage':sum(x['mixed_actor_samples'] for x in audit)/m['online_training_interactions'],
         'final_action_std':a.policy.log_std.exp().detach().tolist(),
         'boundary_and_all_policy_parameters_identical':all(torch.equal(v,b.policy.state_dict()[k]) for k,v in a.policy.state_dict().items()),
         'window_episode_accounting':[],'limits':['Missing old preclip training actions cannot be reconstructed exactly',
             'A missing historical method snapshot is a provenance gap; do not claim a saved hash that did not exist']}
    previous={i:0 for i in range(16)}
    for e in episodes:
        end=e['env_interactions']//16;start=previous[e['env']];previous[e['env']]=end
        out['window_episode_accounting'].append({'env':e['env'],'start_control_step':start,'end_control_step':end,
            'same_512_window':start//512==(end-1)//512,'end_phase':e['phase']})
    output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='window_episode_accounting'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
