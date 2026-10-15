"""Exhaustive first-decision order test, without modifying frozen cache outcomes."""
import argparse,itertools,json,os
from pathlib import Path
import numpy as np
from sb3_contrib import MaskablePPO
from train_skill_options import make_env
from controlled_paths import require_private_location


def main(a):
    require_private_location(a.output);os.umask(0o077);records=[]
    for file in sorted(a.training.glob('policy_seed*.zip')):
        policy=MaskablePPO.load(file,device='cpu');env=make_env(a.dataset,'test',1)
        seen=set()
        for sample_seed in range(100):
            obs,_=env.reset(seed=sample_seed);task=env.core.backend.task['task_id']
            if task in seen:continue
            seen.add(task);actions=env.core.current.actions;mask=env.action_masks()
            for order in itertools.permutations(range(len(actions))):
                row=dict(obs);row['action_features']=obs['action_features'][list(order)]
                action=int(policy.predict(row,action_masks=mask[list(order)],deterministic=True)[0])
                chosen=actions[order[action]]
                records.append({'policy':file.name,'task_id':task,'order':order,'selected_action_id':chosen.action_id,
                                'is_direct':bool(chosen.features[0]),'profile':'FAST' if chosen.features[3] else 'FALLBACK'})
            if len(seen)==2:break
        env.close()
    a.output.write_text(json.dumps({'test':'all 24 first-decision permutations per test case per seed',
        'decisions':len(records),'direct_fraction':float(np.mean([r['is_direct'] for r in records])),
        'note':'random-order robustness, NOT a proof of permutation-equivariant architecture; no IDs enter actor',
        'records':records},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['dataset','training','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
