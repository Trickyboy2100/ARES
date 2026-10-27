"""Compare actual pre-update samples and weights; never runs a simulator/update."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch


def compare(left, right):
    name = 'first_real_rollout_gradient_samples.npz'
    a, b = np.load(left / name), np.load(right / name)
    arrays = {k: {'exact_equal': bool(np.array_equal(a[k], b[k])),
                  'max_absolute_difference': float(np.max(np.abs(a[k] - b[k])))}
              for k in a.files}
    def weights(directory):
        with zipfile.ZipFile(directory / 'policy_before_first_update.zip') as archive:
            return torch.load(io.BytesIO(archive.read('policy.pth')),
                              map_location='cpu', weights_only=True)
    wa, wb = weights(left), weights(right)
    identical = wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
    logs = {}
    for label, directory in [('left', left), ('right', right)]:
        rows = json.loads((directory / 'action_log.json').read_text())
        rows = [row for row in rows if row['mode'] == 'train']
        mask = np.concatenate([np.asarray(row['pre_action_execution_mask']) for row in rows])
        mode = np.concatenate([np.asarray(row['pre_action_mode']) for row in rows])
        raw = np.concatenate([np.asarray(row['raw']) for row in rows])
        clipped = np.concatenate([np.asarray(row['clipped']) for row in rows])
        logs[label] = {'online_action_samples': len(mask),
                       'active_samples_per_dimension': mask.sum(0).tolist(),
                       'all_dimensions_inactive_samples': int((mask.sum(-1) == 0).sum()),
                       'recontact_mode_counts': {str(k): int((mode == k).sum()) for k in np.unique(mode)},
                       'clipped_samples_per_dimension': (raw != clipped).sum(0).tolist()}
    return {'evidence': 'ACTUAL_PHYSX_ROLLOUT_AND_PRE_UPDATE_WEIGHTS',
            'left': str(left.resolve()), 'right': str(right.resolve()),
            'first_sample_count': len(a['actions']), 'arrays': arrays,
            'pre_update_policy_state_dict_exact_equal': identical,
            'training_action_usage': logs,
            'sha256': {str(directory / filename): hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                       for directory in (left, right) for filename in
                       (name, 'policy_before_first_update.zip', 'action_log.json')},
            'limitations': ['Only saved first 512 rollout samples are compared, not the entire first rollout.',
                           'Different old log probabilities are expected for full versus marginalized distributions.',
                           'Gradient isolation and action usage do not establish performance or novelty.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
