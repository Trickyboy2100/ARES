"""CPU-load our completed v8 checkpoints and verify their frozen source hashes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    torch.set_num_threads(1)
    rows = []
    markers = list(root.glob('r*/*/RUN_COMPLETE.json')) + list(root.glob('paired_videos/*/RUN_COMPLETE.json'))
    for marker in sorted(markers):
        run = marker.parent
        manifest = json.loads((run / 'manifest.json').read_text())
        steps = manifest.get('online_training_interactions', 0)
        if not steps:
            continue
        source_map = {'task_source.py': 'task_source_sha256',
                      'boundary_credit_source.py': 'credit_source_sha256',
                      'frame_source.py': 'frame_source_sha256',
                      'geometry_source.py': 'geometry_source_sha256'}
        for filename, key in source_map.items():
            if key in manifest:
                assert digest(run / filename) == manifest[key], (run, filename)
        for filename, expected in manifest.get('dependency_sha256', {}).items():
            assert digest(run / filename) == expected, (run, filename)
        model = PPO.load(run / 'policy.zip', device='cpu')
        assert model.num_timesteps == steps
        assert model.observation_space.shape == (manifest['observation_dim'],)
        assert model.action_space.shape == (manifest['action_dim'],)
        samples = run / 'first_real_rollout_gradient_samples.npz'
        obs = np.load(samples)['observations'][:2] if samples.exists() else np.zeros((2, manifest['observation_dim']), np.float32)
        actions, _ = model.predict(obs, deterministic=True)
        assert np.isfinite(actions).all()
        rows.append({'run': str(run), 'checkpoint_sha256': digest(run / 'policy.zip'),
                     'source_sha256': manifest['task_source_sha256'],
                     'online_interactions': steps, 'observation_dim': manifest['observation_dim'],
                     'action_dim': manifest['action_dim'], 'cpu_load_and_inference': 'PASS',
                     'parameters': sum(p.numel() for p in model.policy.parameters())})
    return {'status': 'PASS', 'models': rows,
            'limitations': ['CPU inference/source consistency checks are not a physics-performance test.',
                           'Diagnostic and visualization training checkpoints are not formal method replicates.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'models': len(result['models'])}))
