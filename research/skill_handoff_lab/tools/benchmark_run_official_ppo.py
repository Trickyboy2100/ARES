"""Run pinned upstream PPO unchanged, preserving its task, reward and checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--task', required=True)
p.add_argument('--name', required=True)
p.add_argument('--steps', type=int, default=256000)
p.add_argument('--seed', type=int, default=31)
a = p.parse_args()
assert a.task in ['PickCube-v1', 'StackCube-v1', 'PegInsertionSide-v1']
assert 0 < a.steps <= 1024000 and a.steps % 3200 == 0
assert '/' not in a.name
base = a.root / 'runs/benchmark_scout_20261027'
out = base / 'runs' / a.name
out.mkdir(parents=True, exist_ok=False)
source = a.root / 'snapshots/benchmark_scout_20261027/ManiSkill/examples/baselines/ppo/ppo.py'
(out / 'upstream_ppo_source.py').write_bytes(source.read_bytes())
command = [str(a.root / 'envs/maniskill-scout-20261027/bin/python'), '-u', str(source),
           '--env-id', a.task, '--exp-name', a.name, '--seed', str(a.seed),
           '--num-envs', '64', '--num-eval-envs', '4', '--total-timesteps', str(a.steps),
           '--num-steps', '50', '--num-eval-steps', '100' if 'Peg' in a.task else '50',
           '--num-minibatches', '8', '--eval-freq', '10', '--capture-video', '--save-model']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4',
           MS_ASSET_DIR=str(base / 'maniskill_assets'),
           VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json')
for key in ['ALL_PROXY', 'all_proxy']:
    env.pop(key, None)
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    env[key] = 'http://127.0.0.1:18897'
record = {'command': command, 'task': a.task, 'seed': a.seed, 'requested_online_steps': a.steps,
          'upstream_commit': '62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3',
          'upstream_ppo_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'scope': 'OFFICIAL_MANISKILL_PPO_PILOT_NO_ALGORITHM_OR_TASK_MODIFICATION',
          'timeout_seconds': 1800, 'status': 'STARTED'}
(out / 'launch.json').write_text(json.dumps(record, indent=2))
started = time.monotonic()
with (out / 'training.log').open('x') as log:
    child = subprocess.Popen(command, cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = child.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        code = child.returncode
        record['timeout'] = True
record.update(returncode=code, wall_seconds=time.monotonic()-started,
              status='COMPLETE' if code == 0 and (out/'final_ckpt.pt').exists() else 'INCOMPLETE')
record['checkpoint_sha256'] = {str(x.name): hashlib.sha256(x.read_bytes()).hexdigest() for x in out.glob('*.pt')}
(out / 'launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
