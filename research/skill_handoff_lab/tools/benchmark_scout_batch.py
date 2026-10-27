"""Bounded sequential independent scouts; preserve failures, no global pkill."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--jobs', type=Path, required=True)
p.add_argument('--backend', choices=['isaac', 'maniskill'], required=True)
p.add_argument('--gpu', type=int, default=1)
a = p.parse_args()
base = a.root / 'runs/benchmark_scout_20261027'
jobs = json.loads(a.jobs.read_text())
assert 0 < len(jobs) <= 8
python = a.root / ('envs/isaac51-check/bin/python' if a.backend == 'isaac' else 'envs/maniskill-scout-20261027/bin/python')
script = base / 'source' / f'benchmark_{a.backend}_scout.py'
rows = []
for job in jobs:
    assert '/' not in job['name']
    command = [str(python), str(script), '--output', str(base / job['name'])] + job['args']
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES',
               VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
               PYTHONPATH=str(a.root / 'envs/video_encoder_runtime'),
               MS_ASSET_DIR=str(base / 'maniskill_assets'))
    for key in ['ALL_PROXY', 'all_proxy']:
        env.pop(key, None)
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        env[key] = 'http://127.0.0.1:18897'
    if a.backend == 'maniskill':
        env['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    started = time.monotonic()
    with (base / f'{job["name"]}.log').open('x') as log:
        child = subprocess.Popen(command, env=env, cwd=base, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timeout = False
        try:
            code = child.wait(timeout=420)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = child.returncode
    row = {'job': job, 'command': command, 'returncode': code, 'timeout': timeout,
           'wall_seconds': time.monotonic() - started,
           'complete': (base / job['name'] / 'manifest.json').exists()}
    rows.append(row)
    (base / f'batch_{a.jobs.stem}.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(row), flush=True)
