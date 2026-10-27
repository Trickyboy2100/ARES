"""Run a preserved original training task with only relocated runtime paths."""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
a = p.parse_args()
cfg = json.loads(a.config.read_text())
out = Path(cfg['output'])
out.mkdir(parents=True, exist_ok=False)
for src in a.source.parent.glob('*.py'):
    shutil.copyfile(src, out/src.name)
shutil.copyfile(a.source.parent/'boundary_credit_source.py', out/'contact_credit_v2.py')
for key in ['asset', 'output', 'scenario_file', 'load_policy', 'legacy_policy']:
    if cfg.get(key):
        cfg[key] = Path(cfg[key])
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, device=f"cuda:{cfg['gpu']}", enable_cameras=not cfg['skip_capture'],
                       kit_args=f"--/renderer/activeGpu={cfg['gpu']} --/renderer/multiGpu/enabled=false")
try:
    sys.path.insert(0, str(out))
    spec = importlib.util.spec_from_file_location('preserved_training', out/a.source.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(argparse.Namespace(**cfg))
finally:
    launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
