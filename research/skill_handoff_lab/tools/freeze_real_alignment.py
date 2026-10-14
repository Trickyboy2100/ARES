"""Inventory already-produced audit artifacts; no inferred labels or hardware I/O."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
a = p.parse_args()
repo = Path(__file__).resolve().parents[3]
source_paths = list((repo / 'isaac_sim/simforge/assets/jaka_minicobo').rglob('*.usd'))
source_paths += list(Path(__file__).resolve().parent.glob('*site_reference.py'))
source_paths += [Path(__file__).resolve().parent / 'real_to_sim_audit.py',
                 Path(__file__).resolve().parent / 'plot_real_alignment.py',
                 repo.parent.parent / 'runs/multicontract_20261014/asset_mesh_colliders_v5/jaka_held.usda']
source_rows = [{'path': str(item), 'bytes': item.stat().st_size,
                'sha256': hashlib.sha256(item.read_bytes()).hexdigest()}
               for item in source_paths if item.is_file()]
rows = []
for item in sorted(a.directory.rglob('*')):
    if item.is_file() and item.name != 'artifact_manifest.json':
        rows.append({'path': str(item.relative_to(a.directory)), 'bytes': item.stat().st_size,
                     'sha256': hashlib.sha256(item.read_bytes()).hexdigest()})
manifest = {'requested_base_commit': '3c20dcd', 'audit_execution_base_commit': 'f5e9eb0',
            'ares_r_read_only_commit': '9de17d2',
            'evidence': 'HISTORICAL/CAD/SCAN_PROXY/OFFLINE_REPLAY/STATIC_RENDER; OCTOBER_UNVERIFIED',
            'october_raw_archive_available': False, 'new_physics_training_steps': 0,
            'additional_model_and_code_sources': source_rows, 'artifacts': rows}
(a.directory / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
print(f'Frozen {len(rows)} artifacts; October originals remain missing.')
