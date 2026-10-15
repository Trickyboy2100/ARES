"""Relocate verified field requests, preserving all original planner constraints."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import yaml
from controlled_paths import require_private_location


def main(a):
    require_private_location(a.output)
    os.umask(0o077)
    if subprocess.run(['git', '-C', str(a.output.parent), 'rev-parse', '--show-toplevel'], capture_output=True).returncode == 0:
        raise ValueError('controlled requests cannot be written inside Git')
    a.output.mkdir(mode=0o700, exist_ok=False)
    cfg = yaml.safe_load(a.asset_yaml.read_text())
    robot_urdf = a.remote_robot_root + '/jaka_minicobo_gripper.urdf'
    cfg['kinematics']['urdf_path'] = robot_urdf
    cfg['kinematics']['asset_root_path'] = a.remote_robot_root
    if a.sixaxis_adapter:
        kin = cfg['kinematics']
        kin['tool_frames'] = ['link6']
        kin['lock_joints'] = {'4C2_Joint1': 0.0}
        names = kin['cspace']['joint_names']
        indexes = [names.index(f'joint{i}') for i in range(1, 7)]
        for key, value in list(kin['cspace'].items()):
            if isinstance(value, list) and len(value) == len(names):
                kin['cspace'][key] = [value[i] for i in indexes]
        # Collision spheres still come from the complete frozen request.
        # Meshes remain in the ORIGINAL URDF; only optional cuRobo display mesh loading is disabled.
        kin['load_meshes'] = False
        kin['mesh_link_names'] = None
    (a.output / 'robot.yml').write_text(yaml.safe_dump(cfg))
    jobs, manifest = [], []
    for i, relative in enumerate(a.select):
        source = a.root / relative / 'planner_request.json'
        original = json.loads(source.read_text()); relocated = copy.deepcopy(original)
        expected = original['collision_model']['asset_sha256'][a.asset_yaml.name]
        assert hashlib.sha256(a.asset_yaml.read_bytes()).hexdigest() == expected
        relocated['robot_yaml_urdf'] = robot_urdf
        relocated['robot_yaml'] = a.remote_directory + '/robot.yml'
        assert relocated['planning_only'] is True and relocated['execution_allowed'] is False
        name = f'request_{i}.json'
        (a.output / name).write_text(json.dumps(relocated, indent=2))
        jobs.append(a.remote_directory + '/' + name)
        manifest.append({'source_relative': str(relative), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'changed_keys': ['robot_yaml_urdf', 'robot_yaml'], 'constraints_changed': False,
                         'budget_changed': False,
                         'planner_yaml': 'RECONSTRUCTED_SIX_AXIS_ADAPTER_NOT_FROZEN_ONSITE' if a.sixaxis_adapter else 'ORIGINAL_PUBLIC_ASSET_WITH_RELOCATED_PATHS',
                         'adapter': 'link6 tool frame, six arm cspace, finger locked at zero; opening UNKNOWN; original URDF retained' if a.sixaxis_adapter else None,
                         'new_request_sha256': hashlib.sha256((a.output/name).read_bytes()).hexdigest()})
    (a.output/'batch.json').write_text(json.dumps(jobs, indent=2))
    (a.output/'RELOCATION_MANIFEST.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--select',action='append',required=True)
    p.add_argument('--asset-yaml',type=Path,required=True);p.add_argument('--remote-directory',required=True)
    p.add_argument('--remote-robot-root',required=True);p.add_argument('--sixaxis-adapter',action='store_true');main(p.parse_args())
