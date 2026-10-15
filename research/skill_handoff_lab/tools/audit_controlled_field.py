"""Private October scene binding and independent modeled trajectory replay.

Outputs (including figures) are CONTROLLED, outside Git. No robot interfaces,
configuration mutation, replacement success labels, or training are involved.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from controlled_paths import require_private_location
from ares_r.perception.robot_collision import arm_link_transforms
from ares_r.motion.independent_path_validation import validate_dense_world
from ares_r.motion.tcp_orientation import validate_level_path
from ares_r.motion.native_execution_package import package_native_preview


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def private_output(path):
    require_private_location(path)
    if subprocess.run(['git', '-C', str(path.parent), 'rev-parse', '--show-toplevel'],
                      capture_output=True).returncode == 0:
        raise ValueError('controlled analysis cannot be written inside Git')
    path.mkdir(mode=0o700, parents=True, exist_ok=False)


def run(root, output, urdf):
    os.umask(0o077)
    private_output(output)
    receipt = read(root / 'UBUNTU_ACCEPTANCE_RECEIPT.json')
    assert receipt['status'] == 'PASS'
    sources = root / 'real_aligned_sources'
    config = sources / 'config_snapshot/site_config_20261014'
    limits = read(config / 'jaka_mini2_motion.site.json')
    lower = np.array(limits['lower_rad']) + limits['soft_limit_margin_rad']
    upper = np.array(limits['upper_rad']) - limits['soft_limit_margin_rad']
    scenes = {p.parent.parent.name: p for p in (sources / 'scenes').glob('*/scene/snapshot.json')}
    requests = sorted(root.rglob('planner_request.json'))
    records, replays, scene_rows = [], [], []
    for sid, snapshot_path in scenes.items():
        snapshot = read(snapshot_path)
        cloud_id = snapshot['environment']['observation']['pointcloud']['pointcloud_id']
        manifest_path = sources / 'body_cloud_artifacts' / cloud_id / 'manifest.json'
        manifest = read(manifest_path)
        scene_revision = dict(snapshot['environment']['observation']['calibration']['revisions'])
        assert scene_revision['T_body_camera'] == manifest['transform_revision']
        assert scene_revision['T_body_camera_validation'] == manifest['validation_revision']
        geometry = read(snapshot_path.parent.parent / 'whole_robot_geometry.json')
        arm_audits = {side: read(snapshot_path.parent.parent / f'{side}_fk_audit.json') for side in ['left', 'right']}
        scene_rows.append({'scene_id': sid, 'snapshot_sha256': sha(snapshot_path), 'body_manifest_sha256': sha(manifest_path),
                           'cloud_id': cloud_id, 'transform_revision': manifest['transform_revision'],
                           'raw_capture_sha256': manifest['source']['sha256'],
                           'state': snapshot['robot_state'], 'tool_readbacks': {s: d['diagnostics']['tool_data'] for s, d in arm_audits.items()},
                           'sdk_fk_max_errors': {s: {'mm': d['max_position_error_mm'], 'deg': d['max_orientation_error_deg']} for s, d in arm_audits.items()},
                           'whole_robot_geometry_sha256': sha(snapshot_path.parent.parent / 'whole_robot_geometry.json'),
                           'geometry_boxes': len(geometry['boxes']), 'request_ids': [],
                           'note': 'Scene targets.json A/B are diagnostic joint-delta goals, not automatically EPIC Preland.'})
    for path in requests:
        request = read(path)
        result_path = path.with_name('planning.json')
        result = read(result_path) if result_path.exists() else {}
        sid = request.get('scene_snapshot_id')
        points = np.array(result.get('trajectory_points_rad') or [], dtype=float)
        constraints = request.get('motion_constraints') or {}
        lock = request.get('orientation_lock') or {}
        components = {k: request.get(k) for k in ['compiled_scene', 'T_body_model', 'T_link6_tcp', 'collision_model',
                     'execution_tool_envelope', 'attached_object_collision', 'orientation_lock', 'motion_constraints', 'planner_clearance_policy']}
        # Fair geometry/contract key excludes start/goal; comparison must ALSO
        # hold starts, goals, budgets and RNG fixed. No grouping creates labels.
        row = {'request_id': str(path.parent.relative_to(root)), 'request_sha256': sha(path), 'scene_id': sid,
               'scene_payload_available': sid in scenes, 'compiled_world_sha256': fingerprint(request.get('compiled_scene')),
               'component_hashes': {k: fingerprint(v) for k, v in components.items()},
               'geometry_contract_fingerprint': fingerprint(components),
               'start_sha256': fingerprint(request.get('start_rad')), 'goal_sha256': fingerprint(request.get('runtime_motion_goal') or request.get('goal_rad')),
               'budget': request.get('planning_parameters'), 'planner_reported': result.get('observed_result', 'UNKNOWN'),
               'trajectory_knots': len(points), 'attached': request.get('attached_object_collision') is not None,
               'central_exclusion': constraints.get('central_exclusion'), 'orientation_policy': lock.get('policy'),
               'goal': request.get('runtime_motion_goal'), 'tool': request.get('controller_tool_pose_mm_rad'),
               'independently_validated': 'NOT_TESTED', 'physical_execution': 'NOT_TESTED'}
        if sid in scenes:
            entry = next(x for x in scene_rows if x['scene_id'] == sid)
            snapshot = read(scenes[sid])
            assert request['compiled_scene']['scene_snapshot_id'] == sid
            obs = snapshot['environment']['observation']
            calibration = request['compiled_scene']['calibration_revision']
            assert calibration['T_body_camera'] == dict(obs['calibration']['revisions'])['T_body_camera']
            entry['request_ids'].append(row['request_id'])
            row['snapshot_to_request_start_delta_rad'] = float(np.max(np.abs(np.array(request['start_rad']) - snapshot['robot_state']['right_joints_rad'])))
            row['scene_binding'] = 'SNAPSHOT_CLOUD_CALIBRATION_VERIFIED'
        if len(points):
            row['reference_soft_limit_check'] = {'config_date': '2026-10-14 REFERENCE_NOT_OCT10_BYTE_FREEZE',
                'margin_rad': limits['soft_limit_margin_rad'], 'minimum_margin_rad': float(np.minimum(points - lower, upper - points).min()),
                'J3_min_rad': float(points[:, 2].min()), 'J3_max_rad': float(points[:, 2].max()),
                'J3_upper_rad': float(upper[2]), 'J3_violation_knots': int((points[:, 2] > upper[2]).sum())}
        records.append(row)
        # Exactly one predeclared successful trajectory per bound scene:
        selected = (path.parent.name == 'transit_plan' or
                    (sid in scenes and lock.get('policy') == 'LEVEL_YAW_FREE_V1') or
                    (sid in scenes and request.get('runtime_motion_goal') is not None))
        if not selected or result.get('observed_result') != 'SUCCESS' or len(points) < 2:
            continue
        expected_urdf = request['collision_model']['asset_sha256'][urdf.name]
        assert sha(urdf) == expected_urdf, 'URDF mismatch: no replay on unverified model'
        local = copy.deepcopy(request)
        local['robot_yaml_urdf'] = str(urdf.resolve())
        start = time.perf_counter()
        validation = validate_dense_world(points.tolist(), local, subdivisions=2)
        base, tool = np.array(request['T_body_model']), np.array(request['T_link6_tcp'])
        def fk(q):
            return base @ arm_link_transforms(urdf, q)['link6'] @ tool
        orientation = validate_level_path(fk, points, subdivisions=4, tolerance_deg=lock.get('max_error_deg', 3.),
                                          final_yaw_rad=lock.get('final_yaw_target_rad'))
        tcp = np.array([fk(q)[:3, 3] for q in points])
        stored_tcp = np.array(result.get('tcp_path_body_m') or [])
        native = {'basis': 'October14 reference config + local readonly packager; NOT live execution authorization'}
        try:
            _, audit = package_native_preview(points.tolist(), float(request['planning_parameters'].get('interpolation_dt', .008)),
                limits, tool_id=1, controller_tool_pose_mm_rad=request['controller_tool_pose_mm_rad'], captured_at_unix=0)
            native.update(status='PASS_REFERENCE_OFFLINE_PACKAGING', audit=audit)
        except Exception as error:
            native.update(status='REJECTED_REFERENCE_OFFLINE_PACKAGING', error=f'{type(error).__name__}:{error}')
        joint_limits_ok = row['reference_soft_limit_check']['minimum_margin_rad'] >= 0
        row['independently_validated'] = {'modeled_world': validation['collision_free'],
            'orientation': orientation['passed'], 'reference_soft_limits': joint_limits_ok,
            'independent_self_collision': validation.get('self_collision_independently_checked'),
            'full_execution_qualification': 'NOT_CERTIFIED'}
        replay = {'request_id': row['request_id'], 'request_sha256': sha(path), 'source_result_sha256': sha(result_path),
                  'source_ares_r_worker_commit': result.get('source_commit'),
                  'replay_ares_r_code_commit': subprocess.check_output(['git', '-C', str(Path(__import__('ares_r.motion.independent_path_validation', fromlist=['x']).__file__).resolve().parents[3]), 'rev-parse', 'HEAD'], text=True).strip(),
                  'independent_validator_sha256': sha(Path(__import__('ares_r.motion.independent_path_validation', fromlist=['x']).__file__)),
                  'world': validation, 'orientation': orientation, 'reference_native': native,
                  'tcp_replay_max_error_m': float(np.linalg.norm(tcp - stored_tcp, axis=1).max()) if stored_tcp.shape == tcp.shape else None,
                  'wall_s': time.perf_counter() - start, 'real_robot_execution': False, 'physics_simulation': False}
        replays.append(replay)
        np.savez_compressed(output / f'replayed_path_{len(replays)}.npz', joints_rad=points, tcp_body_m=tcp)
        print(json.dumps({'replay': row['request_id'], 'collision_free': validation['collision_free'],
                          'orientation': orientation['passed'], 'soft_limits': joint_limits_ok,
                          'native': native['status']}, ensure_ascii=False), flush=True)
    report = {'publication': 'CONTROLLED_NOT_PUBLIC', 'archive_sha256': receipt['archive_sha256'],
              'source_data_immutable': True, 'configuration_note': 'Oct14 reference, not Oct10 dirty config freeze',
              'scenes': scene_rows, 'requests': records, 'independent_replays': replays,
              'request_count': len(records), 'new_gpu_planning': False, 'new_physx_execution': False}
    (output / 'FIELD_BINDINGS_AND_REPLAY.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'Bound {len(scene_rows)} real scene epochs; audited {len(records)} requests; replayed {len(replays)} paths.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--urdf', type=Path, required=True)
    a = p.parse_args()
    run(a.root.resolve(), a.output.resolve(), a.urdf.resolve())
