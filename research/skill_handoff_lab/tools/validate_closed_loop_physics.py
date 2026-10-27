"""Add tool-contract and paired-initial-state checks to the existing OBB audit.

The old result remains intact. The additional report uses actual FK at every
recorded physics step and the terminal post-step state, never commanded poses.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def orientation_errors(actual, reference):
    difference = np.asarray(actual) @ np.asarray(reference).T
    tilt = float(np.arccos(np.clip(difference[2, 2], -1., 1.)))
    yaw = float(abs(np.arctan2(difference[1, 0], difference[0, 0])))
    full = float(np.linalg.norm(Rotation.from_matrix(difference).as_rotvec()))
    return tilt, yaw, full


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    from ares_r.perception.robot_collision import arm_link_transforms
    old = Path(__file__).with_name('validate_contact_composition.py')
    if not (a.run / 'INDEPENDENT_TASK_AND_CONTACT.json').exists():
        spec = importlib.util.spec_from_file_location('original_contact_audit', old)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main(a)
    report = json.loads((a.run / 'INDEPENDENT_TASK_AND_CONTACT.json').read_text())
    data = np.load(a.run / 'actual_physics_trace.npz')
    assert report['trace_sha256'] == sha(a.run / 'actual_physics_trace.npz')
    states, columns = data['states'], list(data['columns'])
    run = json.loads((a.run / 'manifest.json').read_text())
    meta = json.loads((a.asset / 'asset_manifest.json').read_text())
    base = np.array(meta['body_bases']['right'])
    reference = np.array(meta['level_link6_rotation'])
    contract = run.get('tool_contract', {'max_full_rotation_rad': float(np.deg2rad(3.5))})
    contract = dict(contract)
    if contract.get('total_rotation_limit_rad') is not None:
        contract.setdefault('max_full_rotation_rad', contract['total_rotation_limit_rad'])
    contract = {k:v for k,v in contract.items() if v is not None}
    assert any(k in contract for k in ['max_full_rotation_rad', 'max_tilt_rad'])
    if 'max_tilt_rad' in contract:
        assert 'max_yaw_rad' in contract, 'A tilt-only contract silently leaves yaw unbounded'
    maxima, initial, terminal = {}, {}, {}
    allowed = {r['case_id'] for r in report['cases']}
    physical_columns = [*[f'q{i}' for i in range(6)], *[f'qd{i}' for i in range(6)],
                        'obj_x', 'obj_y', 'obj_z', 'obj_qw', 'obj_qx', 'obj_qy', 'obj_qz',
                        'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'paddle_x', 'paddle_y', 'paddle_z']
    indices = [columns.index(c) for c in physical_columns]

    def update(case, q):
        rotation = (base @ arm_link_transforms(a.urdf, q)['link6'])[:3, :3]
        errors = np.array(orientation_errors(rotation, reference))
        maxima[case] = np.maximum(maxima.get(case, np.zeros(3)), errors)

    for step in states:
        for row in step:
            case = int(row[columns.index('case_id')])
            if case not in allowed:
                continue
            if case not in initial:
                initial[case] = dict(zip(physical_columns, map(float, row[indices])))
            update(case, row[:6])
    for record in json.loads((a.run / 'frozen_evaluation.json').read_text()):
        if record['case_id'] not in allowed:
            continue
        post = record.get('post_step_state')
        assert post is not None, 'Exact post-step terminal state is required'
        update(record['case_id'], post['q'])
        terminal[record['case_id']] = post
    for record in report['cases']:
        case = record['case_id']
        tilt, yaw, full = map(float, maxima[case])
        valid = (full <= contract.get('max_full_rotation_rad', float('inf')) + 1e-6
                 and tilt <= contract.get('max_tilt_rad', float('inf')) + 1e-6
                 and yaw <= contract.get('max_yaw_rad', float('inf')) + 1e-6)
        record.update(max_tool_tilt_rad=tilt, max_tool_yaw_rad=yaw,
                      max_tool_full_rotation_rad=full, tool_contract_pass=bool(valid),
                      initial_recorded_physical_state=initial[case],
                      prior_independent_task_pass=record['independent_task_pass'])
        record['independent_task_pass'] = bool(record['independent_task_pass'] and valid)
    report.update(independent_successes=sum(r['independent_task_pass'] for r in report['cases']),
                  tool_contract=contract, tool_reference_rotation=reference.tolist(),
                  additional_validator_sha256=sha(Path(__file__)),
                  inherited_audit_sha256=sha(a.run / 'INDEPENDENT_TASK_AND_CONTACT.json'),
                  initial_state_scope='Recorded cold-reset state; hidden PhysX solver/contact cache not verified')
    report['limitations'].append('Tool yaw extension is a declared research contract, not hardware approval')
    output = a.run / 'INDEPENDENT_CLOSED_LOOP.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'run':str(a.run), 'raw_successes':report['raw_successes'],
                      'independent_successes':report['independent_successes'],
                      'cases':len(report['cases']),
                      'tool_contract_failures':sum(not r['tool_contract_pass'] for r in report['cases'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['run', 'asset', 'urdf']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
