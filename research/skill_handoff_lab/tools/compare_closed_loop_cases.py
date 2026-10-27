"""Pair independently audited physical cases without treating repeats as new data.

This is descriptive development analysis, not a test-set significance claim.
Terminal costs include timeouts; common-success times are separately reported.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    audit = json.loads((path / 'INDEPENDENT_CLOSED_LOOP.json').read_text())
    episodes = json.loads((path / 'frozen_evaluation.json').read_text())
    return ({e['case_id']: e for e in audit['cases']},
            {e['case_id']: e for e in episodes},
            json.loads((path / 'manifest.json').read_text()))


def duration(e):
    # Snapshot time is global simulation time, NOT time since asynchronous reset.
    if 'episode_time_s' in e['post_step_state']:
        return e['post_step_state']['episode_time_s']
    # v8 uses two 120-Hz physics substeps per control step, frozen in its source.
    return e['steps'] / 60.0


def compare(left, right, output):
    la, le, lm = load(left)
    ra, re, rm = load(right)
    assert la.keys() == ra.keys() == le.keys() == re.keys(), 'Unmatched case identifiers'
    assert lm['evaluation_case_seed'] == rm['evaluation_case_seed']
    assert lm['scene_specs'] == rm['scene_specs'], 'Different physical scenes'
    assert lm['success_contract'] == rm['success_contract'], 'Different task success contracts'
    assert lm['asset_sha256'] == rm['asset_sha256'], 'Different physical assets'
    lc = json.loads((left / 'run_config.json').read_text())
    rc = json.loads((right / 'run_config.json').read_text())
    assert lc['episode_seconds'] == rc['episode_seconds'], 'Different execution time budgets'
    rows = []
    for case in sorted(la):
        a, b, x, y = la[case], ra[case], le[case], re[case]
        init_a, init_b = a['initial_recorded_physical_state'], b['initial_recorded_physical_state']
        assert init_a.keys() == init_b.keys()
        rows.append({
            'case_id': case, 'family': x['family'], 'scene_group': x['scene_group'],
            'left_success': a['independent_task_pass'], 'right_success': b['independent_task_pass'],
            'outcome': ('common_success' if a['independent_task_pass'] else 'rescue')
                if b['independent_task_pass'] else ('regression' if a['independent_task_pass'] else 'common_failure'),
            'left_failure': x['failure_reason'], 'right_failure': y['failure_reason'],
            'left_duration_s': duration(x), 'right_duration_s': duration(y),
            'left_terminal_speed_m_s': a['independent_speed_m_s'],
            'right_terminal_speed_m_s': b['independent_speed_m_s'],
            'left_terminal_error_m': a['independent_goal_distance_m'],
            'right_terminal_error_m': b['independent_goal_distance_m'],
            'left_peak_force_N': x['peak_force_N'], 'right_peak_force_N': y['peak_force_N'],
            'max_initial_recorded_component_difference': max(abs(init_a[k] - init_b[k]) for k in init_a),
        })
    common = [r for r in rows if r['outcome'] == 'common_success']
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'PAIRED_CASES.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    provenance = {}
    for label, path in [('left', left), ('right', right)]:
        provenance[label] = {'run': str(path), 'hashes': {
            name: digest(path / name) for name in ['manifest.json', 'INDEPENDENT_CLOSED_LOOP.json',
                                                  'actual_physics_trace.npz', 'frozen_evaluation.json', 'run_config.json']}}
    report = {
        'evidence': 'PAIRED_INDEPENDENTLY_AUDITED_PHYSX_DEVELOPMENT_EVALUATION',
        'cases': len(rows), 'outcomes': {key: sum(r['outcome'] == key for r in rows)
            for key in ['common_success', 'common_failure', 'rescue', 'regression']},
        'all_case_mean_duration_difference_right_minus_left_s':
            sum(r['right_duration_s'] - r['left_duration_s'] for r in rows) / len(rows),
        'common_success_mean_duration_difference_right_minus_left_s':
            sum(r['right_duration_s'] - r['left_duration_s'] for r in common) / len(common) if common else None,
        'max_initial_recorded_component_difference': max(r['max_initial_recorded_component_difference'] for r in rows),
        'same_task_source': lm['task_source_sha256'] == rm['task_source_sha256'],
        'same_action_dim': lm['action_dim'] == rm['action_dim'],
        'same_observation_dim': lm['observation_dim'] == rm['observation_dim'],
        'same_tool_contract': lm.get('tool_contract') == rm.get('tool_contract'),
        'episode_seconds': lc['episode_seconds'],
        'provenance': provenance, 'analyzer_sha256': digest(Path(__file__)),
        'limitations': ['One training seed is not evidence of cross-seed improvement',
            'First recorded physical states exclude hidden PhysX contact solver state',
            'A mixed-unit maximum initial difference is a diagnostic, not a tolerance certificate',
            'Development cases were used for mechanism selection; not an untouched final test',
            'Common-success time excludes failures and must not replace the complete success table'],
    }
    (output / 'PAIRED_SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['provenance', 'limitations']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['left', 'right', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    compare(a.left.resolve(), a.right.resolve(), a.output.resolve())
