"""Index actual completed closed-loop runs and charge observed work once.

This reader never labels a missing audit as failure or reads sealed tests.
The source manifest remains authoritative; callback counts cross-check training.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(root, output):
    rows, details = [], []
    for manifest in sorted(root.rglob('manifest.json')):
        d = manifest.parent
        if not (d / 'RUN_COMPLETE.json').exists():
            continue
        m = json.loads(manifest.read_text())
        if 'online_training_interactions' not in m or 'num_envs' not in m:
            continue
        relative = d.relative_to(root)
        if any(part in {'models', 'history', 'archive'} for part in relative.parts):
            continue
        config_path = d / 'run_config.json'
        config = json.loads(config_path.read_text())
        audit_path = d / 'INDEPENDENT_CLOSED_LOOP.json'
        if not audit_path.exists():
            audit_path = d / 'INDEPENDENT_TASK_AND_CONTACT.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
        episodes = json.loads((d / 'frozen_evaluation.json').read_text())
        actions_path = d / 'action_log.json'
        actual_callback_work = None
        if actions_path.exists():
            actions = json.loads(actions_path.read_text())
            actual_callback_work = sum(len(r['raw']) for r in actions if r.get('mode') == 'train')
            assert actual_callback_work == m['online_training_interactions'], (d, actual_callback_work, m['online_training_interactions'])
        success_ids = sorted(r['case_id'] for r in audit['cases'] if r['independent_task_pass']) if audit else None
        raw_ids = sorted(r['case_id'] for r in episodes if r['success'])
        scenes = d / 'scene.json'
        model = d / 'policy.zip'
        row = {'run': str(relative), 'round': relative.parts[0], 'controller': m['controller'],
               'seed': m['seed'], 'num_envs': m['num_envs'],
               'online_training_interactions': m['online_training_interactions'],
               'training_env_physics_steps': m['training_physics_steps'] * m['num_envs'],
               'all_env_physics_steps': m['env_physics_steps'],
               'evaluation_cases': len(episodes), 'raw_successes': len(raw_ids),
               'independent_successes': len(success_ids) if success_ids is not None else 'UNKNOWN',
               'audit_status': 'COMPLETE' if audit else 'NOT_YET_AUDITED',
               'completed_training_episodes': m['training_episodes'],
               'wall_seconds_excluding_app_start': m['wall_s'],
               'observation_dim': m['observation_dim'], 'action_dim': m['action_dim'],
               'episode_seconds': config['episode_seconds'],
               'feedback_kp': config['feedback_kp'], 'feedback_kd': config['feedback_kd'],
               'policy_mode': config['policy_mode'],
               'recontact_enabled': config.get('recontact_enabled', False),
               'parameter_hold_steps': config.get('parameter_hold'),
               'event_refresh': config.get('event_refresh'),
               'case_seed': m['evaluation_case_seed'], 'split': m['evaluation_role'],
               'scene_sha256': sha(scenes) if scenes.exists() else None,
               'source_sha256': m['task_source_sha256'],
               'policy_sha256': sha(model) if model.exists() else m.get('loaded_policy_sha256'),
               'manifest_sha256': sha(manifest),
               'run_config_sha256': sha(config_path),
               'mean_completion_or_timeout_seconds': sum(
                   r['post_step_state'].get('episode_time_s', r['steps'] * m.get('control_dt_s', 1 / 60))
                   for r in episodes) / len(episodes) if episodes else None}
        assert m['state_writes_during_actions'] == 0, d
        rows.append(row)
        details.append({**row, 'absolute_run':str(d), 'raw_success_case_ids':raw_ids,
                        'independent_success_case_ids':success_ids,
                        'audit_sha256':sha(audit_path) if audit else None,
                        'actual_callback_training_interactions':actual_callback_work,
                        'failure_reasons':{reason:sum(r.get('failure_reason') == reason for r in episodes)
                            for reason in sorted({r.get('failure_reason', 'UNKNOWN') for r in episodes})},
                        'scene_groups':m['scene_specs'],
                        'success_contract':m['success_contract'], 'tool_contract':m.get('tool_contract')})
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / 'COMPLETED_RUNS.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    costkeys = ['online_training_interactions', 'training_env_physics_steps', 'all_env_physics_steps',
                'evaluation_cases', 'completed_training_episodes']
    totals = {key:sum(r[key] for r in rows) for key in costkeys}
    exceptions = []
    failed_work = {key: 0 for key in costkeys}
    unknown_failed_fields = []
    for p in sorted(root.rglob('exception.txt')):
        if (p.parent / 'RUN_COMPLETE.json').exists():
            continue
        budget_path = p.parent / 'OBSERVED_FAILURE_BUDGET.json'
        budget = json.loads(budget_path.read_text()) if budget_path.exists() else None
        if budget:
            for filename, expected in budget.get('evidence', {}).items():
                evidence_path = (p.parent / filename).resolve()
                assert evidence_path.is_relative_to(p.parent.resolve())
                assert sha(evidence_path) == expected, evidence_path
        exceptions.append({'run': str(p.parent.relative_to(root)), 'exception_sha256': sha(p),
                           'observed_budget': budget or 'UNKNOWN',
                           'budget_manifest_sha256': sha(budget_path) if budget else None})
        for key in costkeys:
            if budget is not None and budget.get(key) is not None:
                assert budget[key] >= 0
                failed_work[key] += budget[key]
            else:
                unknown_failed_fields.append({'run': str(p.parent.relative_to(root)), 'field': key})
    result = {'scope':'This session actual synthetic runs, including real evaluation and camera work',
              'completed_run_count':len(rows), 'totals':totals, 'runs':details,
              'incomplete_or_software_failures':exceptions,
              'known_failed_run_workload': failed_work,
              'known_charged_totals_including_failures': {key: totals[key] + failed_work[key] for key in costkeys},
              'unknown_failed_workload_fields': unknown_failed_fields,
              'limitations':['Repeated case evaluations are workload, not independent scene sample sizes',
                            'Simulator startup/warmup is excluded from instrumented physics counters',
                            'No success label is assigned to missing/unfinished audits',
                            'Different scene hashes or action contracts require separately qualified comparisons']}
    (output / 'COMPLETED_RUNS_INDEX.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'completed_runs':len(rows), 'totals':totals, 'incomplete':len(exceptions)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.output.resolve())
