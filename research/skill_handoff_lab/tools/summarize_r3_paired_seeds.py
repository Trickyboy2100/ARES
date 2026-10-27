"""Descriptive paired seed results; repeated cases are not independent seeds."""
import argparse
import json
from pathlib import Path
import statistics

from compare_closed_loop_cases import compare


def main(root, output):
    rows = []
    for seed in (31, 41, 51):
        left = root / f'event_ordinary_seed{seed}_development'
        right = root / f'event_qualified_seed{seed}_development'
        compare(left, right, output / f'r3_seed{seed}_qualified_ablation')
        a = json.loads((left / 'INDEPENDENT_CLOSED_LOOP.json').read_text())
        b = json.loads((right / 'INDEPENDENT_CLOSED_LOOP.json').read_text())
        rows.append({'seed': seed, 'cases': len(a['cases']),
                     'ordinary_pass': a['independent_successes'],
                     'qualified_pass': b['independent_successes'],
                     'difference_cases': b['independent_successes'] - a['independent_successes']})
    stats = {}
    for method in ('ordinary', 'qualified'):
        rates = [r[method + '_pass'] / r['cases'] for r in rows]
        stats[method] = {'mean_success_rate': statistics.mean(rates),
                         'sample_std_across_training_seeds': statistics.stdev(rates),
                         'minimum': min(rates), 'maximum': max(rates),
                         'independent_training_seeds': 3}
    result = {'evidence': 'INDEPENDENTLY_AUDITED_PHYSX_DEVELOPMENT_CASES',
              'rows': rows, 'statistics': stats,
              'mean_paired_case_difference': statistics.mean(r['difference_cases'] for r in rows),
              'limitations': ['Same 32 development cases repeated across three independently trained policies.',
                             'Not 96 independent scenes; not sealed final-test performance.',
                             'Three seeds are insufficient for a strong claim of robustness.',
                             'Full and ordinary methods share physics, observation, execution interface and training budget.']}
    (output / 'R3_PAIRED_SEED_SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
