"""Predeclared first-ID documentary selection from complete qualified case tables.

Does not select a checkpoint or a seed. No candidate output feeds into training.
All paired outcomes and source hashes are retained beside the selected IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(run):
    audit = run / 'INDEPENDENT_CLOSED_LOOP.json'
    if not audit.exists():
        audit = run / 'INDEPENDENT_TASK_AND_CONTACT.json'
    outcomes = {row['case_id']: row for row in json.loads((run / 'frozen_evaluation.json').read_text())}
    qualified = {row['case_id']: row for row in json.loads(audit.read_text())['cases']}
    assert outcomes.keys() == qualified.keys()
    return outcomes, qualified, audit


def main(a):
    left, la, lp = read(a.left)
    right, ra, rp = read(a.right)
    assert left.keys() == right.keys()
    categories = {'P1_rescue': [], 'P2_rescue': [], 'P2_common_failure': [], 'P2_common_success': [], 'P2_regression': []}
    rows = []
    for case in sorted(left):
        l, r = left[case], right[case]
        assert l['family'] == r['family'] and l['scene_group'] == r['scene_group']
        lb, rb = la[case]['independent_task_pass'], ra[case]['independent_task_pass']
        family = 'P2' if l['family'] == 'slot' else 'P1'
        key = f'{family}_rescue' if not lb and rb else (
            'P2_common_failure' if family == 'P2' and not lb and not rb else (
                'P2_common_success' if family == 'P2' and lb and rb else (
                    'P2_regression' if family == 'P2' and lb and not rb else None)))
        if key in categories:
            categories[key].append(case)
        rows.append({'case_id': case, 'family': family, 'left_pass': lb, 'right_pass': rb, 'category': key})
    first = {key: (values[0] if values else None) for key, values in categories.items()}
    priority = ['P2_rescue', 'P2_common_failure', 'P2_common_success']
    chosen = next((first[key] for key in priority if first[key] is not None), None)
    result = {'rule': 'minimum case ID in first available P2 rescue/common-failure/common-success category',
              'checkpoint_selection': 'supplied independently; not selected by this script',
              'first_category_case': first, 'default_new_round_case': chosen,
              'all_cases': rows, 'left_audit_sha256': digest(lp), 'right_audit_sha256': digest(rp),
              'left_frozen_sha256': digest(a.left / 'frozen_evaluation.json'),
              'right_frozen_sha256': digest(a.right / 'frozen_evaluation.json'),
              'selector_sha256': digest(Path(__file__))}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(first))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['left', 'right', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
