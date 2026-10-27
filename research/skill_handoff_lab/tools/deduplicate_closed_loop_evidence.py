"""Recover space from byte-identical completed evidence, preserving every path.

Dry-run by default. Applies only within this session's non-Git experiment root.
Unique checkpoints, traces, source snapshots, logs and videos are never removed.
Do not mutate immutable evidence after linking; create a new version instead.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def completed(p, root):
    for parent in p.parents:
        if parent == root:
            return False
        if (parent / 'RUN_COMPLETE.json').exists():
            return True
    return False


def main(root, report, apply=False):
    root = root.resolve()
    assert root.name == 'client_sci_closed_loop_20261027' and root.parent.name == 'runs'
    assert not (root / '.git').exists()
    assert not report.exists(), 'Keep earlier cleanup reports immutable'
    candidates = defaultdict(list)
    for p in root.rglob('*'):
        if p.is_symlink() or not p.is_file() or not completed(p, root):
            continue
        allowed = p.name in {'actual_physics_trace.npz', 'action_log.json', 'post_step_contact_history.npz'}
        allowed |= p.suffix == '.zip' and p.name.startswith('policy')
        if allowed and p.stat().st_size > 100_000:
            candidates[p.stat().st_size].append(p)
    operations = []
    for size, paths in sorted(candidates.items()):
        if len(paths) < 2:
            continue
        byhash = defaultdict(list)
        for p in sorted(paths):
            byhash[sha(p)].append(p)
        for digest, group in byhash.items():
            source = group[0]
            for duplicate in group[1:]:
                ss, ds = source.stat(), duplicate.stat()
                if (ss.st_dev, ss.st_ino) == (ds.st_dev, ds.st_ino):
                    continue
                if ds.st_nlink != 1:
                    continue  # Do not count storage still owned by unknown links.
                assert ss.st_dev == ds.st_dev, 'Cross-filesystem link is not supported'
                row = {'source': str(source), 'retained_path': str(duplicate), 'sha256': digest,
                       'logical_bytes': size, 'allocated_bytes_before': ds.st_blocks * 512,
                       'applied': False}
                if apply:
                    assert sha(source) == sha(duplicate) == digest
                    fd, tmpname = tempfile.mkstemp(prefix='.dedup-', dir=duplicate.parent)
                    os.close(fd)
                    os.unlink(tmpname)
                    try:
                        os.link(source, tmpname)
                        os.replace(tmpname, duplicate)
                    finally:
                        if os.path.exists(tmpname):
                            os.unlink(tmpname)
                    assert sha(duplicate) == digest and duplicate.stat().st_ino == source.stat().st_ino
                    row['applied'] = True
                operations.append(row)
    result = {'scope': str(root), 'mode': 'APPLIED_HARDLINK_DEDUPLICATION' if apply else 'DRY_RUN',
              'unique_evidence_removed': 0, 'paths_removed': 0,
              'allocated_bytes_released': sum(x['allocated_bytes_before'] for x in operations if x['applied']),
              'operations': operations,
              'limitation': 'Hardlinked immutable files share storage; never edit them in place.'}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'operations'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    main(a.root, a.report, a.apply)
