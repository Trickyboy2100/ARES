"""Create a new immutable synthetic-research bundle, never overwrite v7 delivery."""
import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def bundle(workspace, output):
    root = workspace / 'repos/ARES/research/skill_handoff_lab'
    package = root / 'deliverables/client_sci_20261027'
    runs = workspace / 'runs/client_sci_closed_loop_20261027'
    assert runs.is_dir() and output.parent.resolve().is_relative_to(workspace / 'runs')
    assert not output.exists(), 'Do not overwrite an earlier delivery'
    chosen = {}
    excluded_suffixes = {'.aux', '.bbl', '.blg', '.log', '.out', '.pyc'}
    for path in package.rglob('*'):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and path.suffix not in excluded_suffixes:
            if path.name not in {'CLOSED_LOOP_BUNDLE_INDEX.json', 'PUBLIC_FILE_HASHES.json', 'DELIVERY_BUNDLE_INDEX.json'}:
                chosen['documents/' + str(path.relative_to(package))] = path
    # Explicitly scoped synthetic P1/P2 runs only. No real_aligned data traversal.
    markers = list(runs.glob('r*/*/RUN_COMPLETE.json')) + list(runs.glob('paired_videos/*/RUN_COMPLETE.json'))
    for marker in markers:
        run = marker.parent
        for path in run.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix in {'.py', '.json', '.zip'} and path.name != 'action_log.json':
                chosen['experiments/' + str(path.relative_to(runs))] = path
        for filename in ('actual_physics_trace.npz', 'post_step_contact_history.npz',
                         'first_real_rollout_gradient_samples.npz', 'TRAINING_120HZ_STATE.npz'):
            trace = run / filename
            if trace.exists():
                # Full-observation action logs stay indexed in the raw research archive.
                chosen['experiments/' + str(trace.relative_to(runs))] = trace
    for failure in runs.rglob('exception.txt'):
        chosen['failures/' + str(failure.relative_to(runs))] = failure
        for path in failure.parent.rglob('*'):
            if path.is_file() and not path.is_symlink() and (
                path.suffix in {'.json', '.py', '.csv'} or path.name == 'first_real_rollout_gradient_samples.npz'
            ) and path.name != 'action_log.json':
                chosen['failures/' + str(path.relative_to(runs))] = path
    video_root = runs / 'paired_videos/comparisons_final'
    assert video_root.is_dir(), 'Final comparison directory must exist'
    for path in video_root.rglob('*.mp4'):
        chosen['videos/' + str(path.relative_to(video_root))] = path
    for path in (root / 'configs/client_sci_v8').rglob('*.json'):
        chosen['source/configs/' + str(path.relative_to(root / 'configs'))] = path
    for folder in ('tools', 'tests'):
        for path in (root / folder).glob('*.py'):
            chosen['source/' + folder + '/' + path.name] = path
    entries = {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(chosen.items())}
    manifest = {'scope': 'SELF_BUILT_P1_P2_PHYSX_ONLY_NO_RESTRICTED_FIELD_DATA',
                'entries': entries,
                'limitations': ['Includes historical deliverable material explicitly labeled historical.',
                               'Full action observations remain in the local raw run archive, indexed by hash.',
                               'Preliminary development evaluation is not a sealed final-test claim.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as tar:
        for name, path in sorted(chosen.items()):
            tar.add(path, arcname=name, recursive=False)
        content = (json.dumps(manifest, indent=2) + '\n').encode()
        item = tarfile.TarInfo('CONTENT_SHA256.json'); item.size = len(content)
        tar.addfile(item, io.BytesIO(content))
        links = ['<h1>ARES SCI research: actual evidence</h1>',
                 '<p>Synthetic P1/P2 PhysX research. Preliminary results; no claim of new-method superiority.</p>',
                 '<p><a href="documents/01_CLIENT_PROPOSAL_CN.pdf">Chinese proposal PDF</a> | '
                 '<a href="documents/02_PAPER_SEED/main.pdf">Paper PDF</a> | '
                 '<a href="documents/07_MENTOR_BRIEF_CN.pdf">Mentor brief</a></p>']
        for name in chosen:
            if name.startswith('videos/'):
                safe = html.escape(name, quote=True)
                links.append(f'<h2>{safe}</h2><video controls preload="none" width="960" src="{safe}"></video>')
        content = ('<!doctype html><meta charset="utf-8"><title>ARES research delivery</title>'
                   '<style>body{font:18px system-ui;max-width:1100px;margin:40px auto}video{max-width:100%}</style>'
                   + '\n'.join(links)).encode()
        item = tarfile.TarInfo('OPEN_ME.html'); item.size = len(content)
        tar.addfile(item, io.BytesIO(content))
    with tarfile.open(output, 'r:gz') as tar:
        for name, entry in entries.items():
            content = tar.extractfile(name).read()
            assert len(content) == entry['bytes']
            assert hashlib.sha256(content).hexdigest() == entry['sha256']
    index = {'archive': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output),
             'verified_members': len(entries), 'scope': manifest['scope']}
    (package / '08_ITERATIVE_RESEARCH/CLOSED_LOOP_BUNDLE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle(args.workspace.resolve(), args.output.resolve())
