"""Bundle owned synthetic deliverables and verified models; never field data.

Run after verify_client_delivery.py. The bundle index lives outside its own
archive to avoid a recursive hash. No remote access or source-data deletion.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(workspace):
    repo = workspace / 'repos/ARES'
    root = repo / 'research/skill_handoff_lab'
    package = root / 'deliverables/client_sci_20261027'
    assert json.loads((package / '06_RUNS_MANIFEST/GLOBAL_DELIVERY_QA.json').read_text())['status'] == 'PASS'
    model_index = json.loads((package / '06_RUNS_MANIFEST/MODEL_DELIVERY_INDEX.json').read_text())
    model = Path(model_index['delivery_archive']['path'])
    assert sha(model) == model_index['delivery_archive']['sha256']
    ignored = {'.aux', '.bbl', '.blg', '.log', '.out', '.pyc'}
    selected = {}
    for path in sorted(package.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ignored:
            continue
        if path.name in {'DELIVERY_BUNDLE_INDEX.json', 'PUBLIC_FILE_HASHES.json'}:
            continue
        selected['client_sci_20261027/' + str(path.relative_to(package))] = path
    source_names = [
        'analyze_client_v7_causality.py', 'audit_client_v7_actor.py',
        'audit_client_v7_completed.py', 'contact_composition_v7.py',
        'contact_task_frame_v7.py', 'index_client_v7_costs.py',
        'package_client_v7_models.py', 'publication_render_hooks.py',
        'run_client_v7_queue.py', 'run_contact_v7.py', 'run_publication_render.py',
        'summarize_client_v7.py', 'verify_client_delivery.py', 'bundle_client_delivery.py',
    ]
    for name in source_names:
        selected['research_source/tools/' + name] = root / 'tools' / name
    for path in sorted((root / 'configs/client_sci_v7').glob('*.json')):
        selected['research_source/configs/client_sci_v7/' + path.name] = path
    selected['research_source/tests/test_contact_frame_v7.py'] = root / 'tests/test_contact_frame_v7.py'
    selected['models/MODELS_AND_CONFIGS.tar.gz'] = model
    entries = {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in selected.items()}
    public = {name: record for name, record in entries.items() if name.startswith('client_sci_20261027/')}
    (package / '06_RUNS_MANIFEST/PUBLIC_FILE_HASHES.json').write_text(json.dumps(public, indent=2) + '\n')
    output = workspace / 'runs/client_sci_delivery_20261027'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'ARES_CLIENT_SCI_PACKAGE_20261027.tar.gz'
    manifest = {'scope': 'SYNTHETIC_RESEARCH_ONLY_NO_RESTRICTED_FIELD_DATA',
                'entries': entries, 'excludes': ['LaTeX temporary build files', 'Python caches',
                'raw NPZ/action logs (indexed separately)', 'restricted field data'],
                'note': 'Models include exact per-run source/config snapshots. Current tooling is separately supplied.'}
    with tarfile.open(archive, 'w:gz') as tar:
        for name, path in selected.items():
            tar.add(path, arcname=name, recursive=False)
        content = (json.dumps(manifest, indent=2) + '\n').encode()
        member = tarfile.TarInfo('BUNDLE_CONTENT_HASHES.json')
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    verified = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for name, item in entries.items():
            content = tar.extractfile(name).read()
            assert len(content) == item['bytes'] and hashlib.sha256(content).hexdigest() == item['sha256']
            verified += 1
    index = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
             'verified_regular_members': verified + 1, 'source_entries': entries,
             'scope': manifest['scope'], 'excluded_index': 'DELIVERY_BUNDLE_INDEX.json (no recursive self hash)'}
    (package / '06_RUNS_MANIFEST/DELIVERY_BUNDLE_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({key: index[key] for key in ['path', 'bytes', 'sha256', 'verified_regular_members']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    main(parser.parse_args().workspace.resolve())
