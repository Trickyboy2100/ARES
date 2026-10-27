"""Verify/extract archived evidence or run a new independent PhysX experiment."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT/'ARTIFACT_SHA256.json').read_text())
    for rel, entry in manifest['entries'].items():
        data = (ROOT/rel).read_bytes()
        assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'], rel
        if entry.get('compression') == 'gzip':
            raw = gzip.decompress(data)
            assert len(raw) == entry['original_bytes'] and hashlib.sha256(raw).hexdigest() == entry['original_sha256'], rel
    publication = ROOT/'PUBLICATION_SHA256.json'
    if publication.exists():
        for rel, digest in json.loads(publication.read_text())['entries'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    print(f"PASS: {len(manifest['entries'])} archived artifacts, including decompressed original hashes and publication metadata")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'extract', 'r3', 'r0', 'r0-train', 'encode-r3', 'compose-r0', 'audit'])
    parser.add_argument('--stage', type=Path, help='New extraction directory outside the repository')
    parser.add_argument('--python', default=sys.executable, help='Isaac 5.1 + Isaac Lab 2.3 runtime for simulation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ffmpeg', default='/usr/bin/ffmpeg')
    parser.add_argument('--execute', action='store_true', help='Actually execute printed commands (single GPU)')
    a = parser.parse_args()
    manifest = verify()
    if a.action == 'verify':
        return
    if a.stage is None:
        parser.error('--stage is required')
    stage = a.stage.resolve()
    if a.action == 'extract':
        stage.mkdir(parents=True, exist_ok=False)
        for rel, entry in manifest['entries'].items():
            target = stage/(rel[:-3] if entry.get('compression') == 'gzip' else rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (ROOT/rel).read_bytes()
            target.write_bytes(gzip.decompress(data) if entry.get('compression') == 'gzip' else data)
        print('Extracted to', stage)
        return
    assert (stage/'assets/jaka_paddle.usda').exists(), 'Run extract first'
    commands = []
    work = stage/'new_runs'
    code = stage/'code'
    asset = stage/'assets'
    vendor = code/'vendor'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(vendor)+os.pathsep+env.get('PYTHONPATH', '')

    def cfg(src, scene, name, policy=None):
        data = json.loads((src/'run_config.json').read_text())
        data.update(asset=str(asset), scenario_file=str(stage/'configs'/scene),
                    output=str(work/name), gpu=a.gpu, load_policy=str(policy) if policy else None)
        # Relocated copy is deliberately distinct from the preserved original.
        path = stage/'relocated_configs'/f'{name}.json'
        if a.execute:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2)+'\n')
        return path

    def capture(src, config, name, label, extra):
        return [a.python, str(code/'run_paired_physx_capture.py'), '--source', str(src/'base_task_source.py'),
                '--config', str(config), '--output', str(work/name), '--gpu', str(a.gpu),
                '--ffmpeg', a.ffmpeg, '--label', label] + extra

    if a.action == 'r3':
        src = stage/'R3_online_training_12288/run'
        config = cfg(src, 'r3_train.json', 'r3_compact')
        commands.append(capture(src, config, 'r3_compact', 'COMPACT_LAYOUT_ONLINE_TRAINING_RERUN',
            ['--training-matrix', '--online-training-steps', '12288', '--eval-episodes', '4',
             '--env-spacing', '0.85', '--compact-camera']))
    elif a.action == 'r0':
        for side, label in [('R0_baseline', 'Historical_tuned_damped_feedback'),
                            ('R0_residual41', 'Historical_residual_seed41')]:
            src = stage/'R0_case_001'/side
            policy = stage/'R0_case_001/original_training_seed41/policy.zip' if side.endswith('41') else None
            config = cfg(src, 'r0_test_geometry.json', side, policy)
            commands.append(capture(src, config, side, label, ['--cases', '1']))
    elif a.action == 'r0-train':
        src = stage/'R0_case_001/original_training_seed41'
        config = cfg(src, 'r0_train.json', 'r0_train_seed41')
        commands.append([a.python, str(ROOT/'run_snapshot.py'), '--source', str(src/'task_source.py'),
                         '--config', str(config)])
    elif a.action == 'encode-r3':
        commands.append([a.python, str(code/'export_compact_training_video.py'), '--run',
            str(stage/'R3_online_training_12288/run'), '--output', str(work/'reencoded_r3')])
    elif a.action == 'compose-r0':
        # Recompose the immutable ORIGINAL raw streams; not a new physics run.
        release = work/'recomposed_r0_release'
        if a.execute:
            for name in ['tables', 'figures']:
                (release/name).mkdir(parents=True, exist_ok=True)
        commands.append([a.python, str(stage/'R0_case_001/comparison/composition_source.py'),
            '--left', str(stage/'R0_case_001/R0_baseline'), '--right', str(stage/'R0_case_001/R0_residual41'),
            '--output', str(work/'recomposed_r0'), '--release', str(release), '--iteration', 'R0',
            '--cases', '1', '--left-label', 'Feedback (historical)',
            '--right-label', 'Residual PPO (historical seed41)'])
    elif a.action == 'audit':
        for run in ['R0_case_001/R0_baseline', 'R0_case_001/R0_residual41', 'R3_online_training_12288/run']:
            commands.append([a.python, str(code/'validate_contact_composition.py'), '--run', str(stage/run),
                             '--asset', str(asset), '--urdf', str(asset/'jaka_minicobo_gripper.urdf')])
    for cmd in commands:
        import shlex
        print(shlex.join(cmd), flush=True)
        if a.execute:
            subprocess.run(cmd, check=True, env=env)
    if not a.execute:
        print('Plan only. Add --execute to run. Extraction and configs do not modify archived evidence.')


if __name__ == '__main__':
    main()
