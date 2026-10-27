"""Assemble public synthetic PhysX evidence, retaining byte hashes of originals.

No field-data directory is traversed. This is a bounded publication packager for
the two videos explicitly selected by the researcher on 2026-10-28.
"""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    w, out = a.workspace.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = {}

    def add(src, target, compress=False):
        data = src.read_bytes()
        rel = str(target) + ('.gz' if compress else '')
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        stored = gzip.compress(data, compresslevel=9, mtime=0) if compress else data
        dest.write_bytes(stored)
        entries[rel] = {'bytes': len(stored), 'sha256': sha(stored),
                        'original_bytes': len(data), 'original_sha256': sha(data),
                        'compression': 'gzip' if compress else 'none'}

    def run(src, target, selected_case=False):
        # Full run arrays/logs are kept, including failures and all evaluated cases.
        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix in {'.py', '.json', '.npz', '.zip'}:
                add(f, Path(target)/f.name, f.suffix == '.json' and f.stat().st_size > 500_000)
        for f in sorted((src/'ppo').glob('*')):
            if f.is_file():
                add(f, Path(target)/'ppo'/f.name)
        for f in sorted((src/'video_raw').glob('*')):
            if f.is_file() and (not selected_case or f.name.startswith('case_001')):
                add(f, Path(target)/'video_raw'/f.name, f.suffix == '.json')
        for f in sorted((src/'video_keyframes').glob('*')):
            if f.is_file() and (not selected_case or f.name.startswith('case_001')):
                add(f, Path(target)/'video_keyframes'/f.name)

    closed = w/'runs/client_sci_closed_loop_20261027'
    compact = w/'runs/compact_training_video_20261028'
    run(compact/'compact_run', 'R3_online_training_12288/run')
    for f in sorted((compact/'delivery').iterdir()):
        if f.is_file():
            add(f, Path('R3_online_training_12288/video')/f.name)
    for side in ['R0_baseline', 'R0_residual41']:
        run(closed/'paired_videos'/side, 'R0_case_001/'+side, selected_case=True)
    for f in sorted((closed/'paired_videos/comparisons_final/R0_case_001').iterdir()):
        if f.is_file():
            add(f, Path('R0_case_001/comparison')/f.name)
    training = w/'runs/contact_handoff_20261016_r1/v6b_residual_seed41'
    for f in sorted(training.iterdir()):
        if f.is_file() and f.suffix in {'.py', '.json', '.npz', '.zip'}:
            add(f, Path('R0_case_001/original_training_seed41')/f.name,
                f.suffix == '.json' and f.stat().st_size > 500_000)
    for f in sorted((training/'ppo').glob('*')):
        if f.is_file():
            add(f, Path('R0_case_001/original_training_seed41/ppo')/f.name)
    add(compact/'release_inputs/test_geometry.json', 'configs/r0_test_geometry.json')
    add(compact/'release_inputs/r0_train.json', 'configs/r0_train.json')
    add(compact/'compact_run/scene.json', 'configs/r3_train.json')
    asset = w/'runs/contact_handoff_20261016_r1/asset_narrow'
    add(asset/'jaka_paddle.usda', 'assets/jaka_paddle.usda', compress=True)
    # Only runtime-required research metadata. Drop unrelated historical field
    # request/scene hashes and unused lineage from the original asset manifest.
    meta = json.loads((asset/'asset_manifest.json').read_text())
    keys = ['asset_sha256', 'body_bases', 'start_q', 'joint_lower', 'joint_upper',
            'held_offset_link6_m', 'paddle_center_body_m', 'paddle_dims_m',
            'level_link6_rotation']
    public = {k: meta[k] for k in keys}
    public.update(scope='SELF_BUILT_SIMULATION_ASSET_RUNTIME_METADATA',
                  contact_grasp_tested=False, whole_chassis_safe=False,
                  collision_approximation=meta['collision_approximation'])
    asset_out = out/'assets/asset_manifest.json'
    asset_out.write_text(json.dumps(public, indent=2)+'\n')
    entries[str(asset_out.relative_to(out))] = {
        'sha256': sha(asset_out.read_bytes()), 'bytes': asset_out.stat().st_size,
        'note': 'Runtime-only public derivative; historical field lineage omitted'}
    tools = w/'repos/ARES/research/skill_handoff_lab/tools'
    for name in ['run_paired_physx_capture.py', 'paired_physx_video_hooks.py',
                 'publication_render_hooks.py', 'build_paired_physx_delivery.py',
                 'export_compact_training_video.py', 'validate_contact_composition.py',
                 'validate_closed_loop_physics.py']:
        add(tools/name, Path('code')/name)
    add(w/'repos/ARES/isaac_sim/simforge/robot/jaka_minicobo_gripper.urdf',
        'assets/jaka_minicobo_gripper.urdf')
    add(w/'repos/ARES-R/src/ares_r/perception/robot_collision.py',
        'code/vendor/ares_r/perception/robot_collision.py')
    add(compact/'run.log', 'R3_online_training_12288/run.log', compress=True)
    # Byte-exact checkpoint reference used in the paired video.
    ck = out/'R0_case_001/original_training_seed41/policy.zip'
    assert sha(ck.read_bytes()) == 'b0597000fa35ad72f0f0dfecb518d2191388a5bcc7cd7d21bdfb5d4f762a5198'
    assert sha((out/'R3_online_training_12288/video/actual_online_training_16env.mp4').read_bytes()) == '78a36b25901f1e53615500f9d65798bab844e891c2e600234af2d7d1afa5200e'
    assert sha((out/'R0_case_001/comparison/paired_actual_physx.mp4').read_bytes()) == '78f40ae24cde3e19cb9d595b05a67f4df097d5d3b3332267fe4a7ba4c454d553'
    (out/'ARTIFACT_SHA256.json').write_text(json.dumps({
        'scope': 'SELECTED_TWO_SYNTHETIC_PHYSX_VIDEOS_ONLY_NO_RESTRICTED_FIELD_DATA',
        'base_git_commit': 'e94ace76ea7151edf99950afce7384a0cf09b4d1',
        'entries': entries}, indent=2)+'\n')
    publication = {str(f.relative_to(out)): sha(f.read_bytes())
                   for f in sorted(out.rglob('*'))
                   if f.is_file() and f.name != 'PUBLICATION_SHA256.json'
                   and '__pycache__' not in f.parts}
    (out/'PUBLICATION_SHA256.json').write_text(json.dumps({
        'scope': 'All published files except this self-referential index',
        'entries': publication}, indent=2)+'\n')
    print(json.dumps({'files': len(entries), 'stored_bytes': sum(e['bytes'] for e in entries.values())}))


if __name__ == '__main__':
    main()
