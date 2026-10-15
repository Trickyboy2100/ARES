"""Verify controlled delivery IN MEMORY before immutable, non-Git extraction.

Never publish the extracted payload, robot state, configuration or derived figures.
The public verifier is read and mirrored by stronger path/link checks here, rather
than running its temporary extraction before acceptance. No hardware/network I/O.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import numpy as np
from controlled_paths import require_private_location


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('unsafe archive path')
    return p


def hash_lines(text):
    rows = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        sha, name = line.split(maxsplit=1)
        name = str(checked_name(name.lstrip('*')))
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) or name in rows:
            raise ValueError('invalid or duplicate hash row')
        rows[name] = sha
    return rows


def ingest(archive, destination, index_repo, commit):
    if not __debug__:raise RuntimeError('Run verification without Python -O')
    require_private_location(destination)
    os.umask(0o077)
    if destination.exists():
        raise FileExistsError('Destination exists: never overwrite accepted data')
    # Desktop sandboxes can expose an inert .git permission placeholder. Ask Git
    # whether there is an actual enclosing repository, not merely a directory.
    git_probe = subprocess.run(['git', '-C', str(destination.parent), 'rev-parse', '--show-toplevel'],
                               capture_output=True, text=True)
    if git_probe.returncode == 0:
        raise ValueError('controlled output must be outside Git')
    prefix = 'worklog/evidence/2026-10-12-preland-real-aligned/'
    def public(name):
        return subprocess.check_output(['git', '-C', str(index_repo), 'show', f'{commit}:{prefix}{name}'])
    index = {n: public(n) for n in ['README.md', 'DELIVERY_MANIFEST.json', 'verify_archive.sh',
                                    'REAL_ALIGNED_SOURCE_SHA256SUMS.txt', 'SOURCE_RECORDS_SHA256SUMS.txt']}
    delivery = json.loads(index['DELIVERY_MANIFEST.json'])['full_archive']
    actual = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    assert archive.stat().st_size == delivery['bytes'], 'archive byte count'
    assert actual == delivery['sha256'], 'archive SHA256'
    with gzip.open(archive, 'rb') as stream:
        while stream.read(8 * 1024**2):
            pass  # Consume to EOF to check gzip CRC, including archive padding.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        seen = set()
        files = {}
        for member in members:
            p = checked_name(member.name)
            if member.name in seen:
                raise ValueError('duplicate tar entry')
            seen.add(member.name)
            if not (member.isdir() or member.isfile()):
                raise ValueError('archive links/devices are forbidden')
            if member.isfile() and not any(part.startswith('._') for part in p.parts):
                files[member.name] = member
        top = PurePosixPath(next(iter(files))).parts[0]
        assert all(n.startswith(top + '/') for n in files), 'single archive root'
        assert sum(m.size for m in files.values()) < 4 * 1024**3, 'uncompressed size budget'
        def raw(relative):
            return tar.extractfile(files[top + '/' + relative]).read()
        internal = hash_lines(raw('real_aligned_sources/SHA256SUMS.txt').decode())
        assert len(internal) == 63, 'expected 63 real-aligned source hashes'
        assert internal == hash_lines(index['REAL_ALIGNED_SOURCE_SHA256SUMS.txt'].decode()), 'public/internal hash lists'
        hashes = {n: digest(tar.extractfile(m).read()) for n, m in files.items()}
        checked = []
        for label, rows in [('real_aligned', internal),
                            ('source_records', hash_lines(index['SOURCE_RECORDS_SHA256SUMS.txt'].decode()))]:
            for name, expected in rows.items():
                assert hashes[top + '/' + name] == expected, f'{label} member SHA mismatch'
            checked.append({'manifest': label, 'count': len(rows), 'status': 'PASS'})
        cloud_checks = []
        body_manifests = sorted(n for n in files if '/body_cloud_artifacts/' in n and n.endswith('/manifest.json'))
        assert len(body_manifests) == 3
        for name in body_manifests:
            manifest = json.loads(tar.extractfile(name).read())
            camera_id = PurePosixPath(manifest['source']['path']).parent.name
            camera_prefix = f'real_aligned_sources/raw_camera_captures/{camera_id}'
            camera_bytes = raw(camera_prefix + '/pointcloud_camera_mm.npy')
            camera_manifest = json.loads(raw(camera_prefix + '/manifest.json'))
            assert digest(camera_bytes) == manifest['source']['sha256'] == camera_manifest['sha256']
            camera = np.load(io.BytesIO(camera_bytes), allow_pickle=False).reshape(-1, 3)
            valid = np.isfinite(camera).all(1) & np.any(camera != 0, axis=1)
            camera = camera[valid].astype(np.float64) * .001
            transform = np.array(manifest['T_body_camera'])
            body_npz = np.load(io.BytesIO(tar.extractfile(name.replace('manifest.json', 'body_cloud.npz')).read()), allow_pickle=False)
            body = body_npz['points_body_m']
            assert manifest['frame'] == 'BODY' and manifest['unit'] == 'm'
            assert manifest['source']['unit'] == 'mm'
            assert len(valid) == manifest['raw_point_count'] == camera_manifest['vertex_count']
            assert valid.sum() == manifest['valid_point_count'] == camera_manifest['valid_point_count'] == len(body)
            assert np.allclose(transform, body_npz['T_body_camera'], atol=1e-12, rtol=0)
            predicted = camera @ transform[:3, :3].T + transform[:3, 3]
            error = np.linalg.norm(predicted - body, axis=1)
            assert error.max() < 2e-6, 'CAMERA to BODY mismatch'
            cloud_checks.append({'body_id': PurePosixPath(name).parent.name, 'camera_id': camera_id,
                                 'raw_count': len(valid), 'valid_count': len(body),
                                 'maximum_transform_error_m': float(error.max()),
                                 'transform_revision': manifest['transform_revision'],
                                 'validation_revision': manifest['validation_revision'], 'status': 'PASS'})
        # Acceptance now complete: extract regular payload only, with exclusive writes.
        destination.mkdir(parents=True, mode=0o700)
        inventory = []
        for name, member in sorted(files.items()):
            relative = PurePosixPath(name).relative_to(top)
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            data = tar.extractfile(member).read()
            with output.open('xb') as stream:
                stream.write(data)
            output.chmod(0o400)
            inventory.append({'path': str(relative), 'bytes': len(data), 'sha256': hashes[name]})
    receipt = {'status': 'PASS', 'index_commit': commit, 'archive_sha256': actual,
               'archive_bytes': archive.stat().st_size, 'hash_checks': checked,
               'camera_body_checks': cloud_checks, 'payload_files': inventory,
               'publication': 'CONTROLLED_NOT_PUBLIC', 'mac_metadata': 'ignored, original archive retained',
               'extraction_after_acceptance': True, 'immutability': '0400 files, 0500 dirs; owner-reversible, not WORM'}
    (destination / 'UBUNTU_ACCEPTANCE_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
    (destination / 'UBUNTU_ACCEPTANCE_RECEIPT.json').chmod(0o400)
    for directory in sorted((p for p in destination.rglob('*') if p.is_dir()), reverse=True):
        directory.chmod(0o500)
    destination.chmod(0o500)
    print(json.dumps({k: receipt[k] for k in ['status', 'hash_checks', 'camera_body_checks']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--index-repo', type=Path, required=True)
    p.add_argument('--index-commit', required=True)
    a = p.parse_args()
    ingest(a.archive.resolve(), a.destination.resolve(), a.index_repo.resolve(), a.index_commit)
