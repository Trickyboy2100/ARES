"""Archive finalized offline baseline runs and verify every stored member."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archives = []
    for run in sorted(args.root.iterdir()):
        if not run.is_dir() or not (run / 'RESULTS.json').exists():
            continue  # Never publish an incomplete model as a finished run.
        files = sorted(p for p in run.rglob('*') if p.is_file())
        members = {str(p.relative_to(args.root)): sha(p) for p in files}
        archive = args.out / (run.name + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as target:
            for path in files:
                target.add(path, arcname=str(path.relative_to(args.root)), recursive=False)
        with tarfile.open(archive, 'r:gz') as source:
            actual = {
                member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest()
                for member in source.getmembers() if member.isfile()
            }
        if actual != members:
            raise RuntimeError('archive verification failed: ' + run.name)
        archives.append(dict(run=run.name, archive=archive.name, sha256=sha(archive),
                             bytes=archive.stat().st_size, members=members,
                             results=json.loads((run / 'RESULTS.json').read_text())))
    (args.out / 'INDEX.json').write_text(json.dumps(dict(
        source_root=str(args.root), archives=archives,
        note='Contains actual full trajectories, exact source snapshots, training weights '
             'and replay buffers when produced. Failed finalized smoke runs retained. '
             'SHA verified by reading every archived member; no new physics.'), indent=2))
    print(json.dumps({'archives': len(archives), 'bytes': sum(x['bytes'] for x in archives)}))


if __name__ == '__main__':
    main()
