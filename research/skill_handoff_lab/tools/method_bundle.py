"""Preserve this public-simulation study's real models/traces outside Git."""
import argparse,gzip,hashlib,json,tarfile
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.archive.parent.mkdir(parents=True,exist_ok=True)
    assert not a.archive.is_relative_to(a.runs) and not a.archive.is_relative_to(a.output)
    entries=[]
    for f in sorted(a.runs.rglob('*')):
        if f.is_file() and 'cache' not in f.relative_to(a.runs).parts:entries.append((f,Path('runs/method_evolution_20261028')/f.relative_to(a.runs)))
    for f in sorted(a.output.rglob('*')):
        if f.is_file() and f.suffix not in ['.aux','.log','.out'] and f.name not in ['RESEARCH_BUNDLE.json','BUNDLE_VERIFICATION.json','RELEASE_SHA256.json']:entries.append((f,Path('public_evidence')/f.relative_to(a.output)))
    for f in sorted((a.root/'research/skill_handoff_lab/tools').glob('method_*')):
        if f.is_file():entries.append((f,Path('source/research/skill_handoff_lab/tools')/f.name))
    manifest=[dict(path=str(relative),sha256=sha(f),bytes=f.stat().st_size) for f,relative in entries]
    with a.archive.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as gz:
            with tarfile.open(fileobj=gz,mode='w|') as tar:
                for f,relative in entries:
                    info=tar.gettarinfo(str(f),arcname=str(relative));info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                    with f.open('rb') as stream:tar.addfile(info,stream)
                import io
                content=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('BUNDLE_CONTENTS.json');info.size=len(content);tar.addfile(info,io.BytesIO(content))
    receipt=dict(archive=str(a.archive),sha256=sha(a.archive),bytes=a.archive.stat().st_size,files=len(entries),scope='Only this newly executed public ManiSkill/SAPIEN study: real model weights, exact training source, native trajectories, failures, scalar logs, public figures and videos. No ARES-R/site data.',extract='tar -xzf research_bundle.tar.gz -C NEW_EMPTY_DIRECTORY',contents='BUNDLE_CONTENTS.json verifies every archived file; original provenance absolute paths are retained, loaders accept relative run paths.',unique_checkpoints_deleted=0)
    (a.output/'RESEARCH_BUNDLE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
