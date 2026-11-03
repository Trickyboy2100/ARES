"""Archive public simulated evidence, excluding downloaded third-party source."""
import argparse,json,tarfile,hashlib,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
archives=[]
def archive(name,files):
 entries=[];arc=a.out/f'RAW_{name}.tar.gz'
 with tarfile.open(arc,'w:gz')as t:
  for x in files:
   rel=str(x.relative_to(a.root));b=x.read_bytes();t.add(x,arcname=rel,recursive=False);entries.append(dict(path=rel,size=len(b),sha256=sha(b)))
 assert arc.stat().st_size<95_000_000,(arc,arc.stat().st_size)
 archives.append(dict(archive=arc.name,size=arc.stat().st_size,sha256=sha(arc.read_bytes()),entries=entries))
for folder in sorted(x for x in a.root.iterdir()if x.is_dir()and x.name not in ['figures','video_learning','video_handoff','video_policy_extraction']):
 files=sorted(x for x in folder.rglob('*')if x.is_file()and '__pycache__' not in x.parts and (folder.name!='prior_code'or x.name=='INDEX.json'))
 # Split by uncompressed payload, conservatively below GitHub's file limit.
 chunk=[];size=0;part=0
 for f in files:
  if chunk and size+f.stat().st_size>80_000_000:archive(f'{folder.name}_{part:02d}',chunk);chunk=[];size=0;part+=1
  chunk.append(f);size+=f.stat().st_size
 if chunk:archive(f'{folder.name}_{part:02d}',chunk)
archive('metadata',sorted(x for x in a.root.iterdir()if x.is_file()))
for name in ['figures','video_learning','video_handoff','video_policy_extraction']:
 if (a.root/name).exists():shutil.copytree(a.root/name,a.out/name,dirs_exist_ok=True)
for name in ['README.md','REVIEW.html','ANALYSIS.json','FINAL_RESULTS.json','DECISION_ATTRIBUTION.json','SHARED_CONTROL_AUDIT.json','PRIOR_FIXED_SELECTION.json','DISTRIBUTION_REVISION_PROTOCOL.json','POLICY_EXTRACTION_AUDIT.json','EXTRACTION_COMPARISON.json','PRIOR_ART.md','SEMANTICS_AUDIT.json','JOINT_OUTCOME_AUDIT.json','TRAINING_REPLAY_AUDIT.json','ENVIRONMENT.json','THIRD_PARTY_NOTICES.md','MANISKILL_APACHE_LICENSE.txt']:
 if (a.root/name).exists():shutil.copy2(a.root/name,a.out/name)
for ar in archives:
 with tarfile.open(a.out/ar['archive'])as t:
  for e in ar['entries']:
   b=t.extractfile(e['path']).read();assert len(b)==e['size']and sha(b)==e['sha256']
assets=[dict(path=str(x.relative_to(a.out)),size=x.stat().st_size,sha256=sha(x.read_bytes()))for x in sorted(a.out.rglob('*'))if x.is_file()and x.name!='RAW_MANIFEST.json'and not x.name.startswith('RAW_')]
(a.out/'RAW_MANIFEST.json').write_text(json.dumps(dict(original_root=str(a.root),archives=archives,visible_assets=assets,scope='Public simulator evidence only; third-party downloaded sources excluded; prior peg_value package required separately',verification='Every archive member decompressed and SHA-verified'),indent=2));print('Verified',sum(len(x['entries'])for x in archives),'members;',sum(x['size']for x in archives),'compressed bytes')
