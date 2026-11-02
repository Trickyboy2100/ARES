"""Archive public simulator evidence and verify every member. No external upload."""
import argparse,json,tarfile,hashlib,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);entries=[];archives=[]
 for folder in sorted(x for x in a.root.iterdir()if x.is_dir()and x.name not in ['figures','video_contact','video_learning']):
  files=sorted(x for x in folder.rglob('*')if x.is_file());archive=a.out/f'RAW_{folder.name}.tar.gz'
  with tarfile.open(archive,'w:gz')as t:
   for f in files:
    rel=str(f.relative_to(a.root));t.add(f,arcname=rel,recursive=False);entries.append(dict(archive=archive.name,path=rel,size=f.stat().st_size,sha256=sha(f)))
  assert archive.stat().st_size<95_000_000,'Split archive before Git upload'
  archives.append(dict(file=archive.name,size=archive.stat().st_size,sha256=sha(archive)))
 rootfiles=sorted(x for x in a.root.iterdir()if x.is_file());archive=a.out/'RAW_metadata.tar.gz'
 with tarfile.open(archive,'w:gz')as t:
  for f in rootfiles:t.add(f,arcname=f.name,recursive=False);entries.append(dict(archive=archive.name,path=f.name,size=f.stat().st_size,sha256=sha(f)))
 archives.append(dict(file=archive.name,size=archive.stat().st_size,sha256=sha(archive)))
 for name in ['figures','video_contact','video_learning']:
  source=a.root/name
  if source.exists():shutil.copytree(source,a.out/name,dirs_exist_ok=True)
 for name in ['README.md','REVIEW.html','SUMMARY.json','ANALYSIS.json','ENVIRONMENT.json','PRIOR_RAW_AUDIT.json','REWARD_BOUNDARY_AUDIT.json','CONTACT_AUTHORITY.json','PRIOR_ART_SCOPE.md']:
  if(a.root/name).exists():shutil.copy2(a.root/name,a.out/name)
 # Verify compressed contents, not just archive filenames.
 for ar in archives:
  ee={e['path']:e for e in entries if e['archive']==ar['file']}
  with tarfile.open(a.out/ar['file'])as t:
   for member in t.getmembers():
    expected=ee.pop(member.name);raw=t.extractfile(member).read();assert len(raw)==expected['size'];assert hashlib.sha256(raw).hexdigest()==expected['sha256']
  assert not ee
 assets=[dict(path=str(f.relative_to(a.out)),size=f.stat().st_size,sha256=sha(f))for f in sorted(a.out.rglob('*'))if f.is_file()and f.name!='RAW_MANIFEST.json'and not(f.name.startswith('RAW_')and f.name.endswith('.tar.gz'))]
 (a.out/'RAW_MANIFEST.json').write_text(json.dumps(dict(original_root=str(a.root),scope='Public ManiSkill physical experiments only',archives=[dict(archive=ar['file'],size=ar['size'],sha256=ar['sha256'],entries=[{k:v for k,v in e.items()if k!='archive'}for e in entries if e['archive']==ar['file']])for ar in archives],visible_assets=assets,verification='Every tar member decompressed and SHA-verified'),indent=2));print(len(entries),'raw entries;',sum(x['size']for x in archives),'archive bytes')
if __name__=='__main__':main()
