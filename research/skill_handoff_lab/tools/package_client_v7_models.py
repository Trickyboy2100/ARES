"""Package own synthetic pilot final models/config/source; leave full traces intact."""
import argparse,gzip,hashlib,io,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();members=[];models=[]
for cohort,base in [('formal',root/'formal'),('pre_fix_software_diagnostic',root)]:
    for d in sorted(base.iterdir()):
        if not (d/'RUN_COMPLETE.json').exists() or not (d/'policy.zip').exists():continue
        m=json.loads((d/'manifest.json').read_text())
        if not m['online_training_interactions']:continue
        record={'cohort':cohort,'run':d.name,'online_training_interactions':m['online_training_interactions'],'files':{}}
        for name in ['policy.zip','task_source.py','frame_source.py','boundary_credit_source.py','run_config.json','scene.json','manifest.json','training_episodes.json','credit_audits.json']:
            path=d/name
            if not path.exists():continue
            arc=f'{cohort}/{d.name}/{name}';body=path.read_bytes()
            item={'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body),'archive_member':arc,'original_path':str(path)}
            record['files'][name]=item;members.append((arc,body))
        models.append(record)
index={'scope':'Own SYNTHETIC P2 models only; no site data, production settings or real-robot observations',
       'models':models,'model_count':len(models),'formal_models':sum(x['cohort']=='formal' for x in models),
       'formal_interactions':sum(x['online_training_interactions'] for x in models if x['cohort']=='formal'),
       'software_diagnostic_interactions':sum(x['online_training_interactions'] for x in models if x['cohort']!='formal'),
       'source_authority':'Each model own captured source bytes, not current Git HEAD',
       'all_full_traces_preserved':True}
body=json.dumps(index,indent=2).encode();members.append(('INDEX.json',body))
archive=root/'MODELS_AND_CONFIGS.tar.gz'
with archive.open('wb') as out:
    with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w') as tar:
            for name,content in sorted(members):
                info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o644;info.mtime=0
                tar.addfile(info,io.BytesIO(content))
# Verify every archived file against its original or the generated index.
expected={name:hashlib.sha256(content).hexdigest() for name,content in members}
with tarfile.open(archive,'r:gz') as tar:
    for item in tar.getmembers():
        assert hashlib.sha256(tar.extractfile(item).read()).hexdigest()==expected[item.name]
index['delivery_archive']={'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'verified_members':len(members)}
a.index.parent.mkdir(parents=True,exist_ok=True);a.index.write_text(json.dumps(index,indent=2))
(root/'MODELS_INDEX.json').write_text(json.dumps(index,indent=2));print(json.dumps({k:v for k,v in index.items() if k!='models'},indent=2))
