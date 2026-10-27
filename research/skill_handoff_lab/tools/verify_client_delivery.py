"""Verify the client bundle against owned, completed synthetic evidence.

This does not run a robot, open final-test cases, change a run or extract an
untrusted archive. SB3 loading is restricted to our SHA-verified model archive.
"""
import argparse,csv,hashlib,io,json,re,subprocess,tarfile
from pathlib import Path,PurePosixPath
import numpy as np

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(workspace,deliverable):
    result={'status':'PASS','scope':'OWN_SYNTHETIC_CLIENT_DELIVERY','checks':{}}
    docs=load(deliverable/'DOCUMENT_QA.json');assert docs['status']=='PASS'
    verified=[]
    for x in docs['documents']:
        p=deliverable/x['path'];assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes'],p
        info=subprocess.check_output(['/usr/bin/pdfinfo',str(p)],text=True)
        pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1]);assert pages==x['pages']
        if p.name=='01_CLIENT_PROPOSAL_CN.pdf':assert 6<=pages<=10
        if p.name=='07_MENTOR_BRIEF_CN.pdf':assert pages<=2
        verified.append({'path':x['path'],'pages':pages,'sha256':sha(p)})
    for x in docs['source_copies']:
        assert sha(deliverable/x['source'])==sha(deliverable/x['paper_copy'])==x['sha256']
    result['checks']['compiled_documents']=verified;result['checks']['exact_paper_input_copies']=len(docs['source_copies'])
    figures=load(deliverable/'03_FIGURES/provenance/FIGURE_PROVENANCE.json')
    for group in ['inputs','outputs']:
        for name,x in figures[group].items():
            p=Path(name)
            if not p.is_absolute():p=(workspace/p) if (workspace/p).exists() else deliverable/p
            assert sha(p)==x['sha256'],p
        result['checks']['figure_'+group]=len(figures[group])
    rows=list(csv.DictReader((deliverable/'06_RUNS_MANIFEST/NEW_PILOT_RESULTS.csv').open()))
    assert len(rows)==27
    for x in rows:
        d=Path(x['run']);d=d if d.is_absolute() else workspace/d
        assert (d/'RUN_COMPLETE.json').exists()
        raw=load(d/'frozen_evaluation.json');audit=load(d/'INDEPENDENT_TASK_AND_CONTACT.json')
        assert sum(y['success'] for y in raw)==int(x['raw_successes'])
        assert audit['independent_successes']==int(x['independent_successes'])
        assert sha(d/'INDEPENDENT_TASK_AND_CONTACT.json')==x['audit_sha256']
        assert sha(d/'task_source.py')==x['source_sha256']
        assert int(x['n'])==len(raw)
    index=load(deliverable/'06_RUNS_MANIFEST/NEW_RUNS_INDEX.json')
    assert index['online_training_interactions']==163840 and index['final_tests_evaluated'] is False
    result['checks']['independently_audited_formal_runs']=len(rows)
    models=load(deliverable/'06_RUNS_MANIFEST/MODEL_DELIVERY_INDEX.json');archive=Path(models['delivery_archive']['path'])
    assert sha(archive)==models['delivery_archive']['sha256']
    assert models['formal_interactions']+models['software_diagnostic_interactions']==196608
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1);model_checks=[];member_checks=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            name=PurePosixPath(member.name);assert not name.is_absolute() and '..' not in name.parts and member.isfile()
        for model in models['models']:
            for name,x in model['files'].items():
                data=tar.extractfile(x['archive_member']).read();assert hashlib.sha256(data).hexdigest()==x['sha256'];member_checks+=1
                if name!='policy.zip':continue
                learner=PPO.load(io.BytesIO(data),device='cpu')
                assert learner.observation_space.shape==(64,) and learner.action_space.shape==(3,)
                assert learner.num_timesteps==16384
                prediction,_=learner.predict(np.zeros((1,64),np.float32),deterministic=True)
                assert np.isfinite(prediction).all()
                model_checks.append({'run':model['run'],'cohort':model['cohort'],'sha256':x['sha256'],'cpu_load':'PASS',
                    'parameters':sum(p.numel() for p in learner.policy.parameters()),'recorded_interactions':learner.num_timesteps})
    assert len(model_checks)==12
    result['checks']['model_member_hashes']=member_checks;result['checks']['models_cpu_loaded']=model_checks
    result['checks']['all_new_interactions_including_software_diagnostics']=196608
    frozen={'test_id':'e2ba4ea99c7b4c5430a208f71c57aaa23a4cdecfa81236be4d0019b6a0a84a39',
        'test_geometry':'c398b2d787f28931649e8d45bba485237f75a09f2dfc039667acb354dfe4433d',
        'test_dynamics':'d0d93d77390371431612020eceb2e235b30a69bec1a61920ad5759edda23827c'}
    for name,h in frozen.items():assert sha(workspace/'repos/ARES/research/skill_handoff_lab/configs/client_sci_v7'/f'{name}.json')==h
    result['checks']['sealed_final_test_hashes_unchanged']=True
    result['limitations']=['Hash/load checks are not a robot safety certificate','PDF visual QA is separately recorded',
        'No unobserved final-test outcome is inspected','Pilot is not convergence or new-method superiority proof']
    p=deliverable/'06_RUNS_MANIFEST/GLOBAL_DELIVERY_QA.json';p.write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':'PASS','pdfs':len(verified),'formal_runs':len(rows),'models':len(model_checks),
        'model_member_hashes':member_checks,'figures_checked':len(figures['outputs'])}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--deliverable',type=Path,required=True)
    a=p.parse_args();main(a.workspace.resolve(),a.deliverable.resolve())
