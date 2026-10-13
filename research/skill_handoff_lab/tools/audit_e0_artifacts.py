"""Verify actual E0 source/result/training lineage and freeze all file hashes."""
import argparse,hashlib,json
import torch
from pathlib import Path
from research.skill_handoff_lab.cash.cache_backend import CachedPlanningBackend

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    counts={};verified=0
    for name in ['execution_lower','execution_continuation','execution_variations']:
        rows=json.loads((a.run/name/'e0_summary.json').read_text());counts[name]={'calls':len(rows),'model_qualified':sum(r['model_validated_path'] for r in rows)}
        for job in (a.run/name).iterdir():
            if not (job/'e0_independent_validation.json').exists():continue
            v=json.loads((job/'e0_independent_validation.json').read_text())
            assert v['request_sha256']==sha(job/'request.json')
            assert v['result_sha256']==sha(job/'result.json')
            sources=[a.run/'runtime_source/validate_e0_v1.py',a.run/'runtime_source/validate_e0_v2.py']
            assert v['validator_sha256'] in [sha(source) for source in sources]
            executor=json.loads((job/'validation.json').read_text())
            assert executor['executor_sha256']==sha(a.run/'runtime_source/oracle_executor.py')
            verified+=1
    cache=a.run/'cache.json'
    for split in ['train','validation','test']:CachedPlanningBackend(cache,split)
    training=[]
    for name in ['learning_seed17_v2','learning_seed23_v2','learning_seed17_no_aux_v2']:
        m=json.loads((a.run/name/'manifest.json').read_text());assert m['dataset_sha256']==sha(cache)
        assert next(v for k,v in m['source_sha256'].items() if k.endswith('train_e0_cost.py'))==sha(a.run/'runtime_source/train_e0_cost_v2.py')
        weights=torch.load(a.run/name/'cash_aux.pt',weights_only=True)['model']
        training.append({'run':name,'seed':m['seed'],'steps':m['steps'],'effect_coef':m['effect_coef'],
            'architecture':{'state_dim':weights['state.0.weight'].shape[1],
                'action_dim':weights['action.0.weight'].shape[1],'hidden':weights['state.0.weight'].shape[0],
                'effect_dim':weights['effect.weight'].shape[0]},
            'models_sha256':{p.name:sha(p) for p in (a.run/name).glob('*.pt')}})
    files={str(p.relative_to(a.run)):sha(p) for p in sorted(a.run.rglob('*')) if p.is_file() and p.name!='data_manifest.json'}
    value={'ares_baseline':'4dc7b6483bb9ee9474bc67517058c6ab9ed8695a','ares_r_readonly':'9de17d2abe2775fbcbcdecc04db21c2b9caf1aca',
        'curobo_source':'8e734f3ced1df898990bcd92de40abce475907db','machine':'andyee@172.25.24.190 / GPU0 RTX4090',
        'scope':'DECLARED_RESEARCH_MODEL_PLANNING + OFFLINE_COST_BANDIT; NOT physical task success',
        'validated_artifact_sets':verified,'counts':counts,'training':training,'sha256':files}
    (a.run/'data_manifest.json').write_text(json.dumps(value,indent=2));print(json.dumps({'verified':verified,'files':len(files),'counts':counts}))
if __name__=='__main__':main()
