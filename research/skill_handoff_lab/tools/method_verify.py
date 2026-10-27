"""CPU evidence reconciliation: no simulator result is replaced with a mock."""
import argparse,hashlib,json,subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream',type=Path);a=p.parse_args();out=a.output;workspace=a.runs.parents[1];checks={};index=[];interactions=0;groups=defaultdict(set);allseeds=set()
    sources=load(out/'RAW_SOURCE_INDEX.json')
    for name,source in sources.items():
        result=a.runs/source.get('relative_path',str(Path(name)/Path(source['path']).name));assert sha(result)==source['sha256'];d=load(result);cap=d['native_horizon']
        assert cap=={'StackCube-v1':50,'PegInsertionSide-v1':100,'StackPyramid-v1':250}[d['task']]
        consumed=0
        for f in sorted(result.parent.glob('*.json')):
            if f==result:continue
            trace=load(f)
            if 'transitions' not in trace:continue
            summary=trace.get('summary',{});states=trace['transitions'];assert len(states)<=cap
            seeds=summary.get('seeds',trace.get('seeds',[summary.get('seed')]))
            assert all(s is not None for s in seeds);allseeds.update(seeds);consumed+=len(seeds)*len(states)
            initial=trace.get('initial',trace.get('initial_state'));digest=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest();assert digest==summary.get('initial_sha256',trace.get('initial_sha256'))
            groups[name,tuple(seeds)].add(digest)
            if d['task']=='StackCube-v1':
                assert len(states)==50
                assert states[-1]['success']==summary.get('success',states[-1]['success'])
                # New telemetry distinguishes an issued value from actual phase consumption.
                for state in states:
                    auth=state['authority']
                    if 'issued_parameter_history_m' in auth and 'residual' not in summary.get('policy',''):
                        phase=np.array(auth['phase_before']);applied=np.array(auth['applied_m']);assert np.max(abs(applied[~np.isin(phase,[5,6]),2]),initial=0)<1e-7
                    assert np.max(abs(np.array(auth['clipped'])))<=1+1e-7
            else:
                last=states[-1].get('success',states[-1].get('info',{}).get('success'));expected=summary['success'];assert last==(expected if isinstance(expected,list) else [expected])
            index.append(dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size,task=d['task'],steps=len(states),seeds=seeds,initial_sha256=digest))
        assert consumed==d['physical_interactions'],(name,consumed,d['physical_interactions']);interactions+=consumed
    assert all(len(v)==1 for v in groups.values());assert all(not 300000<=s<=300255 for s in allseeds)
    # Independently reconcile the six actual checkpoint updates, not merely metadata flags.
    model_index=load(out/'MODEL_INDEX.json')
    for r in model_index['models']:
        file=a.runs/r.get('relative_path',str(Path('training/runs')/Path(r['path']).parent.name/Path(r['path']).name));assert sha(file)==r['sha256'];final=torch.load(file,map_location='cpu',weights_only=True);initial=torch.load(file.with_name('ckpt_1.pt'),map_location='cpu',weights_only=True)
        delta=float(torch.sqrt(sum(((final[k]-initial[k])**2).sum() for k in final if k.startswith('actor_mean.'))));assert delta>0 and abs(delta-r['actor_delta_l2'])<1e-6
    for r in model_index['selector_models']:
        file=a.runs/r.get('relative_path',str(Path('selectors_r1')/Path(r['path']).name));assert sha(file)==r['sha256'];b=torch.load(file,map_location='cpu',weights_only=False)
        assert set(b['train_seed_ids']).isdisjoint(b['calibration_seed_ids']);assert all(100000<=s<=100511 for s in b['train_seed_ids']+b['calibration_seed_ids']);assert len(b['models'])==5
        assert not any('success' in name or 'seed' in name or 'case' in name for name in b['features'])
    upstream=a.upstream or workspace/'external/benchmark_scout_20261027/ManiSkill';env=load(out/'ENVIRONMENT_AND_ISOLATION.json')
    for name,digest in env['task_source_sha256'].items():
        rel='mani_skill/envs/tasks/tabletop/'+name;data=subprocess.check_output(['git','-C',str(upstream),'show',env['upstream_commit']+':'+rel]);assert hashlib.sha256(data).hexdigest()==digest==sha(upstream/rel)
    assets=upstream/'mani_skill/assets/robots/panda';save(out/'PUBLIC_ASSET_INDEX.json',dict(source='Pinned public ManiSkill Panda assets; not JAKA/site geometry',upstream_commit=env['upstream_commit'],files=[dict(path=str(f.relative_to(upstream)),sha256=sha(f),bytes=f.stat().st_size) for f in sorted(assets.rglob('*')) if f.is_file()]))
    save(out/'RAW_TRAJECTORY_INDEX.json',index)
    result=load(out/'RESULTS.json');assert result['total_control_interactions']==interactions+614400+sum(r['evaluation_interactions'] for r in model_index['models'])
    checks.update(native_caps_and_terminal_outcomes='PASS',all_recorded_physical_interactions_reconciled='PASS',cold_reset_initial_hashes='PASS',sealed_test_not_executed='PASS',real_actor_weight_updates='PASS',model_and_raw_source_hashes='PASS',train_only_selector_fitting='PASS',unchanged_native_task_sources='PASS',new_consumed_parameter_telemetry='PASS',recorded_physical_trajectories=len(index),actual_interactions=result['total_control_interactions'],independent_method_increment='NOT_ESTABLISHED')
    save(out/'QA.json',dict(checks=checks,scope='CPU reconciliation of actual native runs, not a replacement GPU test; video/PDF visual review is documented separately in REVIEW.md'));print(json.dumps(checks))
if __name__=='__main__':main()
