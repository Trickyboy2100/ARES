"""Narrow metadata refresh; never invokes Matplotlib or rewrites figure outputs.

Run from the workspace root with --apply. The physical-summary JSON may have
expanded audit metadata only if every metric consumed by build_evidence_figures
still exactly reproduces its frozen CSV. Any other stale input/output aborts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parents[5]
CAUTION='Full closed-loop trajectories differ; not isolated force/velocity causation or calibrated significance'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(p):
    q=Path(p)
    return q.resolve() if q.is_absolute() else (WORKSPACE/q).resolve()
def metrics(x):
    # Exact projection consumed by build_evidence_figures.py, physicalpath block.
    return {'comparison':x['label'],'left':x['left'],'right':x['right'],
            'matched_cases':x['n'],'left_successes':x['left_successes'],
            'right_successes':x['right_successes'],
            'rescued_cases':len(x['rescued_case_ids']),
            'lost_cases':len(x['lost_case_ids']),
            'initial_max_difference':x['initial_max_difference'],
            'mean_object_xyz_rms_difference_um':x['mean_object_xyz_rms_difference_m']*1e6,
            'mean_applied_reference_delta_rms_difference_um':x['mean_applied_reference_delta_rms_difference_m']*1e6,
            'caution':CAUTION}
def outputs_match(manifest):
    for name,item in manifest['outputs'].items():
        p=ROOT/name
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],f'Frozen output changed: {name}'

def main(apply):
    manifestpath=ROOT/'03_FIGURES/provenance/FIGURE_PROVENANCE.json'
    source=ROOT/'06_RUNS_MANIFEST/PHYSICAL_CAUSAL_INTERVENTIONS.json'
    csvpath=ROOT/'03_FIGURES/data/new_pilot_physical_interventions.csv'
    recordpath=ROOT/'03_FIGURES/provenance/PROVENANCE_REFRESH_20261027.json'
    manifest=json.loads(manifestpath.read_text());physical=json.loads(source.read_text())
    with csvpath.open() as f:rows=list(csv.DictReader(f))
    projected=[metrics(x) for x in physical['comparisons']]
    assert len(projected)==len(rows)==6
    for actual,frozen in zip(projected,rows):
        assert set(actual)==set(frozen)
        for key,value in actual.items():
            if isinstance(value,(int,float)):
                assert float(frozen[key])==value,f'Plot metric changed: {actual["comparison"]}/{key}'
            else:assert frozen[key]==value,f'Plot field changed: {actual["comparison"]}/{key}'
    outputs_match(manifest)
    keys=[k for k in manifest['inputs'] if resolve(k)==source.resolve()]
    assert len(keys)==1
    targetkey=keys[0]
    for key,item in manifest['inputs'].items():
        if key==targetkey:continue
        p=resolve(key)
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],f'Another stale input: {key}'
    components={}
    for x in physical['comparisons']:
        assert x['initial_max_recorded_physical_state_difference']==0
        assert all(v==0 for v in x['initial_component_maxima'].values())
        components[x['label']]=x['initial_component_maxima']
    old=manifest['inputs'][targetkey].copy();new={'sha256':sha(source),'bytes':source.stat().st_size}
    if old==new:
        print('PASS: input already current, consumed metrics and all frozen outputs verified; no files changed')
        return
    result={'operation':'NARROW_PROVENANCE_INPUT_REFRESH_WITHOUT_REDRAW',
            'date':'2026-10-27','script_sha256':sha(Path(__file__)),
            'reason':'Physical summary now includes expanded measured initial-state audit metadata; every plotted/exported consumed metric remains exactly unchanged.',
            'target_manifest':str(manifestpath.relative_to(ROOT)),
            'only_updated_input_key':targetkey,'old_input_record':old,'new_input_record':new,
            'frozen_export_csv':str(csvpath.relative_to(ROOT)),'frozen_csv_sha256':sha(csvpath),
            'verified_comparisons':len(projected),'verified_consumed_fields':list(projected[0]),
            'reproduced_metric_projection_sha256':hashlib.sha256(json.dumps(projected,sort_keys=True).encode()).hexdigest(),
            'frozen_outputs_verified_before_and_after':len(manifest['outputs']),
            'frozen_output_hashes_unchanged':True,'redraw_performed':False,
            'output_files_modified':[],
            'expanded_initial_component_audit':components,
            'limitations':'Equality covers recorded initial state only; hidden PhysX contact/solver cache is not restored or proven identical. This validation does not reconstruct or assert every unconsumed old JSON field.'}
    if apply:
        manifest['inputs'][targetkey]=new
        manifestpath.write_text(json.dumps(manifest,indent=2)+'\n')
        outputs_match(json.loads(manifestpath.read_text()))
        recordpath.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','applied':apply,'verified_comparisons':len(projected),'frozen_outputs_unchanged':len(manifest['outputs']),'record':str(recordpath) if apply else None}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');main(p.parse_args().apply)
