"""Measured contact-stage diagnostics, no causal labels inferred from distances."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[];plans=0;failedplans=0;prefix_states={};prefix_max=0.;prefix_comparisons=0
    for path in sorted(a.source.glob('[0-9]*.json')):
        r=json.loads(path.read_text());s=r['summary'];plans+=len(r['plans']);failedplans+=sum('position' not in z for z in r['plans'])
        for i,seed in enumerate(s['seeds']):
            ds=s['decisions'][i]
            if len(ds)<2:continue
            before=r['transitions'][ds[1]['step']-1]
            recorded=np.r_[before['q'][i],before['qvel'][i],*[v[i] for v in before['state']['actors'].values()]]
            key=(seed,int(s['policy'][1]))
            if key in prefix_states:
                prefix_max=max(prefix_max,float(np.max(abs(recorded-prefix_states[key]))));prefix_comparisons+=1
            else:prefix_states[key]=recorded
            ts=[t for t in r['transitions'] if t['segments'][i]=='C_CLOSE']
            ps=[z for z in r['plans'] if z['env']==i and z['label']=='C_DESCEND' and 'position' in z]
            if not ts or not ps:continue
            t=ts[-1];plan=ps[-1];tcp=np.array(t['tcp'][i]);cube=np.array(t['state']['actors']['cubeC'][i]);target=np.array(plan['target_p']);opening=sum(t['q'][i][-2:]);b=next(z for z in s['boundaries'][i] if z['stage']==0)
            rows.append(dict(seed=seed,policy=s['policy'],grasp_failed=int(s['errors'][i]=='C_NOT_GRASPED'),native_success=int(s['success'][i]),tcp_target_error_mm=float(np.linalg.norm(tcp[:3]-target)*1000),tcp_cube_xy_mm=float(np.linalg.norm(tcp[:2]-cube[:2])*1000),cube_moved_since_handoff_mm=float(np.linalg.norm(cube[:3]-np.array(b['C'][:3]))*1000),handoff_nearest_obstacle_center_mm=float(min(np.linalg.norm(np.array(b['C'][:2])-np.array(b[n][:2])) for n in ['A','B'])*1000),finger_opening_mm=float(opening*1000)))
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'CONTACT_DIAGNOSIS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    result=dict(planner_calls=plans,failed_plans=failedplans,recorded_prefix_state_comparisons=prefix_comparisons,recorded_prefix_max_abs_difference=prefix_max,scope='Post-execution association; no causal proof or policy input',groups={})
    for fail in [0,1]:
        rr=[r for r in rows if r['grasp_failed']==fail];result['groups'][str(fail)]={k:dict(median=float(np.median([r[k] for r in rr])),q90=float(np.quantile([r[k] for r in rr],.9))) for k in rows[0] if k.endswith('_mm')};result['groups'][str(fail)]['n']=len(rr)
    (a.out/'CONTACT_DIAGNOSIS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
