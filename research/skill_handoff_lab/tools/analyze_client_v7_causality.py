"""Paired physical interventions from completed v7 synthetic validation traces."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def load(d):
    if not (d/'RUN_COMPLETE.json').exists():raise ValueError('Cannot analyze an active/incomplete run')
    r=json.loads((d/'frozen_evaluation.json').read_text());x=np.load(d/'actual_physics_trace.npz')
    states=x['states'];cols=list(x['columns']);cases={}
    for row in r:
        i=row['env'];cid=row['case_id'];trace=states[:,i];trace=trace[trace[:,cols.index('case_id')].astype(int)==cid]
        cases[cid]=(row,trace)
    return cases,cols

def main(a):
    a.output.mkdir(parents=True,exist_ok=True);comparisons=[];details=[]
    pairs=[('B3_seed31_validation','B3_no_outcome_intervention','same-model remove effect history'),
           ('B3_seed31_validation','B3_measured_response_intervention','same-model measured response only removed; command memory retained'),
           ('B3_seed31_validation','B3_no_next_intervention','same-model partial next-contract mask'),
           ('B0_validation','B0_withdraw_validation','existing bounded backoff/recontact controller'),
           ('B0_validation','B4_parameter_-1.0','fixed -4mm phase0 contact endpoint'),
           ('B0_validation','B4_parameter_+1.0','fixed +4mm phase0 contact endpoint')]
    for left,right,label in pairs:
        if not (a.root/left/'RUN_COMPLETE.json').exists() or not (a.root/right/'RUN_COMPLETE.json').exists():continue
        x,cols=load(a.root/left);y,_=load(a.root/right);rows=[]
        for cid in sorted(x.keys()&y.keys()):
            one,p=x[cid];two,q=y[cid];n=min(len(p),len(q))
            initial=np.max(abs(p[0,:12]-q[0,:12]));delta=p[:n,6:9]-q[:n,6:9]
            components={'q':[f'q{i}' for i in range(6)],'qd':[f'qd{i}' for i in range(6)],
                'object_position':['obj_x','obj_y','obj_z'],'paddle_position':['paddle_x','paddle_y','paddle_z'],
                'object_quaternion':['obj_qw','obj_qx','obj_qy','obj_qz'],
                'object_linear_velocity':['vx','vy','vz'],'object_angular_velocity':['wx','wy','wz']}
            initial_components={key:float(np.max(abs(p[0,[cols.index(c) for c in names]]-q[0,[cols.index(c) for c in names]]))) for key,names in components.items()}
            applied_columns=[cols.index('applied_dx'),cols.index('applied_dy')]
            cmd=p[:n,applied_columns]-q[:n,applied_columns]
            phase0=(p[:n,15]==0)&(q[:n,15]==0)
            parameter_delta=(p[:n,cols.index('switch_signal')]-q[:n,cols.index('switch_signal')])*.004
            h1=one.get('handoff_state');h2=two.get('handoff_state')
            rows.append(dict(comparison=label,left=left,right=right,case_id=cid,initial_max_q_object_paddle_difference=float(initial),
                common_trace_samples=n,left_success=int(one['success']),right_success=int(two['success']),
                left_valid_handoff=bool(h1),right_valid_handoff=bool(h2),
                initial_max_recorded_physical_state_difference=max(initial_components.values()),
                **{'initial_'+key+'_max_difference':value for key,value in initial_components.items()},
                handoff_object_difference_m=float(np.linalg.norm(np.array(h1['object_position'])-h2['object_position'])) if h1 and h2 else None,
                left_handoff_speed_m_s=float(np.linalg.norm(h1['object_linear_velocity'])) if h1 else None,
                right_handoff_speed_m_s=float(np.linalg.norm(h2['object_linear_velocity'])) if h2 else None,
                object_xyz_rms_difference_m=float(np.sqrt(np.mean(delta**2))),
                maximum_object_difference_m=float(np.max(np.linalg.norm(delta,axis=-1))),
                applied_reference_delta_rms_difference_m=float(np.sqrt(np.mean(cmd**2))),
                phase0_contact_parameter_rms_difference_m=float(np.sqrt(np.mean(parameter_delta[phase0]**2))) if phase0.any() else None,
                left_seconds=one['steps']/60,right_seconds=two['steps']/60))
        details+=rows
        comparisons.append(dict(label=label,left=left,right=right,n=len(rows),
            left_successes=sum(r['left_success'] for r in rows),right_successes=sum(r['right_success'] for r in rows),
            rescued_case_ids=[r['case_id'] for r in rows if not r['left_success'] and r['right_success']],
            lost_case_ids=[r['case_id'] for r in rows if r['left_success'] and not r['right_success']],
            common_valid_handoff_cases=sum(r['left_valid_handoff'] and r['right_valid_handoff'] for r in rows),
            common_handoff_left_suffix_successes=sum(r['left_success'] for r in rows if r['left_valid_handoff'] and r['right_valid_handoff']),
            common_handoff_right_suffix_successes=sum(r['right_success'] for r in rows if r['left_valid_handoff'] and r['right_valid_handoff']),
            mean_handoff_object_difference_m=float(np.mean([r['handoff_object_difference_m'] for r in rows if r['handoff_object_difference_m'] is not None])),
            initial_max_difference=max(r['initial_max_q_object_paddle_difference'] for r in rows),
            initial_max_recorded_physical_state_difference=max(r['initial_max_recorded_physical_state_difference'] for r in rows),
            initial_component_maxima={key:max(r['initial_'+key+'_max_difference'] for r in rows) for key in components},
            mean_object_xyz_rms_difference_m=float(np.mean([r['object_xyz_rms_difference_m'] for r in rows])),
            mean_applied_reference_delta_rms_difference_m=float(np.mean([r['applied_reference_delta_rms_difference_m'] for r in rows])),
            sources={d:hashlib.sha256((a.root/d/'actual_physics_trace.npz').read_bytes()).hexdigest() for d in [left,right]}))
    if details:
        with (a.output/'PHYSICAL_CAUSAL_INTERVENTIONS.csv').open('w',newline='') as out:
            w=csv.DictWriter(out,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    report={'scope':'cold-start matched reset and continuous trajectories; no partial contact state restore',
            'comparisons':comparisons,'caution':'matched initial recorded q/qd/object pose, linear/angular velocity and paddle pose only; hidden PhysX solver/contact cache not restored or proven identical. Aligned by elapsed physics sample per case, not identical subsequent states; differences combine state-action feedback; no velocity-only causal claim'}
    (a.output/'PHYSICAL_CAUSAL_INTERVENTIONS.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
