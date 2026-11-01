"""CPU-only source-backed summary of native placement-axis scouts."""
import argparse,csv,gzip,hashlib,itertools,json
from pathlib import Path
import numpy as np
from transforms3d.quaternions import quat2mat

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True);rows=[];qa=[];geometry=[]
 for f in sorted(a.root.glob('*/RESULTS.json')):
  r=json.loads(f.read_text())[0];path=Path(r['path']);d=json.load(gzip.open(path,'rt'));bs=r['boundaries'];first=bs[0] if bs else {};last=bs[-1] if bs else {}
  failure_class='SUCCESS' if r['success'] else ('PREDECESSOR_BOUNDARY_FAILURE' if not first.get('valid',False) else 'SUCCESSOR_OR_FINAL_FAILURE')
  candidate=next((x for x in d['decisions'][0]['candidates'] if x['axis_degrees']==r['axis_degrees']),{})
  row=dict(predicted_axis_valid=candidate.get('valid'),predicted_axis_motion_steps=candidate.get('predicted_motion_steps'),failure_class=failure_class,run=f.parent.name,seed=r['seed'],label=r['label'],success=r['success'],error=r['error'],steps=r['steps'],axis_degrees=r['axis_degrees'],axis_aware_release=r.get('axis_aware_release',False),top_offset_m=r['top_offset_m'],decisions=r['decisions'],first_boundary_verified=first.get('valid',False),first_boundary_step=first.get('step'),first_AB_distance_m=first.get('AB_distance_m'),first_support_margin_m=first.get('support_distance_margin_m'),first_target_error_m=first.get('target_xy_error_m'),first_B_displacement_m=float(np.linalg.norm(np.asarray(first['B'][:2])-d['initial']['actors']['cubeB'][0][:2])) if first else None,first_C_displacement_m=float(np.linalg.norm(np.asarray(first['C'][:2])-d['initial']['actors']['cubeC'][0][:2])) if first else None,second_boundary_verified=last.get('valid',False) if len(bs)>1 else False,initial_sha256=r['initial_sha256'],search_wall_s=r['search_wall_s'],execution_planning_wall_s=r['planning_wall_s'],wall_s=r['wall_s'],source_path=str(path),sha256=r['sha256'])
  axis=np.r_[np.cos(np.deg2rad(r['axis_degrees'])),np.sin(np.deg2rad(r['axis_degrees'])),0.]
  place=next((x for x in d['plans'] if x['label']=='A_PLACE'),None);stages={}
  if place:
   for segment in ['A_LIFT','A_PREPLACE']:
    samples=[x for x in d['transitions'] if x['segment']==segment]
    if not samples:continue
    t=samples[-1];A=np.array(t['state']['actors']['cubeA'][0][:7]);B=np.array(t['state']['actors']['cubeB'][0][:7]);tcp=np.array(t['tcp'][0]);Rt=quat2mat(place['target_q']);Rrel=quat2mat(tcp[3:]).T@quat2mat(A[3:]);Rpred=Rt@Rrel;goal=np.array(place['target_p'])+Rt@quat2mat(tcp[3:]).T@(A[:3]-tcp[:3]);ar=.02*np.abs(Rpred.T@axis).sum();br=.02*np.abs(quat2mat(B[3:]).T@axis).sum()
    stages[segment]=dict(predicted_A_axis_radius_m=float(ar),legacy_A_axis_radius_m=float(.02*np.abs(axis[:2]).sum()),predicted_separation_margin_m=float((goal-B[:3])@axis-ar-br),predicted_A_goal=goal.tolist())
  phase_motion={};B0=np.array(d['initial']['actors']['cubeB'][0][:2])
  for t in d['transitions']:
   if t['segment'].startswith('A_'):phase_motion[t['segment']]=max(phase_motion.get(t['segment'],0),float(np.linalg.norm(np.array(t['state']['actors']['cubeB'][0][:2])-B0)))
  geometry.append(dict(run=row['run'],attachment_projected_geometry=stages,B_displacement_max_by_phase_m=phase_motion,scope='CPU reconstruction of recorded measured attachment and commanded TCP pose. Positive projection margin refutes initial overlap along separation axis; does not prove swept gripper clearance or exact contact partner.'))
  rows.append(row);qa.append(dict(run=row['run'],sha_ok=hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],native250=len(d['transitions'])==250,final_success_matches=bool(d['transitions'][-1]['info']['success'][0])==row['success']))
 summary=dict(episodes=len(rows),physical_steps=sum(x['steps'] for x in rows),optimizer_updates=0,initial_hash_consistent=all(len({r['initial_sha256'] for r in rows if r['seed']==s})==1 for s in {r['seed'] for r in rows}),two_boundary_episodes=sum(x['decisions']==2 for x in rows),verified_first_boundary=sum(x['first_boundary_verified'] for x in rows),scope='NATIVE_PARAMETER_SCOUT_NOT_RL_GAIN')
 summary['kinematic_valid_but_precursor_failed']=sum(r['predicted_axis_valid'] and not r['first_boundary_verified'] for r in rows)
 prefix_pairs=[]
 for row in rows:
  if not row['axis_aware_release']:continue
  oldrun=row['run'].replace('_rule','');oldpath=a.root/oldrun/'TRAJECTORY.json.gz'
  if not oldpath.exists():continue
  new=json.load(gzip.open(row['source_path'],'rt'));old=json.load(gzip.open(oldpath,'rt'))
  end=max(t['step'] for t in old['transitions'] if t['segment']=='A_LIFT');ln=new['transitions'][:end];lo=old['transitions'][:end]
  delta={k:max(float(np.max(np.abs(np.array(x[k])-np.array(y[k])))) for x,y in zip(ln,lo)) for k in ['action','q','qvel','tcp']}
  for actor in ['cubeA','cubeB','cubeC']:delta[actor]=max(float(np.max(np.abs(np.array(x['state']['actors'][actor])-np.array(y['state']['actors'][actor])))) for x,y in zip(ln,lo))
  prefix_pairs.append(dict(new=row['run'],original=oldrun,through_step=end,initial_sha_matches=new['summary']['initial_sha256']==old['summary']['initial_sha256'],max_abs_differences=delta,exact_prefix=all(x==0 for x in delta.values())))
 (a.out/'REPAIR_PREFIX_AUDIT.json').write_text(json.dumps(prefix_pairs,indent=2))
 summary['policies']={label:dict(successes=sum(r['success'] for r in rows if r['label']==label),episodes=sum(r['label']==label for r in rows)) for label in sorted({r['label'] for r in rows})}
 for name,data in [('SUMMARY',summary),('EPISODES',rows),('QA',qa),('GEOMETRY_AUDIT',geometry)]: (a.out/f'{name}.json').write_text(json.dumps(data,indent=2))
 if rows:
  with (a.out/'EPISODES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,max(3,len(rows)*.4)));xs=np.arange(len(rows));values=[r['first_support_margin_m']*1000 if r['first_support_margin_m'] is not None else np.nan for r in rows]
 ax.barh(xs,values,color=['#397d76' if r['success'] else '#ba7848' for r in rows]);ax.set_yticks(xs,[r['run']+(' | PASS' if r['success'] else ' | FAIL') for r in rows]);ax.invert_yaxis();ax.set_xlabel('Measured first-release native support-distance margin (mm)');ax.set_title('Native250 pyramid: actual first-skill result vs full-task success');fig.tight_layout();fig.savefig(a.out/'SUPPORT_AND_OUTCOME.png',dpi=200);fig.savefig(a.out/'SUPPORT_AND_OUTCOME.svg');plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
