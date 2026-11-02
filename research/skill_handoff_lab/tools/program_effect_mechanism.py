"""Independent physical metrics; G2A relative-palm metric used as passive audit.
No claim to reproduce LEAP's twelve imposed-wrench experiments on Panda.
"""
import argparse,gzip,json,sys,hashlib,collections
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,'/home/andyee/robot-research/ARES-Skilllab/external/g2a-research-20261102')
from g2a.stress import in_hand_errors,StabilityAccumulator

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];pairs=[]
 for folder in ['tool_effect','tool_controls','train108','evaluation','tool_all_contacts','retained100','endpoint_fixed','endpoint_dense']:
  rp=a.root/folder/'RESULTS.json'
  if not rp.exists():continue
  for r in json.loads(rp.read_text()):
   path=Path(r['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(path,'rt'));s=d['steps'];assert len(s)==100 and bool(s[-1]['native_success'])==r['success'];boundary=d['skill_results'][0]['boundary_step'] if d['skill_results'] else None
   row=dict(group=folder,seed=r['seed'],mass=r['mass'],grasp=r['grasp'],mode=r['mode'],method=r.get('method'),success=r['success'],error=r['error'],boundary=boundary,remaining=None if boundary is None else 100-boundary,initial_sha256=r['initial_sha256'],wall_s=r['wall_s'],phases=dict(collections.Counter(x['phase'] for x in s)),terminal_distance_m=s[-1]['cube_base_distance_m'],path=str(path),sha256=r['sha256'])
   row['finish_step']=max(x['step'] for x in s if x['phase']!='TERMINAL_HOLD');row['duration_right_censored']='Horizon' in str(r['error']);row['planner_calls']=sum('position' in z for z in d['plans']);row['planning_wall_s']=sum(z.get('planning_wall_s',0) for z in d['plans']);row['refresh_count']=sum(z['phase']=='ATTACHMENT_REFRESH' for z in d['plans'])
   if boundary:
    seq=s[boundary:];ref=torch.tensor(s[boundary-1]['tcp_T_tool']).reshape(1,7);tool=torch.tensor([z['tool'] for z in seq]);palm=torch.tensor([z['tcp'] for z in seq]);dp,da=in_hand_errors(tool,palm,ref);flags=(dp<=.015)&(da<=np.deg2rad(5));acc=StabilityAccumulator(1,len(seq)*.05,.05)
    for flag in flags:acc.update(flag.reshape(1))
    contacts=[z for x in s for z in x['physics_substep_contacts'] if z['control_step']>boundary and z['force_norms_N']['held_tool']>.5];first=contacts[0] if contacts else None;before=100 if first is None else first['control_step']-1;n=max(0,before-boundary)
    row.update(passive_g2a_survival_fraction=float(acc.fractions[0]),relative_translation_peak_mm=float(dp.max()*1000),relative_rotation_peak_deg=float(da.max()*180/np.pi),pre_first_tool_cube_contact_rotation_peak_deg=None if n==0 else float(da[:n].max()*180/np.pi),first_tool_cube_contact=first,drift_series_mm=(dp*1000).tolist(),drift_series_deg=(da*180/np.pi).tolist())
   rows.append(row)
 # Strict prefix comparison across physically measured interventions, excluding reset contact buffer.
 for seed,mass,g in sorted({(r['seed'],r['mass'],r['grasp']) for r in rows if r['group']=='tool_effect'}):
  rr=[r for r in rows if r['group'] in ['tool_effect','tool_controls'] and (r['seed'],r['mass'],r['grasp'])==(seed,mass,g)];base=next(r for r in rr if r['mode']=='baseline');b=json.load(gzip.open(base['path'],'rt'))
  for r in rr:
   if r['mode']=='baseline':continue
   d=json.load(gzip.open(r['path'],'rt'));end=base['boundary'];same=bool(end and r['boundary']==end and b['steps'][:end]==d['steps'][:end]);pairs.append(dict(seed=seed,mass=mass,grasp=g,mode=r['mode'],exact_prefix=same,initial_equal=r['initial_sha256']==base['initial_sha256'],rescue=r['success'] and not base['success'],regression=base['success'] and not r['success']))
 scores={};opportunity={}
 for folder in ['tool_effect','tool_controls','train108','evaluation','tool_all_contacts','retained100','endpoint_fixed','endpoint_dense']:
  rr=[r for r in rows if r['group']==folder];scores[folder]={k:dict(n=len([x for x in rr if (x['method'] or x['mode'])==k]),success=sum(x['success'] for x in rr if (x['method'] or x['mode'])==k)) for k in sorted({x['method'] or x['mode'] for x in rr})}
  if folder=='train108':
   contexts=collections.defaultdict(list)
   for r in rr:contexts[r['seed'],r['mass']].append(r)
   opportunity=dict(context_hist=dict(collections.Counter(sum(x['success'] for x in v) for v in contexts.values())),context_count=len(contexts),best_fixed=max(sum(x['success'] for x in rr if (x['grasp'],x['mode'])==k) for k in {(x['grasp'],x['mode']) for x in rr}),oracle=sum(any(x['success'] for x in v) for v in contexts.values()))
 (a.out/'MECHANISM.json').write_text(json.dumps(dict(scores=scores,opportunity=opportunity,pairs=pairs,rows=rows,g2a_source_commit='4b4a4298f6eec3ba6220427d5c863c24cd2529d0',g2a_scope='actual library relative-palm error + irreversible StabilityAccumulator; passive 20Hz observed rollout, 15mm/5deg default tolerances; not imposed12-wrench LEAP benchmark; not task success'),indent=2))
 print(json.dumps(dict(scores=scores,opportunity=opportunity,prefix_exact=sum(x['exact_prefix'] for x in pairs),pairs=len(pairs)),indent=2))
if __name__=='__main__':main()
