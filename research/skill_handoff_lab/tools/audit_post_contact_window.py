"""Raw trace audit: selected vs executed actions and causal contact decision budget."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
for group in ['depth_grid/episodes','accel_grid/episodes','sac_recovered/episodes','evaluation512']:
 for r in json.loads((a.root/group/'RESULTS.json').read_text()):
  f=Path(r['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(f,'rt'));ds=d['decisions'];steps=d['steps'];boundary=ds[1]['boundary_step']if len(ds)>1 else None;event=None
  if boundary is not None:
   start=np.asarray(ds[1]['observation']['cube'][:3])
   for s in steps[boundary:]:
    force=max([v['force_norms_N']['held_tool']for v in s.get('physics_substep_contacts',[])]+[s['tool_cube_force_N']]);motion=float(np.linalg.norm(np.array(s['cube'][:2])-start[:2]))
    if force>.5 and motion>.001 and s['phase']=='PULL':event=s['step'];break
  hook=[s for s in steps if s['phase']=='HOOK'];pull=[s for s in steps if s['phase']=='PULL'];rows.append(dict(group=group,seed=r['seed'],mass=r['mass'],grasp=r['grasp'],height=r.get('hook_height'),success=r['success'],error=r['error'],selected=boundary is not None,hook_executed=len(hook)>0,pull_executed=len(pull)>0,hook_tool_z=None if not hook else hook[-1]['tool'][2],first_effective_pull_contact=event,remaining=None if event is None else 100-event,final_distance=steps[-1]['cube_base_distance_m'],path=str(f),sha256=r['sha256']))
summary={}
for g in dict.fromkeys(r['group']for r in rows):
 rr=[r for r in rows if r['group']==g];budgets=[r['remaining']for r in rr if r['remaining']is not None];summary[g]=dict(n=len(rr),selected=sum(r['selected']for r in rr),hook_executed=sum(r['hook_executed']for r in rr),pull_executed=sum(r['pull_executed']for r in rr),effective_contact=len(budgets),remaining_min=None if not budgets else min(budgets),remaining_median=None if not budgets else float(np.median(budgets)),remaining_max=None if not budgets else max(budgets))
a.out.write_text(json.dumps(dict(definition='causal at end control step: PULL and current/substep force>0.5N and cube XY displacement>1mm since verified boundary; no future success',summary=summary,rows=rows),indent=2));print(json.dumps(summary,indent=2))
