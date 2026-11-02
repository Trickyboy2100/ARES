"""Physical paired contrasts, common prefixes and budget accounting (no simulator)."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np
from audit_recovery_sources import rotation

def trace(row):
 p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];return json.load(gzip.open(p,'rt'))
def difference(a,b,n):
 keys=['q','qvel','tcp','tool','cube','action'];return max([0.]+[float(np.max(np.abs(np.array(x[k])-y[k])))for x,y in zip(a['steps'][:n],b['steps'][:n])for k in keys])
def first_contact(d):
 prev=d['initial_measure']
 for s in d['steps']:
  force=max([z['force_norms_N']['held_tool']for z in s.get('physics_substep_contacts',[])]+[s['tool_cube_force_N']]);motion=np.linalg.norm(np.array(s['cube'][:2])-prev['cube'][:2]);prev=s
  if s['phase']in ['TRANSFER','HOOK','PULL']and force>.5 and motion>.0001:return s['step']
 return None

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();groups={}
 for f in sorted(a.root.glob('*/RESULTS.json')):
  rows=json.loads(f.read_text())
  if not isinstance(rows,list):continue
  tool=f.parent.name.startswith('tool');key='mode'if tool else 'method'if rows and 'method'in rows[0] else 'policy';base='none'if tool else 'nominal';contexts=collections.defaultdict(list);checks=[];counts={}
  for row in rows:
   d=trace(row);assert len(d['steps'])==(100 if tool else 200);assert bool(d['steps'][-1]['native_success'])==bool(row['success']);contexts[(row['seed'],None if key=='method'else row.get('grasp',row.get('angle')))].append((row,d))
  for context,entries in contexts.items():
   nom=next((d for r,d in entries if r[key]==base),None)
   if nom is None:continue
   for r,d in entries:
    boundary=first_contact(nom)if tool and 'early'in f.parent.name else next((x['boundary_step']for x in nom.get('decisions',[])if x['skill']=='RECOVER_CONTACT'),None)if tool else nom['summary']['prefix_steps']
    same_grasp=d['summary'].get('angle')==nom['summary'].get('angle')
    if key=='method'and not same_grasp:boundary=0
    delta=difference(nom,d,boundary or 0)
    assert d['summary']['initial_sha256']==nom['summary']['initial_sha256']
    # Preserve stochastic planner-prefix failures; never classify them as paired rescue.
    checks.append(dict(context=list(context),method=r[key],boundary=boundary,prefix_max_error=delta,paired=delta<1e-6,comparison='FULL_TASK_SAME_INITIAL'if key=='method'and not same_grasp else 'SAME_PHYSICAL_PREFIX',success=r['success'],rescue=not nom['summary']['success']and r['success'],regression=nom['summary']['success']and not r['success'],first_contact_step=first_contact(d),source=r['path']))
  for method in sorted({r[key]for r in rows}):
   rr=[r for r in rows if r[key]==method];cc=[c for c in checks if c['method']==method and c['paired']];counts[method]=dict(success=sum(x['success']for x in rr),n=len(rr),rescues=sum(x['rescue']for x in cc),regressions=sum(x['regression']for x in cc))
  groups[f.parent.name]=dict(methods=counts,episodes=len(rows),physics_steps=sum(x['steps']for x in rows),planner_queries=sum(x.get('planner_queries',x.get('plan_calls',0))for x in rows),planning_wall_s=sum(x.get('query_wall_s',x.get('planning_wall_s',0))for x in rows),wall_s=sum(x['wall_s']for x in rows),checks=checks)
 out=dict(groups=groups,method_status='UNVERIFIED',zero_nominal_source='525d67e',early_event_note='Original early none logger overwrote first event inside path; raw retained. first_contact_step is recomputed from actual interval motion/force for EVERY trace; corrected future logger guards event is None.',episodes=sum(x['episodes']for x in groups.values()),physical_steps=sum(x['physics_steps']for x in groups.values()));(a.root/'ANALYSIS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v['methods']for k,v in groups.items()},indent=2))
if __name__=='__main__':main()
