"""Recompute native outcomes, costs and exact-prefix operator contrasts."""
import argparse,collections,gzip,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
allrows=[];arms={};controls={}
for name in ['online_sac','online_success_bc']:
 h=json.loads((a.root/name/'HISTORY.json').read_text());train=[v for v in h if v['job']['kind']=='train'];ev={}
 for cp in [0,32,64]:
  rows=[v['row']for v in h if v['job']['kind']=='eval' and v['job']['checkpoint']==cp];ev[str(cp)]={str(v['seed']):v['success']for v in rows}
 arms[name]=dict(training_episodes=len(train),training_success=sum(v['row']['success']for v in train),training_steps=sum(v['row']['steps']for v in train),effective_training_decisions=sum(v['effective_decisions']for v in train),updates=sum(v['updates']for v in train),front_actions=dict(collections.Counter(v['row']['decisions'][0]['id']for v in train)),evaluation=ev)
 allrows.extend(v['row']for v in h)
for pth in ['controls/RESULTS.json','remaining_plan/RESULTS.json','refresh_only/RESULTS.json']:
 pth=a.root/pth
 if pth.exists():allrows.extend(json.loads(pth.read_text()))
if (a.root/'controls/RESULTS.json').exists():
 for row in json.loads((a.root/'controls/RESULTS.json').read_text()):controls.setdefault(row['method'],{})[str(row['seed'])]=row['success']
for n in ['resume_full','resume_split']:
 allrows.extend(v['row']for v in json.loads((a.root/n/'HISTORY.json').read_text()))
for r in allrows:
 f=Path(r['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
 d=json.load(gzip.open(f,'rt'));assert len(d['steps'])==100 and bool(d['steps'][-1]['evaluation']['success'][0])==r['success']
for arm in arms.values():
 arm['paired']={}
 for name,control in controls.items():
  pred=arm['evaluation']['64'];assert pred.keys()==control.keys()
  arm['paired'][name]=dict(rescues=[s for s in pred if pred[s] and not control[s]],regressions=[s for s in pred if not pred[s] and control[s]])
mech=json.loads((a.root/'remaining_plan/COMPARISONS.json').read_text());mr=json.loads((a.root/'remaining_plan/RESULTS.json').read_text());mechanism={}
for frame,act in [('measured',0),('measured',4),('measured',5),('remaining_target',4),('remaining_target',5)]:
 rows=[v for v in mr if v['frame']==frame and v['intervention']==act];z=[v for v in mech if v['frame']==frame and v['action']==act]
 mechanism[f'{frame}_{act}']=dict(success=sum(v['success']for v in rows),episodes=len(rows),reachable_contexts=len(z),rescues=sum(v['success']and not v['nominal_success']for v in z),regressions=sum(not v['success']and v['nominal_success']for v in z),target_delta_mm=[1000*v['target_delta_m']for v in z if v['target_delta_m']is not None],first4_command_delta_rad=[v['max_first4_command_delta_rad']for v in z])
refresh=[]
if (a.root/'refresh_only/COMPARISONS.json').exists():refresh=json.loads((a.root/'refresh_only/COMPARISONS.json').read_text())
summary=dict(arms=arms,controls=controls,mechanism=mechanism,refresh=refresh,cost=dict(complete_native_episodes=len(allrows),physical_steps=sum(r['steps']for r in allrows),planning_queries=sum(len(r['queries'])for r in allrows),planning_wall_s=sum(q['wall_s']for r in allrows for q in r['queries']),summed_episode_wall_s=sum(r['wall_s']for r in allrows),critic_updates=sum(x['updates']for x in arms.values())+16,actor_updates=sum(x['updates']for x in arms.values())+16,legacy_aux_updates=sum(x['updates']for x in arms.values())+16),claim='UNVERIFIED',taskbook='NOT_FOUND; requested user path; latest explicit message governed this cycle')
(a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['refresh','mechanism']},indent=2))
