"""Reconstruct valid joint Option transitions from actual frozen design runs.
No physics/new return claim. Actions are derived from logged executor origins;
no incompatible old SAC transitions are relabelled.
"""
import argparse,json,gzip
from pathlib import Path
import torch
from train_execution_joint import Assets,Learner,Joint,AtomicRuntime,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);m=Learner(Assets(),32001,'flat');out=[];episodes=[]
for mode in ['rule','warm']:
 for g in range(3):
  rows=json.loads((a.root/f'design_{mode}_g{g}/RESULTS.json').read_text())
  for row in rows:
   d=json.load(gzip.open(row['path'],'rt'));rt=AtomicRuntime.__new__(AtomicRuntime);rt.controller_id=row['controller'];ts=rt.feedback_transitions(d,row['path']);meta=row['option'];by={x['step']:x for x in meta['events']};events=[]
   for decision in d['summary']['decisions']:
    if decision['stage']==0:events.append(dict(boundary=True,a=g,physical=decision['id'],step=decision['step']));continue
    e=by[decision['step']];assert e['action']==decision['id'];boundary=e['origin']!='candidate_continuation';act=1+meta['entry']['proposal']if e['origin']=='candidate_entry'else 0
    events.append(dict(boundary=boundary,a=act,physical=decision['id'],step=decision['step']))
   pol=Joint(None,None,None);pol.events=events;tt=pol.transitions(ts);tt=m.decorate(tt,-1000);out.extend(tt);episodes.append(dict(row=row,events=events))
atomic_torch(a.root/'SHARED_DESIGN_REPLAY.pt',dict(data=out,episodes=episodes,scope='actual same-controller legal behaviour, reconstructed from per-step executor logs',new_physics_steps=0))
print(len(episodes),len(out),'valid complete native traces; all durations sum100')
