"""Build compatible transition data and reference-policy suffix evidence."""
import copy,hashlib,json
from pathlib import Path
import torch

def build(dirs,model):
 episodes=[];pairs=[];sources={}
 for folder in dirs:
  folder=Path(folder);episodes+=torch.load(folder/'REPLAY.pt',weights_only=False);pairs+=json.loads((folder/'PAIRS.json').read_text());sources[str(folder)]=hashlib.sha256((folder/'REPLAY.pt').read_bytes()).hexdigest()
 versions=set(p['reference_sha']for p in pairs);assert len(versions)==1
 data=[];anchors=[]
 # One real suffix can be shared by two pairs; deduplicate by trace path and step.
 paths={}
 for p in pairs:
  for key in ('baseline_path','branch_path'):paths[p[key]]=min(paths.get(p[key],100),p['step'])
 for e in episodes:
  assert sum(t['duration']for t in e['ts'])==100
  data.extend(copy.deepcopy(e['ts']))
  if e['row']['success'] and e['row']['path']in paths:
   for t in e['ts']:
    if t['stage']!=1 or t['step']<=paths[e['row']['path']]:continue
    s=t['s'];b,m=model.context([s])
    with torch.no_grad():pr=model.probs(model.norm([s]),b,m,[s])[0].tolist()
    anchors.append(dict(s=s,reference_p=pr,path=e['row']['path'],step=t['step']))
 return dict(data=data,pairs=pairs,anchors=anchors,reference_sha=next(iter(versions)),episodes=len(episodes),sources=sources,anchor_semantics='soft reference distribution is a surrogate drift measure; physical continuation used argmax, no safety bound')
