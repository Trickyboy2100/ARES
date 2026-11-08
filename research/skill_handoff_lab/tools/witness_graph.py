"""Training-only action support from complete executed success witnesses.
History identity includes recorded prefix/queue; the Actor still sees83 features.
No runtime lookup, state restoration, or unmeasured success labels.
"""
import json,gzip,hashlib
from pathlib import Path
import torch
from program_rule_option import prefix_hash

def build_graph(dirs):
 episodes=[];pairs=[]
 for folder in map(Path,dirs):
  episodes+=torch.load(folder/'REPLAY.pt',weights_only=False);pairs+=json.loads((folder/'PAIRS.json').read_text())
 entry={}
 for p in pairs:
  key=(p['baseline_path'],p['step'])
  e=entry.setdefault(key,dict(step=p['step'],s=p['s'],baseline=p['baseline'],verified=set(),fallback=set(),paths=set()))
  if p['baseline_success']:e['verified'].add(p['baseline'])
  else:e['fallback'].add(p['baseline'])
  if p['success']:e['verified'].add(p['action'])
  e['paths'].update([p['baseline_path'],p['branch_path']])
 by_path={path:e for e in entry.values()for path in e['paths']};nodes={}
 for ep in episodes:
  if not ep['row']['success']:continue
  path=ep['row']['path'];d=json.load(gzip.open(path,'rt'));dec=d['summary']['decisions'];e=by_path.get(path)
  for t in ep['ts']:
   if t['stage']!=1:continue
   n=t['step'];z=next(x for x in dec if x['step']==n and x['obs']==t['s'])
   identity=dict(prefix=prefix_hash(d,n),queue=z['queue_before'],target=z['active_target_before'],obs=t['s'])
   key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
   node=nodes.setdefault(key,dict(s=t['s'],verified=set(),fallback=set(),sources=[]))
   if e and n==e['step'] and t['s']==e['s']:
    node['verified'].update(e['verified']);node['fallback'].update(e['fallback'])
   else:node['verified'].add(t['a'])
   node['sources'].append(dict(path=path,step=n,actual_action=t['a']))
 for key,n in nodes.items():
  n['history_sha']=key;n['verified']=sorted(n['verified']);n['fallback']=sorted(n['fallback']);n['allowed']=sorted(set(n['verified'])|set(n['fallback']))
 return dict(nodes=list(nodes.values()),scope='finite recorded witness choices only; no unseen-state or stochastic-physics safety guarantee',physical_episodes=len(episodes),pairs=len(pairs))

def merge_graph(old,new):
 nodes={n['history_sha']:dict(n)for n in old['nodes']}
 for n in new['nodes']:
  key=n['history_sha']
  if key not in nodes:nodes[key]=dict(n);continue
  p=nodes[key];assert p['s']==n['s']
  for field in ('verified','fallback','allowed'):p[field]=sorted(set(p[field])|set(n[field]))
  p['sources']=p['sources']+n['sources']
 return dict(nodes=list(nodes.values()),scope=old['scope'],physical_episodes=old['physical_episodes']+new['physical_episodes'],pairs=old['pairs']+new['pairs'])
