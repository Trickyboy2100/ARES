"""Recompute all completed raw physics and training costs, without score filtering."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();groups={};allfiles={}
for f in sorted(a.root.rglob('*.json.gz')):
 d=json.load(gzip.open(f,'rt'))
 if 'steps' not in d or 'summary' not in d:continue
 r=d['summary'];n=len(d['steps']);assert n in [100,200],(f,n)
 final=d['steps'][-1];y=bool(final['evaluation']['success'][0])if 'evaluation'in final else bool(final['native_success']);assert y==bool(r['success'])
 allfiles[str(f.relative_to(a.root))]=dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),steps=n,success=y,seed=r['seed'])
 name=str(f.relative_to(a.root).parts[0]);g=groups.setdefault(name,dict(episodes=0,steps=0,successes=0,planning_queries=0,planning_wall_s=0.,wall_s=0.));g['episodes']+=1;g['steps']+=n;g['successes']+=y;g['planning_queries']+=r.get('plan_calls',len(r.get('queries',[])));g['planning_wall_s']+=r.get('planning_wall_s',sum(q.get('wall_s',0)for q in r.get('queries',[])));g['wall_s']+=r.get('wall_s',0)
training={}
for f in a.root.glob('*/HISTORY.json'):
 h=json.loads(f.read_text());tr=[v for v in h if v['job']['kind']=='train'];curves={}
 nom={v['job']['seed']:v['row']['success']for v in h if v['job']['kind']=='eval'and v['job']['checkpoint']==0}
 for c in sorted(set(v['job']['checkpoint']for v in h if v['job']['kind']=='eval')):
  z=[v for v in h if v['job']['kind']=='eval'and v['job']['checkpoint']==c];curves[c]=dict(n=len(z),success=sum(v['row']['success']for v in z),rescue=[v['job']['seed']for v in z if v['row']['success']and not nom[v['job']['seed']]],regression=[v['job']['seed']for v in z if not v['row']['success']and nom[v['job']['seed']]],front=dict(collections.Counter(v['row']['decisions'][0]['id']for v in z)),rear=dict(collections.Counter(d['id']for v in z for d in v['row']['decisions']if d['stage'])))
 training[f.parent.name]=dict(episodes=len(tr),success=sum(v['row']['success']for v in tr),physical_steps=sum(v['row']['steps']for v in tr),effective_decisions=sum(v['effective_decisions']for v in tr),actor_updates=sum(v['updates']for v in tr),critic_updates=sum(v['updates']for v in tr),unused_aux_updates=sum(v['updates']for v in tr),optimization_wall_s=sum((v.get('log')or{}).get('wall_s',0)for v in tr),curve=curves)
summary=dict(groups=groups,training=training,total={k:sum(v[k]for v in groups.values())for k in ['episodes','steps','planning_queries','planning_wall_s','wall_s']},scope='physical source terminal checks; no sealed data; summed episode wall time not elapsed wall clock')
(a.root/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(a.root/'RAW_INDEX.json').write_text(json.dumps(allfiles,indent=2));print(json.dumps(summary['total']));print(json.dumps(training,indent=2))
