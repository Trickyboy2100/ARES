import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
s=json.loads((a.root/'SUMMARY.json').read_text());rows=[x for x in s['table']if x['method']in ['zero','frozen_rule']or x['method'].endswith('/8')]
labels=[x['method'].replace('shared_train/','Round 1: ').replace('joint_train/','Round 2: ').replace('smdp_train/','Round 3: ').replace('/8','').replace('_',' ')for x in rows]
f,ax=plt.subplots(figsize=(10,6));ax.barh(range(len(rows)),[x['success']for x in rows],color=['#607d8b'if x['method']in ['zero','frozen_rule']else '#277da1'if 'option_sac' in x['method'] else '#e98b39'for x in rows]);ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set_xlim(0,16);ax.set_xlabel('Complete native Peg100 successes / 16 natural development states')
for i,x in enumerate(rows):ax.text(x['success']+.15,i,f"{x['success']}/16  (+{len(x['rescue'])}, -{len(x['regression'])})",va='center',fontsize=8)
ax.set_title('Continuation interface works; independent RL gain remains unverified');f.tight_layout();f.savefig(a.out/'RESULTS.png',dpi=180);f.savefig(a.out/'RESULTS.pdf');plt.close(f)
f,ax=plt.subplots(figsize=(8,4))
colors=['#277da1','#43aa8b','#f9844a','#9b5de5']
for run in ['shared_train','joint_train','smdp_train']:
 items=[x for x in s['table']if x['method'].startswith(run+'/')]
 for method in sorted({x['method'].split('/')[1]for x in items}):
  rr=sorted([x for x in items if x['method'].split('/')[1]==method],key=lambda x:int(x['method'].split('/')[-1]));offset={'shared_train':0,'joint_train':16,'smdp_train':32}[run]
  ax.plot([offset+2*int(x['method'].split('/')[-1])for x in rr],[x['success']for x in rr],'o-',label=run.split('_')[0]+': '+method)
ax.axhline(8,color='black',ls='--',label='Frozen rule');ax.set(xlabel='Cumulative shared native training episodes',ylabel='Native successes / 16',ylim=(0,16),title='Fixed checkpoints; new paired experience each round');ax.legend(fontsize=6,ncol=2);f.tight_layout();f.savefig(a.out/'LEARNING.png',dpi=180);plt.close(f)
