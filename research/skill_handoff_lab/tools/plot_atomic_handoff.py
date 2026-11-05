"""Scientific plots from completed native records; never interpolate successes."""
import argparse,json,gzip
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def resolve_raw(path, root):
 source=Path(path)
 if source.exists():return source
 for n in range(2,len(source.parts)):
  candidate=root.joinpath(*source.parts[-n:])
  if candidate.is_file():return candidate
 raise FileNotFoundError(path)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'figures';out.mkdir(exist_ok=True);s=json.loads((a.root/'SUMMARY.json').read_text())
fig,ax=plt.subplots(figsize=(9,4.5))
for name,v in s['training'].items():
 c=[(int(k),z['success'])for k,z in v['curve'].items()if z['n']==16]
 if c:ax.plot(*zip(*c),marker='o',label=name)
ax.set(xlabel='Own-policy native training episodes',ylabel='Full task success / 16',ylim=(0,16),title='Same action/state interface; actual temporal policy comparison');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'learning.png',dpi=200);fig.savefig(out/'learning.pdf');plt.close(fig)
r=json.loads((a.root/'h1_witness76/RESULTS.json').read_text());chosen=['continue','atomic_once','atomic_option','atomic_phase'];fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
for name in chosen:
 z=next(x for x in r if x['branch']==name);d=json.load(gzip.open(resolve_raw(z['path'],a.root),'rt'));x=[v['evaluation']['peg_head_pos_at_hole'][0][0]*1000 for v in d['steps']];axes[0].plot(range(1,101),x,label=f"{name}: {'success' if z['success']else 'failure'}")
 ev=[v for v in z['decisions']if v['stage'] and v['id']];axes[1].scatter([v['step']for v in ev],[chosen.index(name)]*len(ev),s=65)
axes[0].axvline(76,color='k',ls=':');axes[0].set(ylabel='Peg head axial position (mm)',title='Same recorded76-step prefix; repeated intervention matters');axes[0].legend(fontsize=8);axes[1].set(yticks=range(len(chosen)),yticklabels=chosen,xlabel='Native physical step',xlim=(55,100));axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(out/'witness_timing.png',dpi=200);fig.savefig(out/'witness_timing.pdf');plt.close(fig)
# Learned rescue: compare actual trajectories, not the nominal output of a model.
fig,ax=plt.subplots(figsize=(9,4.5))
for name,k in [('macro_bc64',0),('queue_bc64',32),('macro_bc64',32),('macro_bc64',64)]:
 h=json.loads((a.root/name/'HISTORY.json').read_text());z=next(v['row']for v in h if v['job']==dict(kind='eval',checkpoint=k,seed=761202));d=json.load(gzip.open(resolve_raw(z['path'],a.root),'rt'));ax.plot(range(1,101),[v['evaluation']['peg_head_pos_at_hole'][0][0]*1000 for v in d['steps']],label=f'{name} cp{k}: {int(z["success"])}')
ax.set(xlabel='Native physical step',ylabel='Peg head axial position (mm)',title='761202: intervention timing and actual learned rescue');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'learned_rescue.png',dpi=200);fig.savefig(out/'learned_rescue.pdf');plt.close(fig)
