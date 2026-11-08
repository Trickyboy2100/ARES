"""Frozen native confirmation: retain individual network scores and paired failures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
s=json.loads((a.root/'analysis/SUMMARY.json').read_text());g=s['groups'];arms=['sac','retention','ranking','witness_soft','witness_qp'];labels=['SAC + BC','Reference KL','Supervised rank','Soft support','QP support'];colors=['#3267a8','#cd7738','#9a5c93','#66883d','#008c8c']
assert all(g.get(f'confirm_{x}_{k}',{}).get('episodes')==32 for x in arms for k in [32301,32302]);base=g['confirm_warm']['success']
def success(ax):
 for i,(arm,c)in enumerate(zip(arms,colors)):
  y=[g[f'confirm_{arm}_{k}']['success']for k in [32301,32302]];ax.plot([i-.08,i+.08],y,'o-',color=c,lw=1.5);ax.text(i,max(y)+.55,' / '.join(map(str,y)),ha='center',fontsize=9)
 ax.axhline(base,ls=':',c='#555',label=f'Warm {base}/32');ax.axhline(g['confirm_rule']['success'],ls='--',c='#aaa',label=f"Frozen rule {g['confirm_rule']['success']}/32");ax.set(xticks=range(5),xticklabels=labels,ylim=(0,32),ylabel='Native full-task successes / 32',title='Frozen final128 | two network initializations');ax.tick_params(axis='x',rotation=15);ax.legend(loc='lower left');ax.grid(axis='y',alpha=.2)
def paired(ax):
 for j,k in enumerate([32301,32302]):
  x=np.arange(5)+(j-.5)*.36;r=[len(g[f'confirm_{arm}_{k}']['rescue'])for arm in arms];d=[len(g[f'confirm_{arm}_{k}']['regression'])for arm in arms]
  ax.bar(x,r,.32,color='#008c8c',alpha=1 if j==0 else .55,label='Rescue'if j==0 else None);ax.bar(x,-np.array(d),.32,color='#bd5b56',alpha=1 if j==0 else .55,label='Regression'if j==0 else None)
 ax.axhline(0,c='#555',lw=1);ax.set(xticks=range(5),xticklabels=labels,ylabel='Paired tasks versus warm',title='Each pair: seed32301 / seed32302');ax.tick_params(axis='x',rotation=15);ax.legend();ax.grid(axis='y',alpha=.2)
fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');success(axs[0]);paired(axs[1]);fig.suptitle('New natural development768600–631 | shared native100 and executable Options')
for ext in ['png','pdf']:fig.savefig(a.out/f'CONFIRMATION.{ext}',dpi=200)
plt.close(fig)
for name,fn in [('CONFIRMATION_SUCCESS',success),('CONFIRMATION_PAIRED',paired)]:
 fig,ax=plt.subplots(figsize=(6.5,4.7),layout='constrained');fn(ax)
 for ext in ['png','pdf']:fig.savefig(a.out/f'{name}.{ext}',dpi=200)
 plt.close(fig)
(a.out/'CONFIRMATION_SOURCES.json').write_text(json.dumps(dict(groups={k:v for k,v in g.items()if k.startswith('confirm_')},scope='Actual native outcomes, shared32 scenes; two networks are not64 independent tasks. No best-checkpoint selection.'),indent=2))
