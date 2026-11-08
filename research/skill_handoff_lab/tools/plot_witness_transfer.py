"""Recorded-choice margin versus actual new-state native outcomes (seen dev)."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
names=['sac_32301','retention_32301','witness_soft_32301','witness_qp_32301'];labels=['SAC + BC','Reference KL','Soft support','QP support'];colors=['#3267a8','#cd7738','#66883d','#008c8c'];allm=json.loads((a.root/'WITNESS_MARGINS.json').read_text());rows=[next(x for x in allm if x['policy']==n and x['checkpoint']==64 and x['group']=='requires_nonzero')for n in names];scores=[]
for n in names:
 h=json.loads((a.root/n/'HISTORY.json').read_text());b={x['row']['seed']:x['row']['success']for x in h if x['kind']=='eval'and x['checkpoint']==0};v={x['row']['seed']:x['row']['success']for x in h if x['kind']=='eval'and x['checkpoint']==64};assert len(v)==16;scores.append(dict(success=sum(v.values()),rescue=sum(v[s]and not b[s]for s in v),regression=sum(b[s]and not v[s]for s in v)))
def gaps(ax):
 bp=ax.boxplot([r['margins']for r in rows],tick_labels=labels,patch_artist=True,showfliers=True)
 for patch,c in zip(bp['boxes'],colors):patch.set_facecolor(c);patch.set_alpha(.65)
 ax.axhline(0,c='#444',ls=':',lw=1);ax.set(ylabel='Allowed-versus-forbidden logit gap',title='18 training witnesses requiring nonzero correction');ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelsize=9)
def outcomes(ax):
 x=np.arange(4);ax.bar(x-.17,[z['rescue']for z in scores],.34,color='#008c8c',label='Rescue');ax.bar(x+.17,[z['regression']for z in scores],.34,color='#bb5c52',label='Regression');ax.set(xticks=x,xticklabels=labels,ylabel='Native tasks / 16',title='Separate seen development | full native execution',ylim=(0,4.5),yticks=[0,1,2,3,4]);ax.tick_params(axis='x',labelsize=9);ax.legend();ax.grid(axis='y',alpha=.2)
 for i,z in enumerate(scores):ax.text(i,3.5,f"Success {z['success']}/16",ha='center',fontsize=9)
fig,axs=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained');gaps(axs[0]);outcomes(axs[1]);fig.suptitle('Same seed and 64-episode budget | finite witness preservation and native transfer')
for ext in ['png','pdf']:fig.savefig(a.out/f'WITNESS_TRANSFER.{ext}',dpi=200)
plt.close(fig)
for name,draw in [('WITNESS_MARGINS',gaps),('WITNESS_NATIVE_OUTCOMES',outcomes)]:
 fig,ax=plt.subplots(figsize=(6,4.3),layout='constrained');draw(ax)
 for ext in ['png','pdf']:fig.savefig(a.out/f'{name}.{ext}',dpi=200)
 plt.close(fig)
(a.out/'WITNESS_TRANSFER_SOURCES.json').write_text(json.dumps(dict(margins=rows,outcomes=scores,policies=names,checkpoint=64,scope='Exploratory mechanism analysis; boxplots describe18 recorded states, not confidence intervals or causal proof of margin effect'),indent=2))
