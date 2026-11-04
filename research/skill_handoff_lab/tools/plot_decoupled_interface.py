"""Paper figures from native run summaries; no synthetic performance values."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();s=json.loads((a.root/'SUMMARY.json').read_text());f=a.root/'figures';f.mkdir(exist_ok=True)
styles={'online_sac':('Measured / SAC','#bc4749','--'),'online_success_bc':('Measured / SAC + success BC','#457b9d','--'),'relative_sac64':('Active target / SAC','#bc4749','-'),'relative_bc64':('Active target / SAC + success BC','#457b9d','-')}
fig,ax=plt.subplots(figsize=(7.4,4.3))
for n,(label,col,ls) in styles.items():
 if n not in s['arms']:continue
 e=s['arms'][n]['evaluation'];ks=sorted(int(k)for k,v in e.items()if v['n']==16);ax.plot(ks,[e[str(k)]['success']for k in ks],marker='o',color=col,linestyle=ls,label=label)
ax.axhline(9,color='black',lw=.8,label='Nominal continuation');ax.set(xlabel='Own-policy training episodes',ylabel='Native Peg100 success / 16',ylim=(0,16),xticks=[0,32,64],title='Execution interface × learning rule (seen development)');ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(f/'factorial_learning_curve.png',dpi=220);fig.savefig(f/'factorial_learning_curve.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.4,4.2));labels=[]
for i,(n,(label,col,ls))in enumerate(styles.items()):
 if n not in s['arms']:continue
 e=s['arms'][n]['evaluation'].get('64',{})
 if e.get('n')!=16:continue
 ax.bar(i-.16,len(e['rescue']),.32,color='#2a9d8f');ax.bar(i+.16,-len(e['regression']),.32,color='#e76f51');labels.append((i,label))
ax.axhline(0,color='black',lw=.8);ax.set_xticks([i for i,l in labels],[l.replace(' / ','\n')for i,l in labels],fontsize=8);ax.set(ylabel='Rescues (+) / regressions (−)',title='Final64 paired against nominal; same16 seen initial states');fig.tight_layout();fig.savefig(f/'final64_rescue_regression.png',dpi=220);fig.savefig(f/'final64_rescue_regression.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.4,4.3))
for n,(label,col,ls)in styles.items():
 if n not in s['arms']:continue
 ys=np.array([x['success']for x in s['arms'][n]['episode_curve']],float)
 if len(ys)>=8:ax.plot(np.arange(8,len(ys)+1),np.convolve(ys,np.ones(8)/8,'valid'),color=col,linestyle=ls,label=label)
ax.set(xlabel='Own-policy training episodes',ylabel='Trailing8 episode success',ylim=(0,1),title='Online collection curve (not development evaluation)');ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(f/'online_training_curve.png',dpi=220);plt.close(fig)
