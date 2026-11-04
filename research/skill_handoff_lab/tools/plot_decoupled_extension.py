"""Matched continuation/repeat curves; cumulative episode counts explicit."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();s=json.loads((a.root/'SUMMARY.json').read_text())['arms'];f=a.root/'figures';f.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,seed,suffix in zip(axes,[31601,31602],['extend64','repeat128']):
 for prefix,label,col in [('measured','Measured reference','#a44a3f'),('relative','Active target','#246a73')]:
  n=f'{prefix}_bc_{suffix}'
  if n not in s:continue
  e=s[n]['evaluation'];ks=sorted(int(k)for k,v in e.items()if v['n']==16)
  ax.plot([k+64 if seed==31601 else k for k in ks],[e[str(k)]['success']for k in ks],marker='o',color=col,label=label)
 ax.axhline(9,color='gray',lw=.8,ls=':');ax.set(title=f'Replay/exploration seed{seed}',xlabel='Cumulative own-policy episodes',ylim=(0,16));ax.legend(fontsize=8)
axes[0].set_ylabel('Native success / 16 seen development states');fig.suptitle('SAC + success BC: matched interfaces and training budgets');fig.tight_layout();fig.savefig(f/'matched_extension_repeat.png',dpi=220);fig.savefig(f/'matched_extension_repeat.pdf')
