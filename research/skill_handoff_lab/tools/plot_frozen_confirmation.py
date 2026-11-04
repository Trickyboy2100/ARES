import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();s=json.loads((a.root/'CONFIRMATION_SUMMARY.json').read_text())['pooled'];names=['measured_31601','remaining_target_31601','measured_31602','remaining_target_31602','measured_frozen_rule','remaining_target_frozen_rule','measured_nominal'];labels=['Measured\nseed31601','Active target\nseed31601','Measured\nseed31602','Active target\nseed31602','Measured\nfixed rule','Active target\nfixed rule','Nominal\ncontinue'];colors=['#a44a3f','#246a73','#a44a3f','#246a73','#adb5bd','#879b9d','#203040'];f=a.root/'figures'
fig,ax=plt.subplots(figsize=(10,4.5));bars=ax.bar(range(7),[s[n]['success']for n in names],color=colors);ax.bar_label(bars,padding=3);ax.set_xticks(range(7),labels,fontsize=9);ax.set(ylim=(0,32),ylabel='Native Peg100 success / 32',title='Frozen policies on 32 new natural initial states');ax.text(.01,.96,'Two cohorts reported separately; cohort 2 was an adaptive extension',transform=ax.transAxes,fontsize=9,va='top');fig.tight_layout();fig.savefig(f/'frozen_confirmation32.png',dpi=220);fig.savefig(f/'frozen_confirmation32.pdf');plt.close(fig)
lines=[r'\begin{tabular}{lrrrr}',r'\hline',r'Method & Success/32 & Rescue & Regression & Plan queries \\',r'\hline']
for n,label in zip(names,labels):
 v=s[n];lines.append(label.replace('\n',' ') + f" & {v['success']} & {len(v['rescue'])} & {len(v['regression'])} & {v['planning_queries']}"+r' \\')
lines+=[r'\hline',r'\end{tabular}',r'% Frozen models; second16-state cohort adaptively added. Common inherited weights; two replay/exploration seeds.'];(a.root/'FROZEN_RESULTS.tex').write_text('\n'.join(lines)+'\n')
