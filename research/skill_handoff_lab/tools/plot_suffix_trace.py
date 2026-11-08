"""Actual learning/physical outcomes; independent panels, no idealized curves."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=json.loads(a.summary.read_text());a.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
colors={'sac':'#3478b6','trace':'#d88722','lagged':'#8058a5'}
def save(fig,name):
 fig.tight_layout();fig.savefig(a.out/(name+'.png'),dpi=180);fig.savefig(a.out/(name+'.pdf'));plt.close(fig)
f,ax=plt.subplots(figsize=(6.5,4))
for name,v in d['arms'].items():
 curve=v['curve'];xs=sorted(map(int,curve));ys=[curve[str(x)]['success']/curve[str(x)]['n']*100 for x in xs];ax.plot(xs,ys,'o-',label=name,color=colors[name.split('_')[0]])
ax.set(xlabel='Additional own-policy native episodes',ylabel='Native final success (%)',ylim=(0,100),title='Fixed development scenes; inherited strong checkpoint');ax.legend();save(f,'LEARNING')
f,ax=plt.subplots(figsize=(6.5,4));names=list(d['arms']);x=list(range(len(names)))
for i,n in enumerate(names):
 c=d['arms'][n]['curve'];last=c[str(max(map(int,c)))];ax.bar(i-.18,len(last['rescue']),.35,color='#239b73');ax.bar(i+.18,-len(last['regression']),.35,color='#cf554e')
ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,[n.replace('_32401','')for n in names]);ax.set(ylabel='Rescue (+) / regression (−) vs inherited policy',title='Complete native episodes, same development roots');save(f,'PAIRED')
f,ax=plt.subplots(figsize=(6.5,4))
for n,v in d['arms'].items():
 t=v['training_curve'];ax.plot([r['episode']+1 for r in t],[r['mean_trace_length']for r in t],label=n,color=colors[n.split('_')[0]])
ax.set(xlabel='Own-policy episode',ylabel='Mean sampled backup length (Options)',title='Actual backup activity; not a performance measure');ax.legend();save(f,'TRACE_ACTIVITY')
if d['controls']:
 f,ax=plt.subplots(figsize=(8,4));n=list(d['controls']);ax.bar(range(len(n)),[d['controls'][k]['success']for k in n],color='#3478b6');ax.set_xticks(range(len(n)),[k.replace('confirm_','')for k in n],rotation=25,ha='right');ax.set(ylabel='Complete native successes',title='Frozen final policies on new natural development roots');save(f,'CONFIRMATION')
