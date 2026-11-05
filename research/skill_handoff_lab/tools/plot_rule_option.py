"""Scientific figures from recomputed native simulation results only."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
s=json.loads((a.root/'SUMMARY.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.bbox':'tight'})
def save(fig,name):
    fig.savefig(a.out/(name+'.png'));fig.savefig(a.out/(name+'.pdf'));plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3.8));e=s['pairs']['effects'];v=[e.get('1',0),e.get('-1',0),e.get('0',0)]
bars=ax.bar(['Rescue','Regression','Same outcome'],v,color=['#16856b','#c44848','#8a99a7']);ax.bar_label(bars,padding=3);ax.set(ylabel='Natural training initial states',ylim=(0,64),title='One complete Option versus frozen rule\n64 pairs; 128 complete native episodes');save(fig,'paired_effects')
qfile=a.root/'OLD_Q_CALIBRATION.json'
if qfile.exists():
    q=json.loads(qfile.read_text());fig,ax=plt.subplots(figsize=(6.5,3.8))
    for effect,color in [(-1,'#c44848'),(0,'#8a99a7'),(1,'#16856b')]:
        z=[v for v in q if v['delta']==effect];ax.scatter([v['old_Q_gap']for v in z],[effect]*len(z),color=color,alpha=.65,s=38)
    ax.axvline(0,color='black',ls=':',lw=1);ax.set(xlabel='Frozen historical mean-Q(candidate) − mean-Q(rule)',yticks=[-1,0,1],yticklabels=['Regression','Same outcome','Rescue'],ylim=(-1.4,1.4),title='Historical Q is not the measured rule-relative Option value');save(fig,'legacy_q_mismatch')
pair=next(v for v in json.loads((a.root/'pairs/PAIRS.json').read_text())if v['seed']==763039)
fig,ax=plt.subplots(figsize=(7,4))
for key,color,label in [('baseline','#16856b','Frozen rule: native success'),('candidate','#c44848','One queue-continue deviation: native failure')]:
    d=json.load(gzip.open(pair[key]['path'],'rt'));ax.plot([v['step']for v in d['steps']],[v['evaluation']['peg_head_pos_at_hole'][0][0]*1000 for v in d['steps']],color=color,label=label)
ax.axvspan(79,83,color='#e8b456',alpha=.3,label='Recorded deviation duration: 4 steps');ax.set(xlabel='Actual native physical step',ylabel='Axial peg head in hole frame (mm)',title='763039: rejecting the rule is not harmless continuation\n79-step identical physical prefix; original 100-step deadline');ax.legend(fontsize=8);save(fig,'continue_can_harm_rule')
fig,axes=plt.subplots(1,2,figsize=(10,3.9),sharey=True)
for ax,version in zip(axes,['v1','v2']):
    for name,t in s['training'].items():
        if (name.startswith('v2_')) != (version=='v2'):continue
        if 'entry' in name:continue
        c=t['curve'];xs=[int(x)for x in c if c[x]['n']==16];ys=[c[str(x)]['success']for x in xs]
        rl='stopping' in name;seed=name.rsplit('_',1)[-1]
        ax.plot(xs,ys,marker='o',color='#226bb1'if rl else '#c27224',linestyle='-'if seed=='31801'else '--',label=('Sequence RL'if rl else 'Supervised')+' '+seed)
    if 'dev_rule'in s['controls']:ax.axhline(s['controls']['dev_rule']['success'],color='#555',ls=':',label='Frozen rule')
    ax.set(xticks=[0,32,64],ylim=(0,16),xlabel='Own-policy training episodes',title=version+(' | warm 500 updates'if version=='v1'else ' | warm 2000 updates'));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Complete native successes / 16 dev scenes');fig.suptitle('Fixed final64; common data and equal online budgets',y=1.04);save(fig,'learning_curves')
fig,ax=plt.subplots(figsize=(8,4));names=list(s['training']);x=np.arange(len(names));resc=[len(s['training'][n]['rescue'])for n in names];harm=[len(s['training'][n]['regression'])for n in names]
ax.bar(x-.18,resc,.36,color='#16856b',label='Rescued vs same-initial rule');ax.bar(x+.18,harm,.36,color='#c44848',label='Regressed vs same-initial rule');ax.set_xticks(x,[n.replace('stopping_rl','RL').replace('stopping','RL').replace('supervised','SUP').replace('_','\n')for n in names]);ax.set(ylabel='Online training episodes',title='Exploratory training outcomes, not deployed-policy gains');ax.legend(fontsize=9);save(fig,'online_rescue_regression')
