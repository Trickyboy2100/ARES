"""Standalone data-derived figures; no idealized robot trajectories."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(a.out/(name+'.'+ext),dpi=220)
 if len(fig.axes)>1:
  fig.canvas.draw()
  for i,ax in enumerate(fig.axes):
   box=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03,1.05)
   for ext in ['png','pdf']:fig.savefig(a.out/(name+f'_panel{i+1}.'+ext),dpi=220,bbox_inches=box)
 plt.close(fig)
def run_style(run):
 method=run.split('_')[0];seed=run.rsplit('_',1)[1];names={'sac':'Flat SAC','factor':'Fixed reference/offset','supervised':'Outcome supervision','chart':'Target-chart SAC'};colors={'sac':'#287a98','factor':'#bd783b','supervised':'#8172a6','chart':'#47946c'}
 return dict(label=names.get(method,method)+' /'+seed,color=colors.get(method,'#555555'),linestyle='--'if seed=='32202'else'-')
rows=sum([json.loads(f.read_text())for f in a.root.glob('design_*/RESULTS.json')],[]);seeds=sorted(set(r['seed']for r in rows));labels=['warm']+[f'a{k}'for k in range(12)];mat=np.array([[next(r['success']for r in rows if r['seed']==s and r['label']==label)for s in seeds]for label in labels],int)
fig,ax=plt.subplots(figsize=(11,7));ax.imshow(mat,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(seeds)),[str(s)[-3:]for s in seeds],rotation=45);names=['Warm','Continue','Refresh','Local Y+','Local Y−','Local Z+','Local Z−','Withdraw','Hold','Refresh Y+','Refresh Y−','Refresh Z+','Refresh Z−'];ax.set_yticks(range(13),names)
for y in range(13):
 for x in range(16):ax.text(x,y,str(mat[y,x]),ha='center',va='center',color='white'if mat[y,x]else'#333',fontsize=9)
ax.set(xlabel='Training-design scene765xxx; original reset per cell',title='Complete native Peg100 success: first intervention + common warm continuation');save(fig,'physical_frontier')
fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
for ax,seed in zip(axes,[765003,765010]):
 for physical,label,color in [(2,'Current-reference Y+','#2672a3'),(8,'Refreshed-reference Y+','#d17b26')]:
  r=next(v for v in rows if v['seed']==seed and v['label']==f'a{physical}');d=json.load(gzip.open(r['path'],'rt'));states=[d['initial']]+d['steps'];x=[s['evaluation']['peg_head_pos_at_hole'][0][0]*1000 for s in states];ax.plot(range(101),x,label=f'{label}: success={r["success"]}',color=color)
 ax.axvline(r['entry']['step'],color='#888',ls='--');ax.set(ylabel='Measured axial position (mm)',title=f'Scene{seed}: identical prefix to step{r["entry"]["step"]}, native100 unchanged');ax.legend(fontsize=9)
axes[-1].set_xlabel('Actual physical control step');save(fig,'reference_conditional_effect')
fig,ax=plt.subplots(figsize=(11,4));ax.axis('off');boxes=[(.02,.5,'Task / initial state\n3 learned real grasps'),(.27,.5,'Physical grasp\nactual SkillResult'),(.52,.5,'Shared catalogue\nrule chunk +12 physical Options'),(.79,.5,'Atomic execution\nmeasured duration / release')]
for x,y,t in boxes:ax.text(x,y,t,transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.5',fc='#e3f0f4',ec='#285974'),va='center',fontsize=10)
for x in [.22,.47,.75]:ax.annotate('',xy=(x+.035,.5),xytext=(x-.015,.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#285974'))
ax.annotate('Observed state + remaining budget; reopen only at actual boundary',xy=(.56,.34),xytext=(.9,.16),xycoords='axes fraction',ha='right',arrowprops=dict(arrowstyle='->',connectionstyle='angle3,angleA=90,angleB=0'))
ax.text(.04,.85,'One continuous native task: no object reset at handoff, no extra time',transform=ax.transAxes,fontsize=13);ax.text(.04,.06,'physical0 = continue active queue     |     selector0 = execute frozen rule block',transform=ax.transAxes,fontsize=11);save(fig,'skill_chain')
# Curves are generated only when actual histories exist.
fig,ax=plt.subplots(figsize=(9,5));found=False
for f in sorted(a.root.glob('*_322*/HISTORY.json')):
 h=json.loads(f.read_text());curve={}
 for v in h:
  if v['kind']=='eval':curve.setdefault(v['checkpoint'],[]).append(v['row']['success'])
 curve={k:v for k,v in curve.items()if len(v)==24}
 if curve:found=True;ax.plot(sorted(curve),[sum(curve[k])for k in sorted(curve)],marker='o',**run_style(f.parent.name))
ax.set(xlabel='Own-policy training episodes',ylabel='Complete native successes /24',ylim=(0,24),title='Fixed development765400–423; matched catalogue')
if found:ax.legend(fontsize=9);save(fig,'learning_curves')
else:plt.close(fig)
if (a.root/'LEARNING_SUMMARY.json').exists():
 summary=json.loads((a.root/'LEARNING_SUMMARY.json').read_text());groups=summary['groups'];baseline=groups.get('sac_32201@0')
 selected=[k for k in groups if k.startswith('dev_')]
 for run in summary['learning']:
  keys=[k for k in groups if k.startswith(run+'@') and groups[k]['n']==24]
  if keys:selected.append(max(keys,key=lambda k:int(k.rsplit('@',1)[1])))
 selected=sorted(selected)
 if baseline and selected:
  fig,ax=plt.subplots(figsize=(10,max(4,.47*len(selected)+1.6)));rr=[];gg=[]
  for k in selected:
   b=baseline['outcomes'];v=groups[k]['outcomes'];rr.append(sum(v[s] and not b[s] for s in v));gg.append(sum(b[s] and not v[s] for s in v))
  y=np.arange(len(selected));ax.barh(y,rr,color='#287a98',label='Rescued initial failure');ax.barh(y,-np.array(gg),color='#c8654c',label='Lost initial success');ax.set_yticks(y,[f'{k} ({groups[k]["success"]}/{groups[k]["n"]})' for k in selected]);ax.axvline(0,color='#555',lw=.8);ax.set(xlabel='Paired scene count against shared warm initialization (19/24)',title='Actual complete-task rescue and regression — seen development');ax.legend(loc='lower right',fontsize=9);save(fig,'frozen_pairings')
 fig,ax=plt.subplots(figsize=(10,5))
 for run,data in summary['learning'].items():
  ys=np.array([v['success'] for v in data['curve']],float)
  if len(ys)>=16:ax.plot(np.arange(16,len(ys)+1),np.convolve(ys,np.ones(16)/16,mode='valid'),**run_style(run))
 ax.axvline(64,color='#888888',ls=':',lw=1,label='Warm behavior mixing ends')
 ax.set(xlabel='Own physical training episodes',ylabel='Rolling16 native success fraction',ylim=(0,1),title='Actual exploratory behavior; not greedy deployment performance');ax.legend(fontsize=9);save(fig,'online_behavior')

 if 'confirm_warm' in groups:
  selected=[k for k in groups if k.startswith('confirm_')];b=groups['confirm_warm']['outcomes']
  if all(set(groups[k]['outcomes'])==set(b) for k in selected):
   fig,ax=plt.subplots(figsize=(10,5));y=np.arange(len(selected));rr=[];gg=[]
   for k in selected:
    v=groups[k]['outcomes'];rr.append(sum(v[s]and not b[s]for s in b));gg.append(sum(b[s]and not v[s]for s in b))
   ax.barh(y,rr,color='#287a98',label='Rescue');ax.barh(y,-np.array(gg),color='#c8654c',label='Regression');ax.axvline(0,color='#555',lw=.8)
   ax.set_yticks(y,[f'{k.removeprefix("confirm_")} ({groups[k]["success"]}/{groups[k]["n"]})' for k in selected]);ax.set(xlabel=f'Paired scene count versus warm ({sum(b.values())}/{len(b)})',title='Frozen policies on new natural conditions765600–631');ax.legend();save(fig,'confirmation_pairings')
if (a.root/'POSTCONFIRM_SUPPORT_EFFECT.json').exists():
 effects=json.loads((a.root/'POSTCONFIRM_SUPPORT_EFFECT.json').read_text())['effects']['32202'];fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
 for ax,seed in zip(axes,['765627','765604']):
  pair=effects['pairs'][seed]
  for key,color,label in [('restricted','#888888','Old proposal support'),('full','#287a98','Full executable catalogue')]:
   row=pair[key];d=json.load(gzip.open(row['path'],'rt'));states=[d['initial']]+d['steps'];ax.plot(range(101),[x['evaluation']['peg_head_pos_at_hole'][0][0]*1000 for x in states],color=color,label=f'{label}: success={row["success"]}')
  ax.axvline(pair['exact_physical_prefix'],ls='--',color='#ba6f47');ax.set(title=f'Scene{seed}: same frozen SAC32202; common physical prefix{pair["exact_physical_prefix"]}',ylabel='Peg-head axial coordinate (mm)');ax.legend(fontsize=9)
 axes[-1].set_xlabel('Actual native physical control step');save(fig,'postconfirm_support_physics')
