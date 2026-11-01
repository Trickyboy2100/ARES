"""Scientific figures from actual traces; no simulated or fabricated outcomes."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
rows=[]
for f in (a.root/'target_feedback64').glob('*/*/TRAJECTORY.json.gz'):
 d=json.load(gzip.open(f,'rt'));m=json.loads((f.parent/'MANIFEST.json').read_text())
 if m['hook_y'] is not None:rows.append((d['summary'],m))
keys=sorted({(r['seed'],r['density_scale'],r['target_y']) for r,m in rows});params=[(g,h) for g in [.07,.105] for h in [-.055,-.065,-.075]];z=np.zeros((len(keys),len(params)));dist=z.copy()
for r,m in rows:i=keys.index((r['seed'],r['density_scale'],r['target_y']));j=params.index((r['grasp_x'],m['hook_y']));z[i,j]=r['task_success'];dist[i,j]=r['target_distance_m']*1000
fig,ax=plt.subplots(figsize=(9,5));ax.imshow(z,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(6),[f'g={int(g*1000)}, h={int(h*1000)}' for g,h in params],rotation=25,ha='right');ax.set_yticks(range(len(keys)),[f'{s} | mass x{m} | target y={y:+.2f}' for s,m,y in keys]);ax.set_title('Actual paired parameter interventions (mm)\nCell: final task success, target error');
for i in range(len(keys)):
 for j in range(6):ax.text(j,i,f'{int(z[i,j])} / {dist[i,j]:.0f}mm',ha='center',va='center',color='white' if z[i,j] else '#23354a',fontsize=9)
fig.tight_layout();fig.savefig(a.out/'PHYSICAL_PARAMETER_MATRIX.png',dpi=300);fig.savefig(a.out/'PHYSICAL_PARAMETER_MATRIX.svg');plt.close(fig)
errors={};paired={}
for group in ['absolute_paired16','residual_paired16']:
 rs=json.loads((a.root/group/'RESULTS.json').read_text());errors[group]=[];paired[group]={m:sum(x['task_success'] for x in rs if x['method']==m) for m in ['stap','stap_open']}
 for r in rs:
  if r['method']!='stap':continue
  d=json.load(gzip.open(r['path'],'rt'));pred=np.array(d['selector_logs'][0]['predicted_states'][1]);actual=np.array(d['skill_results'][0]['measurement']['tool'][:3]);errors[group].append(float(np.linalg.norm(pred[14:17]-actual)*1000))
fig,ax=plt.subplots(figsize=(6,4));ax.boxplot(list(errors.values()),tick_labels=['Absolute state','State increment'],showmeans=True);ax.set(ylabel='Measured tool position prediction error (mm)',title='Common dynamics repair; same64 episodes / budget');fig.tight_layout();fig.savefig(a.out/'DYNAMICS_ERROR.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4));x=np.arange(2);ax.bar(x-.18,[paired[g]['stap_open'] for g in paired],.36,label='Open-loop sequence');ax.bar(x+.18,[paired[g]['stap'] for g in paired],.36,label='Measured-state replanning');ax.set_xticks(x,['Absolute dynamics','Residual dynamics']);ax.set(ylim=(0,8),ylabel='Full-task success / 8',title='Same first parameter and physical prefix (exact audit)');ax.legend();fig.tight_layout();fig.savefig(a.out/'POSTERIOR_ABLATION.png',dpi=300);plt.close(fig)
(a.out/'MODEL_DIAGNOSTICS.json').write_text(json.dumps(dict(position_errors_mm=errors,paired=paired),indent=2))
for name in ['video_rescue','video_replan_regression']:
 out=a.out.parent/name
 if not (out/'DATA.json').exists():continue
 data=json.loads((out/'DATA.json').read_text());ss=data['series'];keys=list(ss[0]['signals']);fig,axs=plt.subplots(len(keys),1,figsize=(9,11),sharex=True)
 for ax,key in zip(axs,keys):
  for s,c in zip(ss,['#b37021','#187ba0']):
   ax.plot(np.arange(101)/20,s['signals'][key],label=s['label'],color=c)
   for d in s['decisions']:ax.axvline(d['step']/20,color=c,ls=':',alpha=.6)
  ax.set_ylabel(key,fontsize=9);ax.grid(alpha=.15)
 axs[0].legend(fontsize=8);axs[-1].set_xlabel('Physical simulation time (s); frame k = control step k');fig.suptitle('DATA_DERIVED_PLOT | actual saved states and contacts');fig.tight_layout();fig.savefig(out/'TRACE.png',dpi=300);fig.savefig(out/'TRACE.svg');plt.close(fig)
