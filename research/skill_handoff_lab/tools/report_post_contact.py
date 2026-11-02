"""Raw-linked physical mechanism figures; no new physics or success inference."""
import argparse,json,gzip,collections,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'figures';out.mkdir(exist_ok=True);rows=json.loads((r/'design/RESULTS.json').read_text());groups=collections.defaultdict(list)
for z in rows:groups[(z['seed'],z['grasp'])].append(z)
keys=sorted(groups);actions=[[0,0],[-1,0],[1,0],[0,-1],[0,1]];dist=np.array([[next(z['final_distance']for z in groups[k]if z['action']==u)for u in actions]for k in keys]);success=np.array([[next(z['success']for z in groups[k]if z['action']==u)for u in actions]for k in keys]);labels=['Continue','Lateral −15 mm','Lateral +15 mm','Yaw −0.15 rad','Yaw +0.15 rad']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
eligible=np.array([groups[k][0]['contact_step']is not None for k in keys]);shown=dist*1000;shown[~eligible]=np.nan;cmap=plt.get_cmap('RdYlGn_r').copy();cmap.set_bad('#dddddd');fig,ax=plt.subplots(figsize=(10,9),layout='constrained');im=ax.imshow(shown,cmap=cmap,vmin=400,vmax=900,aspect='auto');ax.set_xticks(range(5),labels,rotation=18,ha='right');ax.set_yticks(range(len(keys)),[f'{k[0]} / grasp {k[1]*1000:.0f} mm'for k in keys]);ax.set_title('Declared intervention matrix | full native100\n✓ terminal success; gray: correction event never reached',loc='left')
for i in range(len(keys)):
 for j in range(5):ax.text(j,i,(f'{dist[i,j]*1000:.0f}'+(' ✓'if success[i,j]else ''))if eligible[i]else ('No event'if j==2 else '—'),ha='center',va='center',fontsize=9,color='black')
fig.colorbar(im,ax=ax,label='Actual cube/base distance (mm), native threshold 600');fig.savefig(out/'DESIGN_MATRIX.png',dpi=220);fig.savefig(out/'DESIGN_MATRIX.pdf');plt.close(fig)
rescues=int(np.sum(~success[:,0]&success.any(1)));regressions=int(np.sum(success[:,0]&(~success[:,1:]).any(1)));seeds=sorted({k[0]for k in keys});nominal_best=sum(any(z['success']and z['action']==[0,0]for z in rows if z['seed']==seed)for seed in seeds);joint_oracle=sum(any(z['success']for z in rows if z['seed']==seed)for seed in seeds)
summary=dict(design_episodes=len(rows),scene_clusters=len(seeds),grasp_contexts=len(keys),actual_postcontact_contexts=int(eligible.sum()),native_success_by_action={label:int(success[:,j].sum())for j,label in enumerate(labels)},fixed_prefix_rescued_contexts=rescues,fixed_prefix_regressed_contexts=regressions,nominal_grasp_oracle=nominal_best,joint_grasp_contact_oracle=joint_oracle,warning='Oracle is retrospective action-set coverage, not a deployable method or learned score.')
(r/'analysis/DESIGN_SUMMARY.json').write_text(json.dumps(summary,indent=2))
for name in ['video_rescue','video_regression']:
 folder=r/name
 if not(folder/'DATA.json').exists():continue
 d=json.loads((folder/'DATA.json').read_text());fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
 for ax,key in zip(axes,['Cube/base distance (mm)','Tool/cube contact proxy (N)','Measured TCP/tool rotation change (deg)']):
  for s,col in zip(d['series'],['#bb7722','#168d9a']):
   ax.plot(range(101),s['signals'][key],label=s['label'],color=col,lw=2)
   event=s['contact_event'];ax.axvline(event['step'],color=col,ls=':',alpha=.6)
  ax.set_ylabel(key);ax.grid(alpha=.2)
 axes[0].axhline(600,color='#333',ls='--',label='Native success threshold');axes[0].legend(fontsize=9);axes[-1].set_xlabel('Actual control step / video frame (20 Hz)');fig.savefig(folder/'TRACE.png',dpi=220);fig.savefig(folder/'TRACE.pdf');plt.close(fig)
 values=[s['signals']['Cube/base distance (mm)'][-1]for s in d['series']];event=d['series'][0]['contact_event'];caption=f'''# MEASURED_POSE_REPLAY / DATA_DERIVED_PLOT

OBSERVED: Same initial state and exact physical prefix through step {event['step']}; native100, {100-event['step']} steps remain. Measured terminal cube/base distances {values[0]:.2f} versus {values[1]:.2f} mm; native terminal success {d['series'][0]['success']} versus {d['series'][1]['success']}.

INTERPRETATION: Post-contact intervention can {'recover pulling progress'if name=='video_rescue'else 'consume budget and destroy an otherwise successful continuation'}. The recorded decision boundary is in PULL and uses actual force plus cumulative cube motion; earlier contacts may already have occurred. It is not assumed to imply secure hooking. See EVENT_TIMING.json. Force plot is maximum measured100Hz substep force in each20Hz control interval.

NOT ESTABLISHED: These cases were selected from ordinary RL training failures, not independent evaluation. They establish action effects, not learned-policy superiority or originality. No new physics during rendering; frame k corresponds exactly to saved physical step k, with initial frame0. Box overlays project cube geometry, not measured contact patches.
''';(folder/'CAPTION.md').write_text(caption);m=json.loads((folder/'MANIFEST.json').read_text());m['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in folder.iterdir()if p.is_file()and p.name!='MANIFEST.json'};(folder/'MANIFEST.json').write_text(json.dumps(m,indent=2))
fig,ax=plt.subplots(figsize=(8,4),layout='constrained');paths=['effect_model','effect_model_physical'];vals=[json.loads((r/x/'EFFECT_MODEL.json').read_text())for x in paths];errors=[min(v['leave_scene_out_mse'].values())for v in vals]+[vals[-1]['leave_scene_out_global_mse']];ax.bar(['18D standardized state','6D physical geometry','Constant effect'],errors,color=['#999','#178e9b','#d6923c']);ax.set(ylabel='Leave-scene-out Jacobian squared error (m²)',title='Measured action-effect prediction | 26 contexts / 21 scenes');ax.set_yscale('log')
for i,v in enumerate(errors):ax.text(i,v*1.08,f'{v:.5f}',ha='center')
fig.savefig(out/'EFFECT_PREDICTION.png',dpi=240);fig.savefig(out/'EFFECT_PREDICTION.pdf');plt.close(fig);print(summary)
