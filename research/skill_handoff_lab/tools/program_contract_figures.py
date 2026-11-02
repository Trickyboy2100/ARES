"""Recompute scientific figures from retained physical evidence (no simulated data)."""
import argparse,json,gzip,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
 def rows(name):return json.loads((a.root/name/'RESULTS.json').read_text())
 def save(fig,name):
  fig.tight_layout();fig.savefig(a.out/(name+'.png'),bbox_inches='tight');fig.savefig(a.out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
 modes=['corrected','corrected_recompute','corrected_expand','corrected_feedback'];rr=rows('feedback')+rows('factor');counts=[sum(r['success']for r in rr if r['mode']==m)for m in modes]
 fig,ax=plt.subplots(figsize=(8,4));bars=ax.bar(range(4),counts,color=['#777777','#777777','#007f85','#007f85']);ax.set_xticks(range(4),['29 cm cap\nno update','29 cm cap\nupdate','36 cm cap\nno update','36 cm cap\nupdate']);ax.set_ylim(0,8.8);ax.set_ylabel('Native successes\n(8 matched conditions)');ax.set_title('Stroke authority explains the rescue; feedback is not necessary')
 for b,n in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,n+.12,f'{n}/8',ha='center')
 save(fig,'01_stroke_factorial')
 rr=rows('evaluation512');methods=list(dict.fromkeys(r['method']for r in rr));fig,ax=plt.subplots(figsize=(10,4));v=[sum(r['success']for r in rr if r['method']==m)for m in methods];ax.bar(range(len(v)),v,color=['#007f85']*2+['#c17b25']*2+['#647a99']*4);ax.set_xticks(range(len(v)),methods,rotation=25,ha='right');ax.set_ylim(0,8.8);ax.set_ylabel('Native successes\n(8 new development scenes)');ax.set_title('Same actuator, observation and horizon: no independent method gain')
 for i,n in enumerate(v):ax.text(i,n+.12,f'{n}/8',ha='center')
 save(fig,'02_matched_baselines')
 fig,axs=plt.subplots(1,2,figsize=(10,4));data=rows('sac_recovered/episodes');window=32;y=[np.mean([r['success']for r in data[max(0,i-window+1):i+1]])for i in range(len(data))];axs[0].plot(np.arange(len(y))+1,y,color='#c17b25');axs[0].set(xlabel='Completed online physics episodes',ylabel='Rolling native success (window 32)',ylim=(0,1),title='Ordinary two-boundary SAC: 1 training seed');axs[0].axvline(128,color='#999',ls='--');axs[0].axvline(256,color='#999',ls='--');tt=[json.loads((a.root/'sac_recovered'/f'TRAINING_{k}.json').read_text())for k in [256,512]];axs[1].plot([t['decisions']for t in tt],[t['optimizer_updates']for t in tt],'o-');axs[1].set(xlabel='Completed RL decisions',ylabel='Measured optimizer updates',title='Checkpoints retained; convergence not established');save(fig,'03_training_and_cost')
 events=json.loads((a.root/'analysis/UNIFIED_EVENTS.json').read_text());fig,ax=plt.subplots(figsize=(8,4));e=[r for r in events if r['group']=='pilot' and r['grasp']==.035 and r['mode']=='corrected'];
 for mass,col in [(.75,'#007f85'),(1.25,'#c17b25')]:
  ss=[r for r in e if r['mass']==mass and r['precontact_rotation_deg']is not None];ax.scatter([str(r['seed'])for r in ss],[r['precontact_rotation_deg']for r in ss],s=90,label=f'Tool density scale {mass}',color=col)
 ax.set(ylabel='TCP-relative tool drift before first contact (deg)',xlabel='Matched diagnosis seed',title='Stable pick does not imply stable transport');ax.legend();save(fig,'04_precontact_drift')
 # Same measured-prefix comparison, no endpoint interpolation or new physics.
 files=[a.root/'feedback/745000_m0.75_g0.035_corrected.json.gz',a.root/'factor/745000_m0.75_g0.035_corrected_expand.json.gz'];ds=[json.load(gzip.open(f,'rt'))for f in files];end=max(i+1 for i,z in enumerate(ds[0]['steps'])if z['phase']=='HOOK');assert ds[0]['steps'][:end]==ds[1]['steps'][:end]
 fig,axs=plt.subplots(2,1,figsize=(8,6),sharex=True)
 for d,label,col in zip(ds,['290 mm capped stroke','321 mm geometric stroke'],['#c17b25','#007f85']):
  axs[0].plot(np.arange(1,101)/20,[z['cube_base_distance_m']*1000 for z in d['steps']],label=label,color=col);axs[1].plot(np.arange(1,101)/20,[z['tool_cube_force_N']for z in d['steps']],color=col)
 axs[0].axhline(600,ls='--',color='black',label='Native success boundary');axs[0].legend();axs[0].set(ylabel='Cube/base XY distance (mm)',title='Identical physical prefix through HOOK; different stroke length');axs[1].set(xlabel='Physics time (s)',ylabel='Tool/cube force proxy (N)')
 for ax in axs:ax.axvline(end/20,ls=':',color='#999')
 save(fig,'05_same_prefix_mechanism');(a.out/'FIGURE_PROVENANCE.json').write_text(json.dumps(dict(source_root=str(a.root),same_prefix_steps=end,source_files=[str(f.relative_to(a.root))for f in files],no_synthetic_values=True),indent=2))
 if not e:raise ValueError('No drift cases selected: fix group selector')
if __name__=='__main__':main()
