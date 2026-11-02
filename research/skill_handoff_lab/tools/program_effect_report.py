"""Recompute paired outcomes, endpoint FK and plots directly from preserved traces."""
import argparse,json,gzip,collections,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
 groups={};raw={};sources=[]
 for folder in a.root.iterdir():
  if not folder.is_dir() or not (folder/'RESULTS.json').exists():continue
  rr=json.loads((folder/'RESULTS.json').read_text());groups[folder.name]=rr
  for r in rr:
   f=Path(r['path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'];raw[str(f)]=json.load(gzip.open(f,'rt'));sources.append(dict(path=str(f),sha256=r['sha256']))
 scores={};steps=0;episodes=0;errors={};prefix=[];endpoints=[]
 for group,rr in groups.items():
  scores[group]={};errors[group]=collections.Counter()
  for r in rr:
   key=r.get('method',r['mode']);s=r['success'];n=len(s) if isinstance(s,list) else 1;su=sum(s) if isinstance(s,list) else int(s);val=scores[group].setdefault(key,dict(n=0,success=0));val['n']+=n;val['success']+=su;episodes+=n;steps+=n*(250 if group.startswith('pyramid') else 100)
   e=r.get('error');errors[group][str(e)]+=n
   if group.startswith('endpoint') and group!='endpoint':
    d=raw[r['path']];plans=[x for x in d['plans'] if x['phase']=='TRANSFER' and 'endpoint_fk'in x];endpoints.append(dict(grasp=r['grasp'],mode=r['mode'],success=r['success'],fk_mm=[x['endpoint_fk']['position_error_mm'] for x in plans],transfer_steps=sum(s['phase']=='TRANSFER' for s in d['steps']),finish_step=max(s['step'] for s in d['steps'] if s['phase']!='TERMINAL_HOLD')))
 # Instrumentation invariance: compare complete actions/measurements except new read-only force fields.
 for r in groups.get('tool_all_contacts',[]):
  if r['mode']=='settle9':continue
  group='tool_effect' if r['mode']=='baseline' else 'tool_controls';other=next(x for x in groups[group] if (x['seed'],x['mass'],x['grasp'],x['mode'])==(r['seed'],r['mass'],r['grasp'],r['mode']));d=raw[r['path']];b=raw[other['path']]
  cleaned=json.loads(json.dumps(d['steps']))
  for s in cleaned:
   for z in s['physics_substep_contacts']:z.pop('additional_forces_N',None)
  prefix.append(dict(grasp=r['grasp'],mode=r['mode'],all100_steps_identical=cleaned==b['steps']))
 paired={};rr=groups['evaluation'];fixed={(r['seed'],r['mass']):r for r in rr if r['method']=='fixed'}
 for method in sorted({r['method'] for r in rr}):
  rs=[r for r in rr if r['method']==method];paired[method]=dict(rescue=sum(r['success'] and not fixed[r['seed'],r['mass']]['success'] for r in rs),regression=sum(not r['success'] and fixed[r['seed'],r['mass']]['success'] for r in rs),scene_clusters=2)
 summary=dict(scores=scores,errors=errors,total_executed_episodes=episodes,total_control_steps=steps,invalid_logger_episodes=len(groups.get('endpoint',[])),valid_episodes=episodes-len(groups.get('endpoint',[])),instrumentation_checks=prefix,endpoints=endpoints,paired_development_vs_fixed=paired,sources=sources)
 (a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['sources','errors']},indent=2))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':250})
 fig,ax=plt.subplots(figsize=(9,4.8));names=list(scores['evaluation']);vals=[scores['evaluation'][n]['success'] for n in names];ax.bar(range(len(names)),vals,color=['#537ba0' if n=='fixed' else '#4ba2a0' for n in names]);ax.set_xticks(range(len(names)),[n.replace('_','\n') for n in names],fontsize=9);ax.set_ylim(0,4.5);ax.set_ylabel('Native successes / 4 paired conditions');ax.set_title('Fresh physical execution: learned methods do not beat fixed\n2 scene clusters × 2 loads; one training seed; development only');
 for i,v in enumerate(vals):ax.text(i,v+.1,str(v),ha='center')
 fig.tight_layout();fig.savefig(a.out/'DEVELOPMENT.png');fig.savefig(a.out/'DEVELOPMENT.pdf');plt.close(fig)
 modes=['baseline','frozen12','settle9','dense02','dense005'];fig,ax=plt.subplots(figsize=(8.5,4.5))
 for g,col in [(.035,'#cf8a3a'),(.07,'#25898e')]:
  values=[next(x['fk_mm'][-1] for x in endpoints if x['grasp']==g and x['mode']==m) for m in modes];ax.plot(modes,values,'o-',label=f'Grasp {g*1000:.0f} mm',color=col)
 ax.set_ylabel('Planned endpoint FK error (mm)');ax.set_title('Successful planning status does not certify endpoint accuracy\nSame target, physical prefix, native100 horizon and dynamic limits');ax.legend();fig.tight_layout();fig.savefig(a.out/'ENDPOINT.png');fig.savefig(a.out/'ENDPOINT.pdf');plt.close(fig)
 fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True)
 from scipy.spatial.transform import Rotation
 for mode,col in [('baseline','#cf8a3a'),('frozen12','#25898e'),('settle9','#9259a3')]:
  r=next(x for x in groups['tool_all_contacts'] if x['grasp']==.035 and x['mode']==mode);d=raw[r['path']];s=d['steps'];b=d['skill_results'][0]['boundary_step'];rot=Rotation.from_quat(np.asarray([x['tcp_T_tool'][3:] for x in s])[:,[1,2,3,0]]);ang=(rot[b-1].inv()*rot).magnitude()*180/np.pi
  axs[0].plot(range(b,101),ang[b-1:],label=f'{mode} ({"pass" if r["success"] else "fail"})',color=col)
  sub=[z for x in s for z in x['physics_substep_contacts']];t=[z['control_step']-.8+.2*i for x in s for i,z in enumerate(x['physics_substep_contacts'])];axs[1].plot(t,[z['force_norms_N']['held_tool'] for z in sub],color=col);axs[2].plot(range(1,101),[x['cube_base_distance_m']*1000 for x in s],color=col)
 axs[0].set_ylabel('TCP/tool rotation change (deg)');axs[0].legend();axs[0].set_title('745000 / load0.75 / grasp35mm: strict shared physical prefix\nPassive relative-palm stability; contact force sampled at100Hz')
 axs[1].set_ylabel('Tool/cube contact force (N)');axs[2].set_ylabel('Cube/base XY distance (mm)');axs[2].axhline(600,color='gray',ls='--');axs[2].set_xlabel('Physical control step (20Hz; full native100)');fig.tight_layout();fig.savefig(a.out/'MECHANISM.png');fig.savefig(a.out/'MECHANISM.pdf');plt.close(fig)
if __name__=='__main__':main()
