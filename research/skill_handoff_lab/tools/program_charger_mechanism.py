"""Source-backed native PlugCharger confirmation and contact-mode diagnostics."""
import argparse, gzip, hashlib, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(root, variant, seed):
 p=next((root/f'{variant}_{seed}').glob('*.json.gz'))
 return p,json.load(gzip.open(p,'rt'))
def rotation(q):
 w,x,y,z=np.array(q)/np.linalg.norm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def arrays(t):
 s=t['steps'];local=np.array([(rotation(z['cube'][3:])@np.diag([-1,-1,1])).T@(np.array(z['tool'][:3])-z['cube'][:3]) for z in s])
 return dict(x=local[:,0]*1000,lateral=np.linalg.norm(local[:,1:],axis=1)*1000,angle=np.array([z['goal_angle_error_rad'] for z in s]),force=np.array([z['tool_cube_force_N'] for z in s]),mode=np.array([int(z['servo']['advance']) if 'servo' in z else (2 if z['phase']=='TERMINAL_HOLD' else np.nan) for z in s]))
def mechanism(t):
 a=arrays(t);ix=[i for i,z in enumerate(t['steps']) if z['phase']=='CONTACT_SERVO']; modes=[z['servo']['advance'] for z in t['steps'] if 'servo' in z];dx=np.diff(a['x'][ix]) if ix else np.array([])
 return dict(servo_steps=len(ix),mode_switches=sum(a!=b for a,b in zip(modes,modes[1:])),backward_travel_mm=float(-dx[dx<0].sum()),backward_steps_over_0_5mm=int((dx<-.5).sum()),net_axial_progress_mm=float(a['x'][ix[-1]]-a['x'][ix[0]]) if ix else None,peak_contact_proxy_N=float(max(a['force'])),first_native_success_step=next((z['step'] for z in t['steps'] if z['native_success']),None))
def boundary(t):
 return next((z for z in reversed(t['steps']) if z['phase']=='ALIGN'),None)
def delta(a,b):
 if a is None or b is None:return None
 keys=['q','qvel','tcp','tool','cube','tool_linear_velocity','tool_angular_velocity','cube_linear_velocity','action','tool_cube_force_N']
 return max(float(np.max(np.abs(np.array(a[k])-np.array(b[k])))) for k in keys)
def pair(rows,a,b):
 pairs=[(next(r for r in rows if r['seed']==s and r['variant']==a),next(r for r in rows if r['seed']==s and r['variant']==b)) for s in range(726000,726008)]
 rescue=sum(not x['success'] and y['success'] for x,y in pairs);regress=sum(x['success'] and not y['success'] for x,y in pairs);n=rescue+regress
 return dict(rescues=rescue,regressions=regress,both_success=sum(x['success'] and y['success'] for x,y in pairs),both_fail=sum(not x['success'] and not y['success'] for x,y in pairs),exact_two_sided_p=min(1,2*sum(math.comb(n,k) for k in range(min(rescue,regress)+1))/2**n) if n else 1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[]; sources=[];checks=[];variants=['frozen_stop','servo_stop','servo_local_hold_stop']
 for seed in range(726000,726008):
  traces=[]
  for variant in variants:
   path,t=load(a.root,variant,seed);traces.append(t);s=t['summary'];z=t['steps'][-1]
   assert len(t['steps'])==200 and s['success']==bool(z['goal_position_error_m']<=.005 and z['goal_angle_error_rad']<=.2)
   assert all(np.all(np.isfinite(z['q']+z['tool'])) for z in t['steps'])
   phase_error=s['error'] or (('CONTACT_WITHOUT_GOAL' if any(z['tool_cube_force_N']>0 for z in t['steps']) else 'NO_TOOL_SOCKET_CONTACT') if not s['success'] else 'SUCCESS')
   rows.append(dict(seed=seed,variant=variant,success=s['success'],error=phase_error,position_mm=z['goal_position_error_m']*1000,angle_rad=z['goal_angle_error_rad'],wall_s=s['wall_s'],planning_wall_s=s['planning_wall_s'],plan_calls=s['plan_calls'],rrt_calls=s['rrt_calls'],**mechanism(t)))
   sources.append(dict(path=str(path),sha256=sha(path),manifest=json.loads((path.parent/'manifest.json').read_text())))
  assert len({t['summary']['initial_sha256'] for t in traces})==1
  checks.append(dict(seed=seed,old_vs_local_align_delta=delta(boundary(traces[1]),boundary(traces[2])),integral_vs_local_align_delta=delta(boundary(traces[0]),boundary(traces[2]))))
 summary=dict(episodes=24,scenes=8,physics_steps=4800,optimizer_steps=0,success={v:sum(r['success'] for r in rows if r['variant']==v) for v in variants},paired={f'{v}_to_local':pair(rows,v,variants[-1]) for v in variants[:2]},boundary_checks=checks,rows=rows,sources=sources)
 (a.out/'CONFIRMATION.json').write_text(json.dumps(summary,indent=2))
 for seed in [725002,725005]:
  out=a.out/str(seed);out.mkdir(exist_ok=True);ts=[load(a.root,v,seed) for v in variants[1:]];fig,axes=plt.subplots(5,1,figsize=(9,9),sharex=True,layout='constrained')
  colors=['#c46220','#146b9a'];labels=['Original servo','Depth-preserving servo']
  for k,ax in zip(['x','lateral','angle','force','mode'],axes):
   for (path,t),color,label in zip(ts,colors,labels):ax.plot(np.arange(80,201),arrays(t)[k][79:],color=color,label=label,lw=1.5)
   ax.set_ylabel(dict(x='Axial position (mm)',lateral='Lateral error (mm)',angle='Orientation (rad)',force='Contact proxy (N)',mode='Control mode')[k]);ax.grid(alpha=.2)
  axes[0].legend();axes[1].axhline(.4,color='gray',ls=':',lw=1);axes[2].axhline(.03,color='gray',ls=':',lw=1);axes[-1].set_yticks([0,1,2],['Align','Advance','Hold']);axes[-1].set_xlabel('Physical control step (20 Hz; native horizon 200)');axes[-1].set_xlim(80,200)
  fig.suptitle(f'Seed {seed} | DATA_DERIVED_PLOT | NO NEW RL CLAIM')
  fig.savefig(out/'TRACE.png',dpi=300);fig.savefig(out/'TRACE.svg');plt.close(fig)
  evidence=[]
  for path,t in ts:evidence.append(dict(path=str(path),sha256=sha(path),success=t['summary']['success'],mechanism=mechanism(t)))
  (out/'mechanism.json').write_text(json.dumps(evidence,indent=2))
 diagnostics=[]
 for path in sorted(a.root.glob('grasp_diagnostic_726*/RESULTS.json')):
  r=json.loads(path.read_text())[0];trajectory=path.parent/r['trajectory'];t=json.load(gzip.open(trajectory,'rt'))
  assert len(t['steps'])==200
  diagnostics.append(dict(**r,source=str(trajectory),manifest=json.loads((path.parent/'manifest.json').read_text()),mechanism=mechanism(t)))
 (a.out/'GRASP_DIAGNOSTIC.json').write_text(json.dumps(diagnostics,indent=2))
 fig,ax=plt.subplots(figsize=(7,4),layout='constrained');counts=[summary['success'][v] for v in variants];bars=ax.bar(['Integral + hold','Original servo','Depth-preserving'],counts,color=['#888888','#c46220','#146b9a'])
 for bar,n in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,n+.1,f'{n}/8',ha='center')
 ax.set_ylim(0,8.8);ax.set_ylabel('Native final successes / 8 new scenes');ax.set_title('Independent cold-process confirmation\nDATA_DERIVED_PLOT | NO NEW RL CLAIM');ax.spines[['top','right']].set_visible(False);fig.savefig(a.out/'CONFIRMATION.png',dpi=300);plt.close(fig)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','sources']},indent=2))
if __name__=='__main__':main()
