"""CPU analysis of substep contact probes; verify unchanged physics first."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from transforms3d.quaternions import quat2mat

def read(path):return json.load(gzip.open(path,'rt'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);results=[];plotdata=[]
 for label,oldlabel in [('g035','g035'),('g070','lookahead')]:
  path=a.root/f'740001_{label}'/'TRAJECTORY.json.gz'
  if not path.exists():continue
  oldpath=a.original/f'740001_{oldlabel}'/'TRAJECTORY.json.gz';d=read(path);old=read(oldpath)
  fields=['action','q','qvel','tcp','tool','cube','tcp_T_tool'];delta={k:max(float(np.max(np.abs(np.array(l[k])-r[k]))) for l,r in zip(d['steps'],old['steps'])) for k in fields}
  exact=all(x==0 for x in delta.values()) and len(d['steps'])==len(old['steps']) and d['summary']['initial_sha256']==old['summary']['initial_sha256']
  substeps=[x for s in d['steps'] for x in s['physics_substep_contacts']];phases={};contacts=[]
  for x in substeps:
   phase=phases.setdefault(x['phase'],dict(samples=0,tool_peak_N=0.,robot_peak_N=0.,robot_peak_link=None,tool_impulse_norm_sum_Ns=0.,robot_impulse_norm_sum_Ns=0.))
   f=x['force_norms_N'];robots={k:v for k,v in f.items() if k!='held_tool'};link=max(robots,key=robots.get);peak=robots[link];phase['samples']+=1;phase['tool_peak_N']=max(phase['tool_peak_N'],f['held_tool']);phase['tool_impulse_norm_sum_Ns']+=f['held_tool']*x['physics_dt_s'];phase['robot_impulse_norm_sum_Ns']+=sum(robots.values())*x['physics_dt_s']
   if peak>phase['robot_peak_N']:phase.update(robot_peak_N=peak,robot_peak_link=link)
   if max(peak,f['held_tool'])>.5:contacts.append(dict(control_step=x['control_step'],substep=x['substep'],phase=x['phase'],tool_force_N=f['held_tool'],robot_max_force_N=peak,robot_link=link if peak>0 else None))
  rows=[]
  for s in d['steps']:
   sub=s['physics_substep_contacts'];rows.append(dict(step=s['step'],phase=s['phase'],tool_peak=max(x['force_norms_N']['held_tool'] for x in sub),robot_peak=max(v for x in sub for k,v in x['force_norms_N'].items() if k!='held_tool'),tool_end=s['tool_cube_force_N'],cube_base_distance=s['cube_base_distance_m']))
  attachment_reference=next(x for x in reversed(d['steps']) if x['phase']=='LIFT')['tcp_T_tool'];R0=quat2mat(attachment_reference[3:]);t0=np.array(attachment_reference[:3]);drift={}
  for phase in ['LIFT','TRANSFER','HOOK','PULL']:
   states=[x for x in d['steps'] if x['phase']==phase]
   if not states:continue
   drift[phase]=dict(rotation_max_deg=max(float(np.arccos(np.clip((np.trace(R0.T@quat2mat(x['tcp_T_tool'][3:]))-1)/2,-1,1))*180/np.pi) for x in states),translation_max_mm=max(float(np.linalg.norm(np.array(x['tcp_T_tool'][:3])-t0)*1000) for x in states))
  result=dict(attachment_reference='actual endLIFT TCP-relative tool pose',attachment_drift=drift,label=label,success=d['summary']['success'],initial_match=d['summary']['initial_sha256']==old['summary']['initial_sha256'],exact_recorded_action_state_match=exact,max_abs_differences=delta,control_steps=len(d['steps']),physical_substeps=len(substeps),all_frames_have5_substeps=all(len(s['physics_substep_contacts'])==5 for s in d['steps']),phases=phases,contact_events_over_half_N=contacts,trace=str(path),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original=str(oldpath),original_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),scope='Read-only contact instrumentation. Impulse norm sums are contact exposure, not net impulse or causal contribution. Attribution requires command/state match and temporal evidence.')
  results.append(result);plotdata.append((label,rows))
 (a.out/'CONTACT_AUDIT.json').write_text(json.dumps(results,indent=2))
 if not results:return
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(len(plotdata),2,figsize=(12,3.5*len(plotdata)),squeeze=False)
 for i,(label,rows) in enumerate(plotdata):
  t=[r['step']/20 for r in rows];ax=axes[i,0];ax.plot(t,[r['tool_peak'] for r in rows],label='Tool:100Hz within-frame peak');ax.plot(t,[r['robot_peak'] for r in rows],label='Any robot link:100Hz peak');ax.plot(t,[r['tool_end'] for r in rows],label='Tool:end-frame20Hz',linestyle='--');ax.set_ylabel('Pair force norm (N)');ax.set_xlabel('Native simulation time (s)');ax.set_title(label+' | actual PhysX contact queries');ax.legend(fontsize=8)
  ax=axes[i,1];ax.plot(t,[r['cube_base_distance'] for r in rows]);ax.axhline(.6,color='gray',linestyle='--');ax.set_xlabel('Native simulation time (s)');ax.set_ylabel('Cube/base XY distance(m)');ax.set_title(label+' | native goal geometry')
 fig.tight_layout();fig.savefig(a.out/'CONTACT_MECHANISM.png',dpi=180);fig.savefig(a.out/'CONTACT_MECHANISM.svg');plt.close(fig)
 print(json.dumps([{k:r[k] for k in ['label','success','exact_recorded_action_state_match','control_steps','physical_substeps','phases']} for r in results],indent=2))
if __name__=='__main__':main()
