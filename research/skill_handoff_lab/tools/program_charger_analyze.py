"""Summarize EXP-001 physical repeats without treating repeats as independent scenes."""
import argparse,collections,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def state(z):
 return np.r_[z['q'],z['qvel'],z['tcp'],z['tool'],z['cube'],z['tool_linear_velocity'],z['tool_angular_velocity'],z['cube_linear_velocity'],z['action'],z['tool_cube_force_N']]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);data={};rows=[];groups={};raw=[]
 for path in sorted(a.root.glob('*/RESULTS.json')):
  group=path.parent.name;rr=json.loads(path.read_text());groups[group]=dict(episodes=len(rr),scenes=len({r['seed'] for r in rr}),success=sum(r['success'] for r in rr),steps=sum(r['steps'] for r in rr),plans=sum(r['plan_calls'] for r in rr),rrt=sum(r['rrt_calls'] for r in rr),wall_s=sum(r['wall_s'] for r in rr),plan_s=sum(r['planning_wall_s'] for r in rr),failures=collections.Counter())
  for r in rr:
   source=path.parent/r['trajectory'];assert hashlib.sha256(source.read_bytes()).hexdigest()==r['sha256'];raw.append(dict(path=str(source),sha256=r['sha256']))
   with gzip.open(source,'rt') as f:t=json.load(f)
   ss=t['steps'];assert len(ss)==200;end=ss[-1];assert bool(end['goal_position_error_m']<=.005 and end['goal_angle_error_rad']<=.2)==r['success'];key=(r['seed'],r['angle'],r['standoff']);data[group,key]=t
   error=r['error'] or '';force=max(z['tool_cube_force_N'] for z in ss)
   failure='SUCCESS' if r['success'] else ('GRASP_CHECK' if 'GRASP' in error or 'LOST_ON_LIFT' in error else 'PLAN_OR_IK' if 'PLAN' in error or 'IK' in error else 'BUDGET' if 'HORIZON' in error else 'NO_TOOL_SOCKET_CONTACT' if force<.1 else 'CONTACT_WITHOUT_NATIVE_GOAL')
   groups[group]['failures'][failure]+=1;lift=[z for z in ss if z['phase']=='LIFT'];contact=[z for z in ss if z['tool_cube_force_N']>.1];row=dict(group=group,seed=r['seed'],angle=r['angle'],standoff=r['standoff'],success=int(r['success']),failure=failure,error=r['error'],verified_lift=bool(lift and lift[-1]['grasped']),first_contact_step=contact[0]['step'] if contact else None,peak_contact_N=force,final_position_mm=end['goal_position_error_m']*1000,final_angle_rad=end['goal_angle_error_rad'],final_grasp_signal=end['grasped'],first_success_step=next((z['step'] for z in ss if z['native_success']),None),success_last10=sum(z['native_success'] for z in ss[-10:]),peak_joint_tracking_error_rad=max(max(abs(np.array(z['joint_command_error_rad']))) for z in ss),wall_s=r['wall_s'],plan_calls=r['plan_calls']);rows.append(row)
 checks=[]
 for (group,key),t in data.items():
  if group!='frozen_process0' or ('frozen_process1',key) not in data:continue
  u=data['frozen_process1',key];assert t['summary']['initial_sha256']==u['summary']['initial_sha256'];delta=float(np.max(abs(np.array([state(z) for z in t['steps']])-np.array([state(z) for z in u['steps']]))));checks.append(dict(seed=key[0],angle=key[1],standoff=key[2],max_recorded_state_difference=delta,same_outcome=t['summary']['success']==u['summary']['success'],rrt0=t['summary']['rrt_calls'],rrt1=u['summary']['rrt_calls']))
 with (a.out/'EPISODES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 comparison={}
 for group in groups:
  if group.startswith('frozen'):continue
  pairs=[]
  for (gg,key),t in data.items():
   if gg!=group or ('frozen_process0',key) not in data:continue
   base=data['frozen_process0',key];assert base['summary']['initial_sha256']==t['summary']['initial_sha256'];pairs.append((base['summary']['success'],t['summary']['success']))
  comparison[group]=dict(n=len(pairs),rescues=sum(not x and y for x,y in pairs),regressions=sum(x and not y for x,y in pairs))
 cold=[]
 for (group,key),t in data.items():
  if not group.startswith('servo_stop_'):continue
  other='frozen_stop_'+group[len('servo_stop_'):]
  if (other,key) not in data:continue
  b=data[other,key];assert b['summary']['initial_sha256']==t['summary']['initial_sha256']
  aa=[z for z in b['steps'] if z['phase']=='ALIGN'];bb=[z for z in t['steps'] if z['phase']=='ALIGN']
  boundary=None if not aa or not bb else float(np.max(abs(np.array(state(aa[-1]))-np.array(state(bb[-1])))))
  cold.append(dict(seed=key[0],baseline=b['summary']['success'],servo=t['summary']['success'],same_alignment_boundary=boundary==0,boundary_difference=boundary))
 crashes=[]
 for p in sorted(a.root.glob('*/*_CRASH.json.gz')):
  with gzip.open(p,'rt') as f:t=json.load(f)
  crashes.append(dict(path=str(p),physical_steps=len(t['steps']),error=t['error'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 protocol={}
 for name,lo,hi in [('design',724000,724004),('confirmation',725000,725008)]:
  pairs=[c for c in cold if lo<=c['seed']<hi];protocol[name]=dict(scenes=len(pairs),baseline=sum(c['baseline'] for c in pairs),servo=sum(c['servo'] for c in pairs),rescues=sum(not c['baseline'] and c['servo'] for c in pairs),regressions=sum(c['baseline'] and not c['servo'] for c in pairs),verified_alignment_pairs=sum(c['boundary_difference'] is not None for c in pairs),all_available_alignment_boundaries_identical=all(c['same_alignment_boundary'] for c in pairs if c['boundary_difference'] is not None))
 fig,ax=plt.subplots(figsize=(7,4));xx=np.arange(2);ax.bar(xx-.18,[protocol[k]['baseline'] for k in ['design','confirmation']],.35,label='Frozen integral + hold',color='#94a3b8');ax.bar(xx+.18,[protocol[k]['servo'] for k in ['design','confirmation']],.35,label='Bounded servo + hold',color='#0d9488');ax.set_xticks(xx,['Design: 4 scenes','Frozen confirmation: 8 scenes']);ax.set_ylabel('Native terminal successes');ax.set_ylim(0,9);ax.legend();fig.tight_layout();fig.savefig(a.out/'matched_controls.png',dpi=180);plt.close(fig)
 local_hold=[]
 for (group,key),t in data.items():
  if not group.startswith('servo_local_hold_stop_'):continue
  b=data['servo_stop_'+str(key[0]),key];aa=[z for z in b['steps'] if z['phase']=='ALIGN'];bb=[z for z in t['steps'] if z['phase']=='ALIGN'];delta=float(np.max(abs(np.array(state(aa[-1]))-np.array(state(bb[-1])))));local_hold.append(dict(seed=key[0],previous_servo=b['summary']['success'],local_hold=t['summary']['success'],alignment_boundary_difference=delta))
 output=dict(local_hold_posthoc_pairs=local_hold,protocols=protocol,cold_process_pairs=cold,crashes=crashes,total_including_crashes=sum(g['steps'] for g in groups.values())+sum(c['physical_steps'] for c in crashes),groups=groups,cross_process_repeats=checks,comparison_vs_frozen=comparison,recorded_steps=sum(g['steps'] for g in groups.values()),optimizer_steps=0,raw_root=str(a.root),raw=raw,claim='No independent RL mechanism evaluated')
 (a.out/'METRICS.json').write_text(json.dumps(output,indent=2));fig,ax=plt.subplots(figsize=(8,4));names=[k for k in groups if not k.startswith(('frozen_stop_','servo_stop_'))];ax.bar(range(len(names)),[groups[k]['success']/groups[k]['episodes'] for k in names],color='#277d8e');ax.set_xticks(range(len(names)),names,rotation=20,ha='right');ax.set_ylim(0,1.1);ax.set_ylabel('Native terminal success fraction');ax.set_title('Different diagnostic subsets: inspect counts, not pooled independent trials')
 for i,k in enumerate(names):ax.text(i,groups[k]['success']/groups[k]['episodes']+.03,f"{groups[k]['success']}/{groups[k]['episodes']}",ha='center')
 fig.tight_layout();fig.savefig(a.out/'controls.png',dpi=180);plt.close(fig)
 # Stage-level physical effect on the historically matched contact case.
 fig,axs=plt.subplots(2,1,figsize=(8,6),sharex=True)
 for group in [k for k in groups if k in ['frozen_stop_721000','servo_stop_721000']]:
  key=(721000,15.,.025)
  if (group,key) not in data:continue
  ss=data[group,key]['steps'];axs[0].plot(np.arange(1,201)/20,[z['goal_position_error_m']*1000 for z in ss],label=group);axs[1].plot(np.arange(1,201)/20,[z['tool_cube_force_N'] for z in ss],label=group)
 axs[0].set_ylabel('Actual tool-goal distance [mm]');axs[0].axhline(5,c='black',ls='--');axs[0].legend();axs[1].set_ylabel('Tool-socket contact force [N]');axs[1].set_xlabel('Simulated seconds');fig.tight_layout();fig.savefig(a.out/'contact_effect.png',dpi=180);plt.close(fig);print(json.dumps({k:v for k,v in output.items() if k not in ['raw','cross_process_repeats']},indent=2))
if __name__=='__main__':main()
