"""Exact-prefix and raw-path checks plus real100Hz transfer dynamics."""
import argparse,gzip,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from transforms3d.quaternions import quat2mat

def read(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def compare(x,y,n=None):
    n=n if n is not None else min(len(x['steps']),len(y['steps']));keys=['action','q','qvel','tcp','tool','cube','tcp_T_tool']
    delta={k:max([float(np.max(np.abs(np.array(a[k])-np.array(b[k])))) for a,b in zip(x['steps'][:n],y['steps'][:n])],default=0.) for k in keys}
    return dict(initial_exact=x['initial']==y['initial'],steps=n,max_abs_difference=delta,exact=all(v==0 for v in delta.values()))
def describe(d,limits):
    ref=next(x for x in reversed(d['steps']) if x['phase']=='LIFT')['tcp_T_tool'];R=quat2mat(ref[3:]);angle=lambda s:float(np.degrees(np.arccos(np.clip((np.trace(R.T@quat2mat(s['tcp_T_tool'][3:]))-1)/2,-1,1))))
    records=[x for x in d['steps'] if x['phase']=='TRANSFER'];subs=[x for s in records for x in s['physics_substep_contacts']];allsubs=[x for s in d['steps'] for x in s['physics_substep_contacts']];events=[x for x in allsubs if x['force_norms_N']['held_tool']>1e-8];first=events[0] if events else None;before=d['steps'][first['control_step']-2] if first and first['control_step']>1 else None
    qv=np.array([x['robot_qvel'][:7] for x in subs]);qa=np.array([x['robot_qacc'][:7] for x in subs]);changes=[float(np.max(abs(np.array(s['physics_substep_contacts'][0]['robot_qvel'])-s['physics_substep_contacts'][-1]['robot_qvel']))) for s in records]
    return dict(success=d['summary']['success'],error=d['summary']['error'],control_steps=len(d['steps']),physical_substeps=len(allsubs),transfer_steps=len(records),transfer_substeps=len(subs),transfer_max_attachment_rotation_deg=max(map(angle,records)),transfer_100Hz_peak_velocity_rad_s=float(abs(qv).max()),transfer_100Hz_max_URDF_velocity_ratio=float(np.max(abs(qv)/limits)),transfer_100Hz_peak_qacc_rad_s2=float(abs(qa).max()),transfer_100Hz_samples_above_nominal_point9_velocity=int(np.any(abs(qv)>.9,axis=1).sum()),transfer_100Hz_samples_above_nominal_point9_acceleration=int(np.any(abs(qa)>.9,axis=1).sum()),transfer_max_within_control_qvel_change=float(max(changes)),first_nonzero_tool_cube_contact=None if first is None else dict(step=first['control_step'],substep=first['substep'],phase=first['phase'],force_N=first['force_norms_N']['held_tool']),attachment_before_first_contact_deg=angle(before) if before else None,transfer_tool_peak_N=max(x['force_norms_N']['held_tool'] for x in subs),final_cube_distance_m=d['steps'][-1]['cube_base_distance_m'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--urdf',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);xml=ET.parse(a.urdf).getroot();limits=np.array([float(xml.find(f"joint[@name='panda_joint{i}']/limit").get('velocity')) for i in range(1,8)]);rows=[]
    for grasp,oldlabel in [('035','g035'),('070','lookahead')]:
        paths={s:a.root/f'740001_g{grasp}_transfer_stride{s}'/'TRAJECTORY.json.gz' for s in [1,2]}
        if not all(p.exists() for p in paths.values()):continue
        d={s:read(p) for s,p in paths.items()};old=read(a.original/f'740001_{oldlabel}'/'TRAJECTORY.json.gz');prefix=d[2]['decisions'][1]['boundary_step'];raw={s:next(p['position'] for p in data['plans'] if p['phase']=='TRANSFER') for s,data in d.items()};raw_exact=raw[1]==raw[2]
        row=dict(grasp=grasp,paths={str(s):str(p) for s,p in paths.items()},sha256={str(s):hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()},fresh_control_vs_historical=compare(d[2],old),slowdown_prefix_vs_control=compare(d[1],d[2],prefix),second_boundary_steps={str(s):x['decisions'][1]['boundary_step'] for s,x in d.items()},same_raw_transfer_path=raw_exact,raw_transfer_sha256={str(s):hashlib.sha256(json.dumps(q).encode()).hexdigest() for s,q in raw.items()},control=describe(d[2],limits),slowdown=describe(d[1],limits))
        historical_contact=a.original.parent/'multiskill_contact_probe'/f'740001_g{grasp}'/'TRAJECTORY.json.gz'
        if historical_contact.exists():
            history=read(historical_contact);x=[v for st in history['steps'] for v in st['physics_substep_contacts']];y=[v for st in d[2]['steps'] for v in st['physics_substep_contacts']]
            delta=max(abs(old['force_norms_N'][k]-new['force_norms_N'][k]) for old,new in zip(x,y) for k in old['force_norms_N'])
            row['fetch_force_audit']=dict(samples=len(x),force_norm_max_difference=delta,exact=len(x)==len(y) and delta==0)
        rows.append(row)
    report=dict(episodes=len(rows)*2,physical_control_steps=sum(r[k]['control_steps'] for r in rows for k in ['control','slowdown']),physical_substeps=sum(r[k]['physical_substeps'] for r in rows for k in ['control','slowdown']),optimizer_updates=0,resource_context='SHARED_GPU_CONTENTION',dynamics_contract='Separate actual100Hz URDF velocity ratio from uncertified acceleration/full TOPP contract; no hardware validation',rows=rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for row in rows:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for stride,label,color in [(2,'Original stride2','#b85c43'),(1,'Slower stride1','#326eaa')]:
            data=read(Path(row['paths'][str(stride)]));ref=next(x for x in reversed(data['steps']) if x['phase']=='LIFT')['tcp_T_tool'];R=quat2mat(ref[3:]);states=[x for x in data['steps'] if x['phase']=='TRANSFER'];raw=next(x['position'] for x in data['plans'] if x['phase']=='TRANSFER');idx=list(range(0,len(raw),stride))
            if idx[-1]!=len(raw)-1:idx.append(len(raw)-1)
            progress=np.array(idx[:len(states)])/max(1,len(raw)-1)
            angles=[float(np.degrees(np.arccos(np.clip((np.trace(R.T@quat2mat(x['tcp_T_tool'][3:]))-1)/2,-1,1)))) for x in states]
            vel=[max(float(np.max(abs(np.array(z['robot_qvel'][:7])))) for z in x['physics_substep_contacts']) for x in states]
            axes[0].plot(progress,angles,label=label,color=color);axes[1].plot(progress,vel,label=label,color=color)
        axes[0].set_ylabel('Actual attachment rotation from endLIFT (deg)');axes[1].set_ylabel('Actual100Hz within-frame peak joint speed (rad/s)')
        for ax in axes:ax.set_xlabel('Commanded native path sample progress');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
        fig.suptitle('Same physical prefix and exact joint path | grasp '+row['grasp']+'mm');fig.tight_layout();fig.savefig(a.out/('SLOWDOWN_'+row['grasp']+'.png'),dpi=180);fig.savefig(a.out/('SLOWDOWN_'+row['grasp']+'.svg'));plt.close(fig)
    (a.out/'SLOWDOWN_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
