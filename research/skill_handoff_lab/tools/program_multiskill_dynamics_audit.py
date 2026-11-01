"""CPU audit of saved Pull commands/physical velocities; no claimed certification."""
import argparse,gzip,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from transforms3d.quaternions import quat2mat

def read(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pull-root',type=Path,required=True);ap.add_argument('--contact-root',type=Path,required=True);ap.add_argument('--urdf',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    urdf=ET.parse(a.urdf).getroot();limits=np.array([float(urdf.find(f"joint[@name='panda_joint{i}']/limit").get('velocity')) for i in range(1,8)]);rows=[]
    for case in ['740001_g035','740001_lookahead','740000_lookahead','740000_g105','740001_g105']:
        p=a.pull_root/case/'TRAJECTORY.json.gz';d=read(p);states=d['steps'];qv=np.array([x['qvel'][:7] for x in states]);commands=np.array([x['action'][:7] for x in states]);q=np.array([x['q'][:7] for x in states]);transfer=np.array([x['phase']=='TRANSFER' for x in states]);raw=np.array(next(x for x in d['plans'] if x['phase']=='TRANSFER')['position']);ref=next(x for x in reversed(states) if x['phase']=='LIFT')['tcp_T_tool'];R0=quat2mat(ref[3:])
        drift=lambda x:float(np.degrees(np.arccos(np.clip((np.trace(R0.T@quat2mat(x['tcp_T_tool'][3:]))-1)/2,-1,1))))
        rows.append(dict(case=case,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_success=d['summary']['success'],transfer_control_steps=int(transfer.sum()),actual_max_joint_velocity_rad_s=float(abs(qv).max()),actual_max_URDF_velocity_ratio=float(np.max(abs(qv)/limits)),frames_exceeding_nominal_planner_point9_velocity=int(np.any(abs(qv)>.9+1e-6,axis=1).sum()),command_max_finite_difference_velocity_rad_s=float(abs(np.diff(commands,axis=0)/.05).max()),command_max_finite_difference_acceleration_rad_s2=float(abs(np.diff(commands,n=2,axis=0)/.05**2).max()),actual20Hz_qvel_difference_peak_rad_s2=float(abs(np.diff(qv,axis=0)/.05).max()),transfer_raw_path_fd_velocity=float(abs(np.diff(raw,axis=0)/.05).max()),transfer_raw_path_fd_acceleration=float(abs(np.diff(raw,n=2,axis=0)/.05**2).max()),transfer_tracking_error_peak_rad=float(abs(commands[transfer]-q[transfer]).max()),transfer_attachment_rotation_peak_deg=max(drift(x) for x in states if x['phase']=='TRANSFER')))
    chronology=[]
    for label in ['g035','g070']:
        p=a.contact_root/('740001_'+label)/'TRAJECTORY.json.gz';d=read(p);ref=next(x for x in reversed(d['steps']) if x['phase']=='LIFT')['tcp_T_tool'];R0=quat2mat(ref[3:]);events=[x for s in d['steps'] for x in s['physics_substep_contacts'] if x['force_norms_N']['held_tool']>.5];first=events[0];before=d['steps'][first['control_step']-2]
        angle=float(np.degrees(np.arccos(np.clip((np.trace(R0.T@quat2mat(before['tcp_T_tool'][3:]))-1)/2,-1,1))))
        chronology.append(dict(label=label,first_tool_contact_over_half_N=dict(step=first['control_step'],substep=first['substep'],phase=first['phase']),preceding_control_frame=before['step'],preceding_attachment_rotation_deg=angle,preceding_attachment_translation_mm=float(np.linalg.norm(np.array(before['tcp_T_tool'][:3])-ref[:3])*1000),scope='100Hz contact force threshold and20Hz attachment chronology; not a no-contact proof below0.5N'))
    result=dict(urdf=str(a.urdf),urdf_sha256=hashlib.sha256(a.urdf.read_bytes()).hexdigest(),urdf_arm_velocity_limits_rad_s=limits.tolist(),urdf_acceleration_limit_present=False,planner_defaults='mplib defaults ones for velocity AND acceleration, native solver scales both by.9; not URDF velocity-scaled',execution='TracePlanner manual stride2 sends every2nd native TOPP position at20Hz, with endpoint included. Native timing/acceleration guarantees are not preserved.',limitations='20Hz physical qvel samples do not bound100Hz peaks; finite differences of commands are not actual actuator acceleration or torque; no hardware safety certification.',rows=rows,contact_chronology=chronology)
    (a.out/'DYNAMICS_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
