"""Saved dual-arm trajectory dynamics audit;20Hz evidence, not a safety certificate."""
import argparse,gzip,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,action='append',required=True);ap.add_argument('--urdf',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    xml=ET.parse(a.urdf).getroot();limits=np.array([float(xml.find(f"joint[@name='panda_joint{i}']/limit").get('velocity')) for i in range(1,8)]);rows=[]
    for root in a.root:
        for p in sorted(root.glob('*/TRAJECTORY.json.gz')):
            with gzip.open(p,'rt') as f:d=json.load(f)
            robots=[]
            for i in ['0','1']:
                states=d['steps'];v=np.array([s['measurement'][i]['qvel'][:7] for s in states]);q=np.array([s['measurement'][i]['q'][:7] for s in states]);cmd=np.array([s['action']['panda_wristcam-'+i][:7] for s in states]);plans=[np.array(x['position']) for x in d['plans'] if x.get('arm')==int(i) and x.get('position') and len(x['position'])>=3]
                robots.append(dict(arm=int(i),actual20Hz_max_velocity_rad_s=float(abs(v).max()),actual20Hz_max_URDF_velocity_ratio=float(np.max(abs(v)/limits)),actual20Hz_frames_over_URDF_velocity=int(np.any(abs(v)>limits,axis=1).sum()),actual20Hz_frames_over_planner_point9_velocity=int(np.any(abs(v)>.9+1e-6,axis=1).sum()),actual20Hz_velocity_fd_accel_rad_s2=float(abs(np.diff(v,axis=0)/.05).max()),command_fd_velocity_rad_s=float(abs(np.diff(cmd,axis=0)/.05).max()),command_fd_acceleration_rad_s2=float(abs(np.diff(cmd,n=2,axis=0)/.05**2).max()),tracking_max_rad=float(abs(cmd-q).max()),raw_planned_path_fd_velocity=max([float(abs(np.diff(x,axis=0)/.05).max()) for x in plans],default=None),raw_planned_path_fd_acceleration=max([float(abs(np.diff(x,n=2,axis=0)/.05**2).max()) for x in plans],default=None)))
            rows.append(dict(case=p.parent.name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),success=d['summary']['success'],robots=robots))
    summary=dict(episodes=len(rows),native_control_steps=sum(len(json.load(gzip.open(r['path'],'rt'))['steps']) for r in rows),urdf=str(a.urdf),urdf_sha256=hashlib.sha256(a.urdf.read_bytes()).hexdigest(),URDF_arm_velocity_limits=limits.tolist(),URDF_acceleration_limit_available=False,actual20Hz_max_URDF_ratio=max(x['actual20Hz_max_URDF_velocity_ratio'] for r in rows for x in r['robots']),command_fd_accel_peak=max(x['command_fd_acceleration_rad_s2'] for r in rows for x in r['robots']),status='DYNAMIC_SKILL_CONTRACT_NOT_PROVEN',scope='Native success and action-space validity do not imply dynamical certification. Sampled arm velocity remains below URDF bounds, but100Hz peaks are unmeasured; stride3 does not preserve original.9rad/s/.9rad/s2 TOPP timing; command finite differences are not physical acceleration. Gripper dynamic limits not certified.',rows=rows)
    (a.out/'DYNAMICS_AUDIT.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
