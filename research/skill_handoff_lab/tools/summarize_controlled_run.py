"""Private GPU/PhysX diagnostics and hashed output inventory, no training labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from controlled_paths import require_private_location


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    require_private_location(a.output)
    rows=[];reference=read(a.asset/'SITE_ASSET_MANIFEST.json')['reference_limits']
    low=np.array(reference['lower_rad'])+reference['soft_limit_margin_rad'];high=np.array(reference['upper_rad'])-reference['soft_limit_margin_rad']
    for file in sorted(a.gpu.glob('*/result.json')):
        result=read(file);request_path=file.with_name('request.json');request=read(request_path)
        q=np.array(result.get('trajectory_points_rad') or [])
        validation=read(file.with_name('validation.json'))
        row={'job':file.parent.name,'request_sha256':sha(request_path),'new_planning':result['observed_result'],
             'knots':len(q),'world_and_orientation_validation':validation['independent_validation'],
             'attached':request.get('attached_object_collision') is not None,
             'timing_s':result['timing_s'],'new_physics_execution':False,
             'full_execution_certified':False,'planner_yaml':'reconstructed six-axis adapter; original onsite YAML absent'}
        if len(q):
            row['reference_soft_margin_rad']=float(np.minimum(q-low,high-q).min())
            row['J3_peak_rad']=float(q[:,2].max())
            row['reference_soft_limits_pass']=bool(row['reference_soft_margin_rad']>=0)
        rows.append(row)
    (a.output/'NEW_GPU_REPLAY_SUMMARY.json').write_text(json.dumps({'publication':'CONTROLLED_NOT_PUBLIC',
        'not_fair_policy_comparison':'two archived requests have different scene, attachment, yaw and collision discretization',
        'replays':rows,'throughput_established':False,'two_cases_only':True},indent=2))
    if a.physics:
        states=np.load(a.physics/'PHYSX_STATES.npz')['rows'];result=read(a.physics/'PHYSX_REPLAY_RESULT.json')
        dt=result['physics_dt_s'];t=states[:,0]
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        axes[0].plot(t,np.degrees(np.max(abs(states[:,1:7]-states[:,7:13]),axis=1)))
        axes[0].set(xlabel='physics time (s)',ylabel='max joint tracking error (deg)',title='Actual Articulation vs command')
        qualification=read(a.physics/'INDEPENDENT_PHYSX_ACCEPTANCE.json')
        axes[1].plot(t,states[:,16]);axes[1].axhline(qualification['level_constraint_deg'],c='red',ls='--',label='bound transport request cap')
        axes[1].set(xlabel='physics time (s)',ylabel='level error (deg)',title='Actual tool posture, not planner posture');axes[1].legend(fontsize=7)
        acceleration=np.max(abs(np.diff(states[:,1:7],n=2,axis=0)),axis=1)/(dt*dt)
        axes[2].plot(t[2:],acceleration);axes[2].axhline(max(reference['max_acceleration_rad_s2']),c='red',ls='--',label='dated reference acceleration cap')
        axes[2].set(xlabel='physics time (s)',ylabel='max joint acceleration (rad/s^2)',title='Unfiltered 120 Hz finite differences');axes[2].legend(fontsize=7)
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle('CONTROLLED | real PhysX transport replay, NOT RL or successful placement',fontsize=11)
        fig.tight_layout();fig.savefig(a.output/'PHYSX_EXECUTION_DIAGNOSIS.png',dpi=160);plt.close(fig)
    inventory=[]
    for file in sorted(a.output.rglob('*')):
        if file.is_file() and file.name!='ANALYSIS_ARTIFACT_MANIFEST.json':
            inventory.append({'path':str(file.relative_to(a.output)),'bytes':file.stat().st_size,'sha256':sha(file)})
    (a.output/'ANALYSIS_ARTIFACT_MANIFEST.json').write_text(json.dumps({'publication':'CONTROLLED_NOT_PUBLIC',
        'code_execution_base':'b0c8b58','artifacts':inventory},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['gpu','asset','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--physics',type=Path);main(p.parse_args())
