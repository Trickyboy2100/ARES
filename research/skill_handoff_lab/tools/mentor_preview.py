"""Evidence-labelled scientific plots; absent experiments have no fake curves."""
import argparse, hashlib, json, shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from itertools import product
from ares_r.motion.independent_path_validation import validate_dense_world
from ares_r.perception.robot_collision import arm_link_transforms

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--static',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.static/'ares_usd_geometry.png',a.output/'fig01_isaac_scene.png')
    rows=[];paths=[]
    for directory in sorted(a.run.glob('execution_*/*')):
        if not directory.is_dir() or not (directory/'validation.json').exists():continue
        validation=json.loads((directory/'validation.json').read_text())
        req=json.loads((directory/'request.json').read_text())
        result=json.loads((directory/'result.json').read_text()) if (directory/'result.json').exists() else {}
        status=validation['status']
        check=directory/'diagnostic_independent_validation.json'
        if check.exists() and json.loads(check.read_text())['diagnostic_scope_pass']:status='DIAGNOSTIC_SCOPE_PASS'
        rows.append([directory.parent.name+'/'+directory.name,req.get('candidate_id','historical replay'),
            status,validation.get('failure_class'),round(validation['wall_s'],3)])
        if result.get('tcp_path_body_m'):paths.append((directory.name,result,req))
    preflight=[]
    requests=a.run/'batch_v3'
    for path in sorted(requests.glob('request_[0-9]*.json')):
        req=json.loads(path.read_text());req['robot_yaml_urdf']=str(requests/'robot.urdf')
        entry={'candidate_id':req['candidate_id'],'source_request_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for key,state in [('start','start_rad'),('goal','goal_rad')]:
            entry[key]=validate_dense_world([req[state],req[state]],req,subdivisions=2)
        preflight.append(entry)
    (a.output/'endpoint_preflight.json').write_text(json.dumps(preflight,indent=2))
    fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
    for name,result,req in paths:
        xyz=result['tcp_path_body_m'];ax.plot(*zip(*xyz),linewidth=3,label='Fixed 1-degree diagnostic | raw cuRobo')
        ax.scatter(*xyz[0],s=30);ax.scatter(*xyz[-1],s=30)
        joints=result['trajectory_points_rad'];body=np.asarray(req['T_body_model'])
        for index,color in [(0,'black'),(-1,'purple')]:
            links=arm_link_transforms(a.run/'batch_v3/robot.urdf',joints[index])
            chain=[(body@links[name])[:3,3] for name in ['base_link']+['link%d'%i for i in range(1,7)]]
            ax.plot(*zip(*chain),color=color,linewidth=2,alpha=.7)
        for key,box in req['compiled_scene']['cuboids'].items():
            pose=box['pose'];q=pose[3:];rot=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
            corners=np.asarray(list(product((-1,1),repeat=3)))*np.asarray(box['dims'])/2
            local=corners@rot.T+pose[:3]
            world=(np.c_[local,np.ones(8)]@np.asarray(req['T_body_model']).T)[:,:3]
            for i in range(8):
                for j in range(i+1,8):
                    if np.sum(corners[i]!=corners[j])==1:
                        ax.plot(*zip(world[i],world[j]),color='grey',alpha=.25,linewidth=.7)
    if paths:
        ax.set_xlabel('BODY X (m)');ax.set_ylabel('BODY Y (m)');ax.set_zlabel('BODY Z (m)')
        ax.set_title('REAL cuRobo diagnostic + frozen collision obstacles\nOne path only: paired handoff trajectories NOT COMPLETED')
        inset=fig.add_axes([.66,.7,.25,.18]);q=np.asarray(paths[0][1]['trajectory_points_rad'])
        inset.plot(np.arange(len(q))*.008,np.rad2deg(q[:,0]-q[0,0]))
        inset.set_xlabel('Planner time (s)',fontsize=8);inset.set_ylabel('Joint 1 delta (deg)',fontsize=8)
        inset.tick_params(labelsize=7)
    else:
        ax.set_axis_off();ax.text2D(.08,.5,'NOT COMPLETED: no cuRobo trajectory produced\nNo interpolation or fabricated path shown',transform=ax.transAxes,fontsize=16)
    fig.savefig(a.output/'fig02_handoff_trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6));axes[0].axis('off')
    table=axes[0].table(cellText=[[x[0],x[2].replace('PLANNER_NO_VALIDATED_PATH','NO VALIDATED PATH'),x[4]] for x in rows],
        colLabels=['Run','Oracle status','Wall s'],colWidths=[.25,.55,.2],loc='center')
    table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2)
    axes[0].set_title('Real Oracle attempts (software error = UNKNOWN)')
    for index,key in enumerate(('start','goal')):
        axes[1].bar([i+(index-.5)*.3 for i in range(len(preflight))],
            [x[key]['min_clearance_m']*1000 for x in preflight],width=.3,label=key)
    axes[1].axhline(0,color='black');axes[1].legend();axes[1].set_xticks(range(len(preflight)),['candidate '+str(i) for i in range(len(preflight))])
    axes[1].set_ylabel('Independent endpoint proxy clearance (mm)')
    axes[1].set_title('Cabinet frozen candidates | CPU URDF check\nIncludes uncommissioned conservative chassis proxy')
    fig.tight_layout();fig.savefig(a.output/'fig03_candidate_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.axis('off');ax.text(.05,.55,
        'REAL TRAINING: NOT COMPLETED\nNo qualified branching dataset / no valid scene split\nRanker, MaskablePPO and CASH results: NOT TESTED\nFixture unit tests are excluded from research results',fontsize=16)
    fig.savefig(a.output/'fig04_learning_comparison.png',dpi=150);plt.close(fig)
    manifest={'fig01':'STATIC_USD_CPU_GEOMETRY_RENDER; NOT Isaac runtime',
        'static_source':json.loads((a.static/'manifest.json').read_text()),
        'fig02':'ONLY raw cuRobo paths when present; missing comparison explicitly incomplete',
        'fig03':'Real Oracle attempt status + independent CPU endpoint proxy check',
        'fig04':'NOT_COMPLETED; no learning curves generated','oracle_rows':rows,
        'sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.output.glob('fig*.png')}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
