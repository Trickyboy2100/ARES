"""Real-data mentor figures; never substitute rendering for physics evidence."""
import argparse,json,hashlib,shutil,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from ares_r.perception.robot_collision import arm_link_transforms
from research.skill_handoff_lab.bench.model import PinnedRobotModel

SIGNS=np.array(list(itertools.product([-1,1],repeat=3)))
FACES=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
def box(ax,t,dims,color,alpha=.12):
    points=SIGNS*np.array(dims)/2;points=points@t[:3,:3].T+t[:3,3]
    ax.add_collection3d(Poly3DCollection([points[i] for i in FACES],facecolor=color,edgecolor=color,alpha=alpha,linewidth=.4))
def model_scene(ax,m,request):
    for name,b in request['compiled_scene']['cuboids'].items():
        t=np.eye(4);t[:3,:3]=Rotation.from_quat([*b['pose'][4:],b['pose'][3]]).as_matrix();t[:3,3]=b['pose'][:3]
        box(ax,m.body_base@t,b['dims'],'#858d98',.08)
    ax.set(xlabel='BODY X (m)',ylabel='BODY Y (m)',zlabel='BODY Z (m)',xlim=(-.45,.85),ylim=(-.85,.45),zlim=(.65,1.95))
    ax.set_box_aspect([1.3,1.3,1.3]);ax.view_init(22,-62)
def robot(ax,m,q,color,alpha=.7):
    transforms=arm_link_transforms(m.urdf,q)
    xyz=np.array([(m.body_base@transforms[k])[:3,3] for k in ['base_link']+[f'link{i}' for i in range(1,7)]])
    ax.plot(*xyz.T,color=color,alpha=alpha,linewidth=3,marker='o',markersize=3)
def held(ax,m,q,r,color):
    t=m.body_base@arm_link_transforms(m.urdf,q)['link6']@np.array(r['attached_object_collision']['T_link6_object'])
    box(ax,t,r['tray_provenance']['observed_world_aabb_m'],color,.3)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True)
    p.add_argument('--static-preview',type=Path,required=True)
    p.add_argument('--isaac-capture',type=Path);a=p.parse_args()
    out=a.output/'03_PRELIMINARY_RESULTS';out.mkdir(parents=True,exist_ok=True)
    method=a.output/'01_METHOD';method.mkdir(parents=True,exist_ok=True)
    (a.output/'02_PAPER_PLAN').mkdir(parents=True,exist_ok=True)
    preview_target=out/'fig01_static_usd_preview_NOT_ISAAC.png'
    if a.static_preview.resolve()!=preview_target.resolve():
        shutil.copyfile(a.static_preview,preview_target)
    capture=None
    if a.isaac_capture:
        capture=json.loads((a.isaac_capture/'capture_manifest.json').read_text())
        process=json.loads((a.isaac_capture/'process_manifest.json').read_text())
        if capture['evidence_level']!='ISAAC_5_1_STATIC_RENDER' or process['returncode']!=0:
            raise ValueError('Isaac figure requires successful real capture provenance')
        fig,ax=plt.subplots(figsize=(12.8,10.2))
        ax.imshow(plt.imread(a.isaac_capture/'isaac51_scene.png'));ax.set_axis_off()
        ax.set_title('ACTUAL ISAAC SIM 5.1 / RTX4090 GPU1 / STATIC_RENDER',fontsize=14)
        fig.text(.5,.025,'ARES original USD | BODY mapping unqualified | Tray dynamic mesh convex-hull fallback\n'
            'No task execution / no contact grasp / no vectorized physics training',ha='center',fontsize=10)
        fig.tight_layout(rect=[0,.065,1,1]);fig.savefig(out/'fig01_isaac_scene.png',dpi=120);plt.close(fig)
        shutil.copyfile(a.isaac_capture/'capture_manifest.json',out/'fig01_isaac_capture_manifest.json')
    m=PinnedRobotModel(a.ares,a.ares_r);jobs=[]
    for index in [0,2]:
        job=a.run/'execution_continuation'/str(index)
        r=json.loads((job/'request.json').read_text());v=json.loads((job/'e0_independent_validation.json').read_text())
        if not v['model_validated_path']:raise ValueError('paired plot requires independently qualified raw paths')
        result=json.loads((job/'result.json').read_text());jobs.append((r,result,v))
    fig=plt.figure(figsize=(13,6));ax=fig.add_subplot(121,projection='3d');small=fig.add_subplot(122)
    model_scene(ax,m,jobs[0][0])
    for (r,result,v),color in zip(jobs,['#1769a0','#e0782e']):
        points=np.array(result['trajectory_points_rad']);xyz=np.array([m.fk(q)[:3,3] for q in points])
        ax.plot(*xyz.T,color=color,linewidth=3,label=f"roll {r['candidate_features']['roll_branch']}")
        for q in [points[0],points[-1]]:robot(ax,m,q,color,.6);held(ax,m,q,r,color)
        small.plot(np.arange(len(points))*.008,np.linalg.norm(points-points[0],axis=1),color=color,
            label=f"roll {r['candidate_features']['roll_branch']}: path {v['joint_path_rad']:.3f} rad")
    ax.legend();ax.set_title('Two real cuRobo held-transport paths\nURDF FK + declared obstacle/held cuboids')
    small.set(xlabel='Raw planner trajectory time (s)',ylabel='Joint displacement from start (rad)',title='Same BODY TCP task, different roll terminals')
    small.legend();fig.suptitle('MODELED PLANNING VALIDATION ONLY | historical lower chassis, no contact/Isaac execution',fontsize=11)
    fig.tight_layout();fig.savefig(out/'fig02_held_trajectories.png',dpi=160);plt.close(fig)
    summaries={name:json.loads((a.run/name/'e0_summary.json').read_text()) for name in ['execution_lower','execution_continuation','execution_variations']}
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    states=[];times=[]
    for v in summaries['execution_lower']:
        job=a.run/'execution_lower'/str(len(states));result=json.loads((job/'result.json').read_text())
        states.append(1 if v['model_validated_path'] else 0 if result.get('trajectory_points_rad') else -1)
        times.append(result['timing_s']['planning_reported_solve'])
    axes[0].imshow([states],vmin=-1,vmax=1,cmap='RdYlGn',aspect='auto');axes[0].set_xticks(range(8));axes[0].set_yticks([])
    for i,s in enumerate(states):axes[0].text(i,0,{1:'PASS',0:'REJECT',-1:'UNKNOWN'}[s],ha='center',fontsize=7)
    axes[0].set(title='Cross-IK raw calls\nNO SOLUTION is unknown, not task-infeasible',xlabel='Frozen request index')
    axes[1].bar(range(8),times);axes[1].set(xlabel='Frozen request index',ylabel='Actual solve s',title='Single RTX4090 / 4 seeds / 2 attempts')
    normal=[];flipped=[]
    for group in range(6):
        vals=[]
        for roll in [0,1]:
            name=f'E0-goal-{group}-roll-{roll}';v=next(v for v in summaries['execution_variations'] if v['candidate_id']==name)
            vals.append(v['joint_path_rad'])
        normal.append(vals[0]);flipped.append(vals[1])
    x=np.arange(6);axes[2].bar(x-.17,normal,.34,label='NORMAL');axes[2].bar(x+.17,flipped,.34,label='FLIPPED')
    axes[2].legend();axes[2].set(xlabel='Pre-frozen goal group',ylabel='Measured joint path (rad)',title='12/12 model-qualified paths\nAll feasible; cost differences only')
    fig.suptitle('REAL planning + independent validation | not physical placement success',fontsize=11)
    fig.tight_layout();fig.savefig(out/'fig03_candidate_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,5));labels=[];values=[]
    for seed in [17,23]:
        run=a.run/f'learning_seed{seed}_v2';curve=json.loads((run/'cash_curve.json').read_text())
        rank=json.loads((run/'ranker_curve.json').read_text())
        axes[0].plot([r['steps'] for r in curve],[r['train_mean_return'] for r in curve],label=f'CASH train seed {seed}')
        axes[0].plot([r['steps'] for r in curve],[r['validation_mean_return'] for r in curve],linestyle='--',label=f'validation {seed}')
        axes[1].plot([r['steps'] for r in curve],[r['effect_loss'] for r in curve],label=f'effect MSE seed {seed}')
        if seed==17:
            comp=json.loads((run/'comparison.json').read_text());labels=[r['method'] for r in comp];values=[r['mean_cost_joint_rad'] for r in comp]
    axes[0].legend(fontsize=7);axes[0].set(xlabel='Cached one-step interactions',ylabel='1 - measured joint path rad',title='Actual CASH updates / real Oracle cache')
    axes[1].legend(fontsize=7);axes[1].set(xlabel='Cached interactions',ylabel='Auxiliary effect MSE',title='Successor-effect head jointly trained')
    axes[2].bar(range(len(labels)),values);axes[2].set_xticks(range(len(labels)),labels,rotation=55,ha='right',fontsize=8)
    axes[2].set(ylabel='Test mean joint path rad (lower better)',title='Two frozen test groups, both seeds agree\nCASH ties greedy/fixed; no superiority')
    fig.suptitle('PLANNING COST PILOT ONLY | 3 train / 1 validation / 2 test groups; not multi-stage/Isaac RL',fontsize=11)
    fig.tight_layout();fig.savefig(out/'fig04_real_learning.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for seed in [17,23]:
        run=a.run/f'learning_seed{seed}_v2';rank=json.loads((run/'ranker_curve.json').read_text())
        axes[0].semilogy([r['epoch'] for r in rank],[r['train_mse'] for r in rank],label=f'train {seed}')
        axes[0].semilogy([r['epoch'] for r in rank],[r['validation_mse'] for r in rank],linestyle='--',label=f'validation {seed}')
        import csv
        rows=list(csv.DictReader((run/'ppo/progress.csv').open()));rows=[r for r in rows if r.get('train/entropy_loss')]
        axes[1].plot([int(r['time/total_timesteps']) for r in rows],[-float(r['train/entropy_loss']) for r in rows],label=f'PPO entropy {seed}')
    axes[0].set(xlabel='Regression epoch',ylabel='Measured cost prediction MSE',title='Supervised cost Ranker, not feasibility classifier')
    axes[1].set(xlabel='Actual PPO cache steps',ylabel='Policy entropy',title='Ordinary MaskablePPO optimization log')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'fig05_ranker_ppo_logs.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));ax.set_axis_off()
    blocks=[('Observed skill state\n+ next-skill contract',.02),('Legal candidate set\nIK / roll / preparation',.27),('CASH policy + value\n+ successor-effect head',.53),('cuRobo + independent\ntrajectory validation',.79)]
    for label,x in blocks:ax.text(x,.6,label,transform=ax.transAxes,bbox={'boxstyle':'round','facecolor':'#e5edf6'},fontsize=11,ha='left',va='center')
    for x in [.21,.47,.73]:ax.annotate('',xy=(x+.045,.6),xytext=(x,.6),xycoords='axes fraction',arrowprops={'arrowstyle':'->'})
    ax.text(.03,.22,'Implemented now: one-step real planning-cost cache + Ranker / PPO / CASH auxiliary updates.\nProposed next: D0 → D1 → D2 conditional transitions, online feedback and physical validation.\nNo claim that a successful cost pilot proves long-horizon or physical task success.',fontsize=11,transform=ax.transAxes)
    fig.tight_layout();fig.savefig(method/'CASH_METHOD_V0.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d');r,result,v=jobs[0];pts=np.array(result['trajectory_points_rad'])
    indexes=np.linspace(0,len(pts)-1,32).astype(int)
    def animate(frame):
        ax.clear();model_scene(ax,m,r);q=pts[indexes[frame]];robot(ax,m,q,'#1769a0');held(ax,m,q,r,'#e6b72c')
        ax.set_title(f'RAW cuRobo FK replay frame {indexes[frame]}/{len(pts)-1}\nNOT Isaac physics / NOT training video',fontsize=10)
        return []
    FuncAnimation(fig,animate,frames=len(indexes),interval=100).save(out/'held_path_FK_REPLAY_NOT_PHYSICS.gif',writer=PillowWriter(fps=10));plt.close(fig)
    files=list(a.output.rglob('*.png'))+list(a.output.rglob('*.gif'))
    manifest={'evidence':'REAL_CUROBO_UNDER_DECLARED_RESEARCH_MODEL_AND_ACTUAL_CACHE_TRAINING',
        'Isaac_physical_environments':0,'Isaac_training_video':'NOT_TESTED','full_chassis_qualified':False,
        'source_run':str(a.run.resolve()),'source_cache_sha256':hashlib.sha256((a.run/'cache.json').read_bytes()).hexdigest(),
        'files':{str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    manifest['learning_manifests']={str(seed):json.loads((a.run/f'learning_seed{seed}_v2/manifest.json').read_text()) for seed in [17,23]}
    manifest['isaac_static_capture']=capture
    manifest['isaac_static_limitations']=['Original USD BODY mapping unqualified',
        'PhysX dynamic Tray/Mesh triangle geometry fell back to convex hull',
        'No task execution, contact grasp, vectorized physics or Isaac policy training validated']
    (a.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
if __name__=='__main__':main()
