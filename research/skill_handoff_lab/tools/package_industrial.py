"""Measured industrial PPO pilot packaging; absent experiments remain untested."""
import argparse,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);rows=[]
methods=[('nominal','industrial_nominal16_seed31_v7'),('PPO','industrial_ppo16_seed31_v7'),
         ('residual PPO','industrial_residual16_seed31_v7'),('contract PPO','industrial_contract16_seed31_v8')]
fig,ax=plt.subplots(1,2,figsize=(11,4))
for label,name in methods:
    root=a.runs/name
    if not (root/'frozen_evaluation.json').is_file():
        rows.append({'method':label,'status':'NOT_TESTED'});continue
    manifest=json.loads((root/'manifest.json').read_text());evaluation=json.loads((root/'frozen_evaluation.json').read_text())
    cases=evaluation['episodes'];n=len(cases);success=sum(r['success'] for r in cases);rate=success/n if n else 0
    z=1.96;den=1+z*z/n;mid=(rate+z*z/(2*n))/den;half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
    row={'method':label,'status':'PASS_RUN','test_cases':n,'successes':success,'success_rate':rate,
        'wilson_95_interval':[mid-half,mid+half],'mean_time_s':np.mean([r['episode_control_steps']/60 for r in cases]),
        'mean_depth_m':np.mean([r['depth_m'] for r in cases]),'failed':sum(r['failed'] for r in cases),
        'timeout':sum(r['timeout'] for r in cases),'training_interactions':0 if label=='nominal' else 65536,
        'seed':31,'test_seed':1731,'scene_scope':'single parameterized wide cabinet, isolated held arm',
        'runtime_source_sha256':manifest['source_sha256']};rows.append(row)
    episodes=json.loads((root/'industrial_episodes.json').read_text())
    if label!='nominal':
        x=[];rates=[];returns=[]
        for start in range(0,len(episodes),32):
            batch=episodes[start:start+32];x.append(batch[-1]['control_step']*16)
            rates.append(np.mean([r['success'] for r in batch]))
            # Reconstruct terminal reward from measured state and the frozen equation.
            values=[]
            for r in batch:
                distance=math.hypot(.06-r['depth_m'],r['alignment_m'])
                terminal=2*math.exp(-25*distance)+5*r['depth_m']-30*r['alignment_m']-2*r['orientation_error_rad']-.002*min(r['contact_force_N'],100)-.01+10*r['success']
                values.append(r['return_before_terminal_reward']+terminal)
            returns.append(np.mean(values))
        ax[0].plot(x,rates,label=label);ax[1].plot(x,returns,label=label)
for axis in ax:axis.legend();axis.grid(alpha=.2);axis.set_xlabel('Real PhysX environment interactions')
ax[0].set_ylabel('Episode success\n(32-episode blocks)');ax[0].set_ylim(-.02,1.02)
ax[1].set_ylabel('Episode return\n(measured terminal reconstructed)')
fig.suptitle('T1 cabinet entry: actual physics training, seed 31\nSingle wide research geometry; NOT cross-task/generalization proof')
fig.tight_layout(rect=(0,0,1,.88));fig.savefig(a.output/'fig_industrial_learning.png',dpi=150);plt.close(fig)
(a.output/'comparison.json').write_text(json.dumps(rows,indent=2))
completed=[r for r in rows if r['status']=='PASS_RUN']
fig,axis=plt.subplots(figsize=(7,4));axis.bar([r['method'] for r in completed],[r['success_rate'] for r in completed])
axis.set_ylim(0,1.1);axis.set_ylabel('Frozen test success rate');axis.set_title('Same 64 fixed physical cases / seed 1731\nPilot only: no multi-seed or unseen geometry claim')
for i,r in enumerate(completed):axis.text(i,r['success_rate'],f"{r['successes']}/{r['test_cases']}",ha='center',va='bottom')
fig.tight_layout();fig.savefig(a.output/'fig_industrial_test.png',dpi=150);plt.close(fig)
raw=a.output/'raw';raw.mkdir(exist_ok=True)
for root in sorted(a.runs.iterdir()):
    if root.is_dir():shutil.copytree(root,raw/root.name,dirs_exist_ok=True)
    elif root.is_file():shutil.copy2(root,raw/root.name)
for filename in ['physical_scene.png','online_ppo_physics.gif','video_frame_records.json']:
    source=a.runs/'industrial_ppo16_seed31_v7'/filename
    if source.is_file():shutil.copy2(source,a.output/filename)
demo=a.runs/'industrial_policy_demo_v9/policy_evaluation.gif'
if demo.is_file():shutil.copy2(demo,a.output/'single_env_success_failure.gif')
fig,axis=plt.subplots(figsize=(11,4));axis.axis('off')
boxes=[(.04,.65,'Observed state (25)'),(.04,.23,'Skill Contract (15)'),(.30,.65,'PPO Actor-Critic (40)'),
       (.55,.65,'Nominal + bounded residual\ncommand integration'),(.80,.65,'DLS IK / soft limits'),
       (.80,.23,'Real PhysX execution\nindependent success / reward')]
for x,y,label in boxes:axis.text(x,y,label,ha='center',va='center',transform=axis.transAxes,
    bbox={'boxstyle':'round,pad=.6','facecolor':'#e4edf5','edgecolor':'#31536a'},fontsize=10)
for start,end in [((.13,.65),(.23,.65)),((.14,.26),(.26,.56)),((.41,.65),(.46,.65)),
                  ((.68,.65),(.72,.65)),((.80,.53),(.80,.36)),((.14,.23),(.65,.56)),((.70,.23),(.34,.54))]:
    axis.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#31536a'})
axis.set_title('Implemented contract-conditioned residual PPO prototype\nSingle-contract training completed; cross-contract transfer NOT_TESTED',pad=15)
fig.tight_layout();fig.savefig(a.output/'fig_contract_method.png',dpi=150);plt.close(fig)
manifest=[{'path':str(f.relative_to(a.output)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
    for f in sorted(a.output.rglob('*')) if f.is_file() and f.name!='BUNDLE_MANIFEST.json']
(a.output/'BUNDLE_MANIFEST.json').write_text(json.dumps({'base_commit':'d49af0d','artifacts':manifest,
    'scope':'actual industrial T1 pilot, no full robot or measured cabinet safety'},indent=2))
print(json.dumps(rows,indent=2))
