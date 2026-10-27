"""Actual completed online episode curves and 32-case DEVELOPMENT audits.

Never reads sealed ID/Geometry/Dynamics test files. No fitted/synthetic points,
no confidence band for a single seed, no train-check8 as independent testing.
"""
import argparse,csv,hashlib,json,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
SCRIPT_BYTES=Path(__file__).read_bytes()
SCRIPT_SHA256=hashlib.sha256(SCRIPT_BYTES).hexdigest()


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(path,rows):
    if not rows:return
    with path.open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(path):return json.loads(path.read_text())
def audit(run):
    p=run/'INDEPENDENT_CLOSED_LOOP.json'
    return p if p.exists() else run/'INDEPENDENT_TASK_AND_CONTACT.json'
def name(run):
    known={'feedback32_zero_steering':'Feedback (zero lateral steering)', 'feedback32_yaw_zero':'Feedback (zero-yaw reference)'}
    if run.name in known:return known[run.name]
    clean=re.sub(r'_seed\d+','',run.name).replace('_development','')
    known={'event_qualified':'Event execution-active PPO','event_ordinary':'Event ordinary PPO','periodic_qualified':'Periodic execution-active PPO'}
    return known.get(clean,clean.replace('effect_param_ppo','Effect-history PPO').replace('param_ppo','Parameter PPO').replace('_',' '))

def shortname(run,m):
    n=run.name
    if 'feedback' in n:return 'Feedback nominal' if 'no_recontact' in n else 'Feedback recontact' if 'recontact' in n else 'Feedback'
    mode=('Fixed active' if 'fixed_qualified' in n else 'Fixed ordinary' if 'fixed_ordinary' in n else 'Periodic active' if 'periodic_qualified' in n else 'Exec-active' if 'event_qualified' in n else
          'Ordinary' if 'event_ordinary' in n else 'History' if 'effect' in n else 'Parameter')
    return f'{mode} s{m["seed"]}'


def main(a):
    figures=a.output/'figures';tables=a.output/'tables';figures.mkdir(parents=True,exist_ok=True);tables.mkdir(exist_ok=True)
    train={};development={};inputs={};trainrows=[];evalrows=[]
    for round_name in ['r1','r2','r3']:
        base=a.root/round_name
        if not base.exists():continue
        for run in sorted(base.iterdir()):
            mfile=run/'manifest.json'
            if not mfile.exists() or not (run/'RUN_COMPLETE.json').exists():continue
            m=load(mfile)
            formal=(m['online_training_interactions']>=32768 and 'seed' in run.name and 'probe' not in run.name
                    and (round_name!='r3' or m['observation_dim']>=87))
            if formal:
                episodes=load(run/'training_episodes.json');assert all(not e['evaluation'] for e in episodes)
                train.setdefault(round_name,[]).append((run,m,episodes))
                for file in [mfile,run/'training_episodes.json',run/'task_source.py']:
                    inputs[str(file.resolve())]=sha(file)
                for i,e in enumerate(episodes):
                    trainrows.append({'round':round_name.upper(),'run':run.name,'seed':m['seed'],'episode_index':i,'online_interactions_when_done':e['env_interactions'],'episode_return':e['return'],'raw_training_success':e['success'],'duration_s':e['steps']/60,'failure_reason':e['failure_reason'],'source_episode_sha256':sha(run/'training_episodes.json')})
            if m['evaluation_role']=='development' and m['test_episodes']==32 and 'probe' not in run.name and run.name not in {'feedback32_gpu0','feedback32_gpu1'} and audit(run).exists():
                rows=load(run/'frozen_evaluation.json');ar=load(audit(run));assert len(rows)==len(ar['cases'])==32
                development.setdefault(round_name,[]).append((run,m,rows,ar))
                for file in [mfile,run/'frozen_evaluation.json',audit(run),run/'run_config.json']:inputs[str(file.resolve())]=sha(file)
                episode_limit=load(run/'run_config.json')['episode_seconds']
                armap={r['case_id']:r for r in ar['cases']}
                for e in sorted(rows,key=lambda e:e['case_id']):
                    q=armap[e['case_id']]
                    evalrows.append({'round':round_name.upper(),'run':run.name,'seed':m['seed'],'case_id':e['case_id'],'scene_group':e['scene_group'],'raw_success':e['success'],'independent_success':q['independent_task_pass'],'failure_reason':e['failure_reason'],'physics_duration_s':e['steps']/60,'episode_limit_s':episode_limit,'final_distance_m':e['distance_m'],'scene_sha256':sha(run/'scene.json'),'success_contract':json.dumps(m['success_contract'],sort_keys=True),'tool_contract':json.dumps(m.get('tool_contract',{}),sort_keys=True),'audit_sha256':sha(audit(run))})
    outputs=[]
    if train:
        plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5})
        fig,axes=plt.subplots(len(train),2,figsize=(7.2,2.35*len(train)+.4),squeeze=False)
        colors=['#177e91','#c77b3f','#6e68a8','#4a8c69']
        for row,(round_name,runs) in enumerate(train.items()):
            for index,(run,m,e) in enumerate(runs):
                color=('#177e91' if ('effect' in run.name or 'event_qualified' in run.name) else
                       '#6e68a8' if 'periodic_qualified' in run.name or 'fixed_qualified' in run.name else
                       '#808c94' if 'fixed_ordinary' in run.name else '#c77b3f')
                linestyle={31:'-',41:'--',51:':'}.get(m['seed'],'-.')
                x=np.array([v['env_interactions'] for v in e]);r=np.array([v['return'] for v in e]);s=np.array([v['success'] for v in e],float)
                for col,y in enumerate([r,s]):
                    axes[row,col].scatter(x/1000,y,c=color,alpha=.17,s=12)
                    if len(y)>=8:
                        axes[row,col].plot(x[7:]/1000,np.convolve(y,np.ones(8)/8,'valid'),color=color,ls=linestyle,label=shortname(run,m))
                axes[row,0].set_title(f'{round_name.upper()}: actual training return')
                axes[row,1].set_title(f'{round_name.upper()}: raw episode success')
            for ax in axes[row]:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Online interactions (thousands)')
            axes[row,0].set_ylabel('Episode return');axes[row,1].set_ylabel('Success / 8-episode mean');axes[row,1].set_ylim(-.05,1.05);axes[row,0].legend(fontsize=8.5,frameon=False,ncol=2,handlelength=1.5,columnspacing=.6)
        fig.suptitle('Actual online PhysX learning — not frozen-test performance',fontweight='bold',fontsize=11)
        fig.text(.015,.01,'Dots: completed episodes; lines: trailing 8-episode mean. No fitted data or cross-seed CI.\nUnfinished episodes unlabeled. Different contracts not pooled; diagnostics / visual rerun excluded.',fontsize=8.5,color='#526676')
        fig.tight_layout(rect=(0,.065,1,.96))
        for ext in ['pdf','svg','png']:
            p=figures/f'FigR_online_learning.{ext}';fig.savefig(p,dpi=160,bbox_inches='tight');outputs.append(p)
        plt.close(fig)
    if development:
        heights=[max(1.4,len(runs)*.23+.7) for runs in development.values()]
        fig,axes=plt.subplots(len(development),1,figsize=(7.2,sum(heights)+.4),squeeze=False,gridspec_kw={'height_ratios':heights})
        for ax,(round_name,runs) in zip(axes[:,0],development.items()):
            matrix=[];labels=[]
            for run,m,rows,ar in runs:
                arm={r['case_id']:r for r in ar['cases']};matrix.append([int(arm[i]['independent_task_pass']) for i in range(32)])
                labels.append(f'{shortname(run,m)}: {sum(matrix[-1])}/32')
            matrix=np.array(matrix)
            ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap=ListedColormap(['#bd795d','#4d927f']))
            ax.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(32),xticklabels=range(32),xlabel='DEVELOPMENT case ID (not sealed final test)',title=f'{round_name.upper()}: independent qualification; brown fail / green pass')
            ax.tick_params(axis='x',labelsize=6.5);ax.tick_params(axis='y',labelsize=9);ax.set_xticks(np.arange(-.5,32,1),minor=True);ax.set_yticks(np.arange(-.5,len(labels),1),minor=True);ax.grid(which='minor',color='white',lw=.8);ax.tick_params(which='minor',bottom=False,left=False)
        fig.suptitle('All paired cases: successes, shared failures and regressions',fontweight='bold',fontsize=11)
        fig.text(.015,.005,'Actual independently audited evaluations; same case IDs are paired, not independent replicates.\nTraining check8 / probes excluded. Development evidence only; no final-test advantage claim.',fontsize=8.5,color='#526676');fig.tight_layout(rect=(0,.075,1,.96))
        for ext in ['pdf','svg','png']:
            p=figures/f'FigR_development_cases.{ext}';fig.savefig(p,dpi=160,bbox_inches='tight');outputs.append(p)
        plt.close(fig)
    writecsv(tables/'ONLINE_TRAINING_EPISODES.csv',trainrows);writecsv(tables/'DEVELOPMENT_QUALIFIED_CASES.csv',evalrows)
    outputs += [p for p in [tables/'ONLINE_TRAINING_EPISODES.csv',tables/'DEVELOPMENT_QUALIFIED_CASES.csv'] if p.exists()]
    (a.output/'ITERATIVE_FIGURE_PROVENANCE.json').write_text(json.dumps({'scope':'Actual training and development only; no sealed-test access','source_script_sha256':SCRIPT_SHA256,'inputs':inputs,'outputs':{str(p.relative_to(a.output)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs},'moving_average_episodes':8,'training_episode_count':len(trainrows),'development_rows':len(evalrows)},indent=2))
    print(json.dumps({'training_episodes':len(trainrows),'development_case_rows':len(evalrows),'rounds':list(train)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
