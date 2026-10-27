"""Rebuild publication figures/tables from immutable synthetic PhysX evidence.

No benchmark result is typed into this script. Raw and independent results are
joined by case ID, and missing evidence aborts rather than becoming failure.
"""
import argparse, csv, gzip, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COLORS={'damped':'#566579','ppo':'#9a9fa7','residual':'#147d92','recover':'#d47a39'}
LABELS={'damped':'Damped feedback','ppo':'Direct PPO','residual':'Residual PPO','recover':'Residual + hold/proceed'}
SEEDS=[31,41,51]
SPLITS=['test_id','test_geometry','test_dynamics']
ROOT=Path(__file__).resolve().parents[2]
FIG=ROOT/'03_FIGURES'; TABLE=ROOT/'04_TABLES'
INPUTS={}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def csv_write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def box(ax,x,y,w,h,title,body,color='#eaf3f4',status='VERIFIED',dashed=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.014,rounding_size=.025',fc=color,ec='#526676',lw=1.1,linestyle='--' if dashed else '-'))
    ax.text(x+.022,y+h-.038,title,va='top',fontsize=11,fontweight='bold',color='#203544')
    ax.text(x+.022,y+h-(.085 if h<.20 else .105),body,va='top',fontsize=8.7 if h<.20 else 9.3,color='#394e5c',linespacing=1.4 if h<.20 else 1.6)
    ax.text(x+w-.015,y+.018,status,ha='right',va='bottom',fontsize=7.5,fontweight='bold',color='#92702c' if dashed else '#147d75')
def arrow(ax,a,b,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.3,color='#526676',linestyle='--' if dashed else '-'))

def diagrams():
    fig,ax=plt.subplots(figsize=(13.8,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,.97,'ARES: from structured tasks to verified physical outcomes',fontsize=18,fontweight='bold',color='#203544')
    box(ax,.03,.65,.27,.23,'Task and perception interfaces','Language / vision / point clouds\nStructured targets and scene snapshots',color='#f0f2f5',status='SYSTEM BACKGROUND')
    box(ax,.365,.65,.27,.23,'Skill Contract and Runtime','Goals, legal parameters, termination\nTask/Scheme, registry and validation',status='IMPLEMENTED SYSTEM')
    box(ax,.70,.65,.27,.23,'Paper research layer','Task-conditioned residual parameters\nOutcome-aware adaptation under test',color='#fff4df',status='PROTOTYPE / PROPOSED',dashed=True)
    box(ax,.03,.23,.27,.25,'Non-contact motion','cuRobo: IK, geometry and trajectories\nNot a contact-control safety certificate',status='SEPARATE PLANNING TRACK')
    box(ax,.365,.23,.27,.25,'Contact skill + controller','Bounded Cartesian actions to joints\nJAKA + free workpiece, P1 / P2',status='ACTUAL PHYSX')
    box(ax,.70,.23,.27,.25,'Execution evidence','Post-step states, task checks, failures\nPhysical rewards and PPO updates',status='VERIFIED RESEARCH PIPELINE')
    for a,b in [((.30,.765),(.365,.765)),((.635,.765),(.70,.765)),((.50,.65),(.50,.48)),((.76,.65),(.62,.48)),((.30,.355),(.365,.355)),((.635,.355),(.70,.355))]:arrow(ax,a,b)
    ax.text(.63,.60,'bounded parameters',fontsize=8,color='#526676')
    arrow(ax,(.93,.48),(.93,.65),True)
    ax.text(.715,.555,'measured outcome feedback',ha='left',fontsize=8.5,color='#92702c')
    ax.text(.03,.12,'Solid: implemented interfaces or verified synthetic experiments. Dashed: method increment still being tested.',fontsize=10,color='#526676')
    ax.text(.03,.075,'Not claimed: end-to-end multimodal Actor, real tray grasping, field deployment, or a validated new algorithm.',fontsize=10,color='#a35435')
    save(fig,'Fig1_system_layers')
    fig,ax=plt.subplots(figsize=(13.8,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,.97,'Task-conditioned residual skill: execution and learning graph',fontsize=18,fontweight='bold',color='#203544')
    box(ax,.025,.64,.23,.24,'Current observation','Robot state; workpiece pose / twist\nStage, goals and task geometry\n64D state-based pilot inputs',status='PROTOTYPE v7')
    box(ax,.33,.70,.27,.18,'Nominal feedback','Damped geometric contact control\nTask-frame version tested in pilot',status='IMPLEMENTED BASELINE')
    box(ax,.33,.43,.27,.18,'Residual Actor','XY correction + contact offset\nStandard PPO policy-gradient updates',status='IMPLEMENTED PILOT')
    box(ax,.70,.60,.27,.25,'Common constraint interface','Nominal + learned correction\nWorkspace, command and joint limits\nCartesian / DLS to joint targets',status='IMPLEMENTED')
    box(ax,.70,.20,.27,.24,'True physical transition','Continuous 120 Hz PhysX state\n60 Hz control, no object teleport\nIndependent post-step qualification',status='ACTUAL PHYSX')
    box(ax,.025,.20,.23,.24,'Result + command history','Prior response / angular motion\nPrior action + target command\nB3: observed; B2: zero history',color='#fff4df',status='IMPLEMENTED; GAIN UNPROVEN',dashed=True)
    for a,b in [((.255,.76),(.33,.79)),((.255,.70),(.33,.52)),((.60,.79),(.70,.77)),((.60,.52),(.70,.66)),((.835,.60),(.835,.44))]:arrow(ax,a,b)
    arrow(ax,(.70,.29),(.255,.29),True);arrow(ax,(.255,.38),(.33,.46),True)
    ax.text(.46,.32,'physical reward + GAE / PPO',ha='center',fontsize=9.5,color='#147d75')
    arrow(ax,(.70,.40),(.60,.48))
    ax.text(.025,.095,'No learned successor-Q head or high-level switch. History enters the Actor; its independent benefit still requires paired ablation.',fontsize=10,color='#a35435')
    save(fig,'Fig2_method_graph')

def historical(evidence):
    cases=[];counts=[]
    for split in SPLITS:
        for method in ['damped','ppo','residual','recover']:
            for seed in ([None] if method=='damped' else SEEDS):
                name=f'v6b_{split}_damped' if method=='damped' else f'v6b_{split}_v6b_{method}_seed{seed}'
                d=evidence/name
                raw=load(d/'frozen_evaluation.json.gz');audit=load(d/'INDEPENDENT_TASK_AND_CONTACT.json.gz')
                byid={x['case_id']:x for x in audit['cases']}
                assert len(raw)==32 and len(byid)==32
                for r in raw:
                    a=byid[r['case_id']]
                    cases.append({'split':split,'method':method,'seed':seed or 'NA','case_id':r['case_id'],'family':'P1' if r['family']=='push' else 'P2','raw_success':int(r['success']),'independent_success':int(a['independent_task_pass']),'steps':r['steps'],'source_run':name})
                for family in ['All','P1','P2']:
                    z=[x for x in cases if x['source_run']==name and (family=='All' or x['family']==family)]
                    counts.append({'split':split,'method':method,'seed':seed or 'NA','family':family,'independent_successes':sum(x['independent_success'] for x in z),'n':len(z),'raw_successes':sum(x['raw_success'] for x in z),'source_run':name})
    csv_write(FIG/'data/historical_cases.csv',cases);csv_write(FIG/'data/historical_counts.csv',counts)
    paired=[];summaries=[]
    for split in SPLITS:
        for family in ['All','P1','P2']:
            for m in ['damped','ppo','residual','recover']:
                z=[c for c in counts if c['split']==split and c['method']==m and c['family']==family]
                vals=[c['independent_successes']/c['n'] for c in z]
                summaries.append({'split':split,'family':family,'method':m,'training_seeds':len(z) if m!='damped' else 0,'n_cases_per_seed':z[0]['n'],'success_rate_seed_mean':np.mean(vals),'success_rate_seed_sample_sd':np.std(vals,ddof=1) if len(vals)>1 else None,'seed_min':min(vals),'seed_max':max(vals),'uncertainty':'Seed SD/range, NOT confidence interval or independent scene count'})
            for a,b in [('residual','damped'),('recover','residual')]:
                for s in SEEDS:
                    left={c['case_id']:c['independent_success'] for c in cases if c['split']==split and c['method']==a and c['seed']==s and (family=='All' or c['family']==family)}
                    right={c['case_id']:c['independent_success'] for c in cases if c['split']==split and c['method']==b and c['seed']==('NA' if b=='damped' else s) and (family=='All' or c['family']==family)}
                    assert left.keys()==right.keys()
                    delta=np.array([left[k]-right[k] for k in sorted(left)])
                    rng=np.random.default_rng(20261027);boots=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
                    paired.append({'split':split,'family':family,'method_a':a,'method_b':b,'seed':s,'paired_cases':len(left),'a_only_success':int((delta==1).sum()),'b_only_success':int((delta==-1).sum()),'paired_success_rate_difference':delta.mean(),'case_bootstrap_lower_95':np.quantile(boots,.025),'case_bootstrap_upper_95':np.quantile(boots,.975),'uncertainty':'Within-fixed-cases resampling only; not training-seed/population CI; exploratory historical analysis'})
    csv_write(FIG/'data/historical_seed_summary.csv',summaries);csv_write(FIG/'data/historical_paired_differences.csv',paired)
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.7),sharey=True)
    methods=['damped','ppo','residual','recover'];titles=['Test-ID','Test-Geometry','Test-Dynamics']
    for row,family in enumerate(['P1','P2']):
        for ax,split,title in zip(axes[row],SPLITS,titles):
            for k,m in enumerate(methods):
                z=[c for c in counts if c['split']==split and c['method']==m and c['family']==family]
                vals=np.array([c['independent_successes']/c['n']*100 for c in z])
                ax.bar(k,vals.mean(),color=COLORS[m],alpha=.20,width=.65)
                ax.scatter(k+np.linspace(-.10,.10,len(vals)),vals,color=COLORS[m],s=40,marker='D' if m=='damped' else 'o',zorder=5)
                ax.text(k,min(vals.mean()+9,108),'/'.join(str(c['independent_successes']) for c in z),ha='center',fontsize=8,color=COLORS[m])
            ax.set(xticks=range(4),xticklabels=['Feedback','PPO','Residual','+ Hold'],ylim=(0,112),title=f'{title} | {family} (16 cases)')
            ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
            if ax in axes[:,0]:ax.set_ylabel('Independent task success (%)')
    fig.suptitle('Historical v6b: successes separated by physical task',fontsize=16,fontweight='bold',x=.51,y=1.01)
    fig.text(.01,-.012,'Dots: training seeds 31 / 41 / 51; diamond: non-learning feedback. Numbers are successes / 16. Same cases are paired, not independent replicates.\nPreviously inspected historical tests; no tuning or final-test claim for the new pilot. Synthetic JAKA PhysX tasks, not field or an official benchmark.',fontsize=9,color='#526676')
    fig.tight_layout();save(fig,'Fig5_historical_frozen_tests')
    rows=[]
    for method in ['ppo','residual','recover']:
        for seed in SEEDS:
            d=evidence/f'v6b_{method}_seed{seed}'
            tr=load(d/'training_episodes.json.gz');cfg=load(d/'manifest.json')
            for b in range(8192,65537,8192):
                z=[r for r in tr if b-8192<r['env_interactions']<=b]
                rows.append({'method':method,'seed':seed,'interaction_bin_end':b,'completed_episodes':len(z),'training_successes':sum(r['success'] for r in z),'training_success_rate':np.mean([r['success'] for r in z]) if z else None,'mean_episode_return':np.mean([r['return'] for r in z]) if z else None,'training_source_sha256':cfg['task_source_sha256']})
    csv_write(FIG/'data/historical_learning.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.5))
    for m in ['ppo','residual','recover']:
        ys=[]
        for s in SEEDS:
            z=[r for r in rows if r['method']==m and r['seed']==s];x=np.array([r['interaction_bin_end'] for r in z])/1000;y=np.array([r['training_success_rate'] for r in z],float)*100;ys.append(y)
            axes[0].plot(x,y,color=COLORS[m],alpha=.28,lw=1)
        ys=np.array(ys);present=np.isfinite(ys).any(axis=0)
        mean=np.full(len(x),np.nan);low=mean.copy();high=mean.copy()
        mean[present]=np.nanmean(ys[:,present],axis=0);low[present]=np.nanmin(ys[:,present],axis=0);high[present]=np.nanmax(ys[:,present],axis=0)
        axes[0].plot(x,mean,color=COLORS[m],lw=2.5,label=LABELS[m]);axes[0].fill_between(x,low,high,color=COLORS[m],alpha=.09)
        for s in SEEDS:
            ev=[]
            for k in [32768,65536]:
                name=f'v6b_validation_v6b_{m}_seed{s}_32k' if k==32768 else f'v6b_{m}_seed{s}_validation'
                p=evidence/name/'INDEPENDENT_TASK_AND_CONTACT.json.gz'
                if not p.exists():continue
                a=load(p);z=[c for c in a['cases'] if c['case_id']<32]
                ev.append((k/1000,sum(c['independent_task_pass'] for c in z)/len(z)*100))
            if ev:axes[1].plot(*zip(*ev),marker='o',color=COLORS[m],alpha=.55,label=LABELS[m] if s==31 else None)
    for ax in axes:
        ax.set(xlabel='Actual online environment interactions (thousands)',ylabel='Task success (%)',ylim=(-2,102));ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('Exploratory training episodes (8,192-step bins)');axes[1].set_title('Independent deterministic validation (same 32 cases)')
    axes[0].legend(frameon=False,fontsize=9,loc='lower right');axes[1].legend(frameon=False,fontsize=9,loc='lower right')
    fig.suptitle('Historical online learning: train and evaluation are not interchangeable',fontsize=14,fontweight='bold',y=1.03)
    fig.text(.01,-.03,'Thin lines: individual training seeds. Thick curves: seed mean; pale envelope: observed seed range, NOT a confidence interval.\nThe fixed 65,536-interaction checkpoint was evaluated; no best-checkpoint selection. Direct-PPO 32k validation is not available.',fontsize=9,color='#526676')
    fig.tight_layout();save(fig,'Fig4_historical_online_learning')
    tables(counts)
    return counts

def tables(counts):
    TABLE.mkdir(parents=True,exist_ok=True)
    def cell(split,m,f):
        z=[r for r in counts if r['split']==split and r['method']==m and r['family']==f]
        return ', '.join(str(r['independent_successes']) for r in z)+' / '+str(z[0]['n'])
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Method & Task & ID & Geometry & Dynamics \\',r'\midrule']
    for m in ['damped','ppo','residual','recover']:
        for f in ['All','P1','P2']:lines.append(f"{LABELS[m] if f=='All' else ''} & {f} & "+' & '.join(cell(s,m,f) for s in SPLITS)+r' \\')
        lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (TABLE/'Tab2_historical_results.tex').write_text('\n'.join(lines)+'\n')
    env=[{'component':'Physics / control','value':'Isaac Sim 5.1; Isaac Lab 2.3.0; 120 / 60 Hz','status':'Verified synthetic simulation'}, {'component':'P1 / P2','value':'Free-workpiece two-stage push / deep guided slot','status':'Self-built; not official benchmark'}, {'component':'Robot / object','value':'Single JAKA; 40 mm cube; 30 mm research paddle','status':'No real tray grasping claim'}, {'component':'Actor / policy','value':'56D state-based observation; MLP 64--64','status':'Not vision deployment'}, {'component':'Task acceptance','value':'Post-step state and 30 controls stable completion','status':'Independent task/geometry checks'}, {'component':'Formal cohort','value':'3 methods x 3 seeds x 65,536 interactions','status':'Historical formal training'}, {'component':'Full historical cohort','value':'901,120 interactions; 16 trained runs including diagnostics','status':'Not 16 formal seeds'}]
    csv_write(TABLE/'Tab1_environment.csv',env)
    lines=[r'\begin{tabular}{lp{5.5cm}p{4.3cm}}',r'\toprule',r'Component & Specification & Evidence boundary \\',r'\midrule']+[f"{r['component']} & {r['value']} & {r['status']} "+r'\\' for r in env]+[r'\bottomrule',r'\end{tabular}']
    (TABLE/'Tab1_environment.tex').write_text('\n'.join(lines)+'\n')
    csv_write(TABLE/'Tab2_historical_results.csv',counts)

def physical_diagnostic(evidence):
    """Predeclared historical examples, not model selection or causal proof."""
    d=evidence/'v6b_residual_seed31_validation';p=d/'actual_physics_trace.npz'
    if not p.exists():return
    INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
    records=load(d/'frozen_evaluation.json.gz');byid={r['case_id']:r for r in records}
    a=np.load(p);states=a['states'];cols={str(n):i for i,n in enumerate(a['columns'])}
    rows=[];fig,axes=plt.subplots(2,2,figsize=(11.3,6.7))
    for case,color in [(1,'#b55e45'),(3,'#147d92')]:
        r=byid[case];x=states[states[:,:,cols['case_id']]==case];assert len(x)>10
        t=np.arange(len(x))*float(a['sample_dt_s']);pos=x[:,[cols[n] for n in ['obj_x','obj_y','obj_z']]]
        goal=np.array(r['post_step_state']['goal']);dist=np.linalg.norm(pos-goal,axis=1)*1000
        vel=x[:,[cols[n] for n in ['vx','vy','vz']]];speed=np.linalg.norm(vel,axis=1)*1000
        w=x[:,[cols[n] for n in ['wx','wy','wz']]];omega=np.linalg.norm(w,axis=1)
        q=x[:,[cols[n] for n in ['obj_qw','obj_qx','obj_qy','obj_qz']]];yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
        label=f"Case {case}: {'accepted' if r['success'] else 'failed'}"
        for ax,y in zip(axes.flat,[dist,speed,omega,np.degrees(yaw)]):ax.plot(t,y,color=color,lw=1.7,label=label)
        for i in range(0,len(x),4):rows.append({'case_id':case,'time_s':t[i],'phase':int(x[i,cols['stage']]),'goal_distance_mm':dist[i],'speed_mm_s':speed[i],'angular_speed_rad_s':omega[i],'object_yaw_deg':np.degrees(yaw[i]),'raw_task_success':r['success']})
    for ax,ylabel in zip(axes.flat,['Final-target distance (mm)','Linear speed (mm/s)','Angular speed (rad/s)','Workpiece yaw (deg)']):
        ax.set(xlabel='Physical time since recorded case start (s)',ylabel=ylabel);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0,0].axhline(8,color='#8c8c8c',ls=':',lw=1);axes[0,1].axhline(10,color='#8c8c8c',ls=':',lw=1);axes[1,0].axhline(.2,color='#8c8c8c',ls=':',lw=1)
    axes[0,0].legend(frameon=False);fig.suptitle('Historical contact-state diagnostic: actual successful and failed suffixes',fontsize=14,fontweight='bold')
    fig.text(.01,-.02,'Explicit diagnostic cases 1 and 3 from residual seed31 validation. Recorded 120 Hz PhysX states; CSV decimated 4x for readability.\nThese are different cases, not a controlled velocity/rotation intervention; no causal or population claim.',fontsize=9,color='#526676');fig.tight_layout()
    save(fig,'FigS1_actual_contact_state_diagnostic');csv_write(FIG/'data/historical_contact_state_examples.csv',rows)

def pilot(path):
    """Actual-run interface: CSV rows only, no fabricated bars.

    columns: method,seed,split,family,n,raw_successes,independent_successes,
    interactions,run,source_sha256,audit_sha256. independent_successes may be
    blank ONLY when audit is incomplete; that row is displayed as Not tested.
    """
    if path and path.exists():
        with path.open() as f:allrows=list(csv.DictReader(f))
        INPUTS[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
        rows=[r for r in allrows if r['split']=='validation']
        if allrows:csv_write(FIG/'data/new_pilot_all_completed_runs.csv',allrows)
    else:rows=[]
    indexpath=path.with_name('NEW_RUNS_INDEX.json') if path else None
    runindex=load(indexpath)['runs'] if indexpath and indexpath.exists() else []
    (TABLE/'PILOT_INGESTION.md').write_text(pilot.__doc__+'\nNo row is extrapolated to missing seeds. Use --pilot-csv after completed runs and audits.\n')
    specs=[('Tab3_new_pilot.tex',['Method','Seed','Task','Independent success','Model train steps'])]
    for name,header in specs:
        ls=[r'\begin{tabular}{llllr}',r'\toprule',' & '.join(header)+r' \\',r'\midrule']
        if not rows:ls.append(r'New pilot & -- & P2 & Not yet tested & TBD \\')
        for r in rows:
            v=f"{r['independent_successes']}/{r['n']}" if r.get('independent_successes') else 'Not tested'
            label=r['method']
            pretty={'B0_withdraw':'Feedback + backoff','B3_measured_response_intervention':'B3 measured-history mask (eval)','B3_no_next_intervention':'B3 next-goal mask (eval)','B3_no_outcome_intervention':'B3 history mask (eval)','B3_no_next':'B3 no next goal (train)','B4_parameter_+1.0':'Fixed contact +4 mm','B4_parameter_-1.0':'Fixed contact -4 mm'}
            label=pretty.get(label,label)
            cols=[label,r['seed'],r['family'],v,r['interactions']]
            ls.append(' & '.join(c.replace('_',r'\_') for c in cols)+r' \\')
        ls.extend([r'\bottomrule',r'\end{tabular}']);(TABLE/name).write_text('\n'.join(ls)+'\n')
    ls=[r'\begin{tabular}{lllllr}',r'\toprule',r'Variant & Frame & Result/command history & Next goal & Seeds & Train budget \\',r'\midrule']
    for method in sorted({r['method'] for r in rows}):
        z=[r for r in rows if r['method']==method];frame='Body' if method=='B1' else 'Task'
        history='Observed' if method.startswith('B3') else 'Zero / none';nextgoal='Observed';budget=','.join(sorted({r['interactions'] for r in z}));label=method
        is_intervention='intervention' in method
        if 'no_outcome' in method:history='Suppressed at eval' if is_intervention else 'Zero';label='B3 no history (eval)' if is_intervention else 'B3 no history (train)'
        if 'no_next' in method:nextgoal='Suppressed at eval' if is_intervention else 'Hidden during training';label='B3 no next goal (eval)' if is_intervention else 'B3 no next goal (train)'
        if is_intervention and ('measured' in method or 'response' in method):history='Command memory only';label='B3 no measured effect (eval)'
        if method=='B0_withdraw':label='Feedback + backoff'
        if method=='B4_parameter_+1.0':label='Fixed contact +4 mm'
        if method=='B4_parameter_-1.0':label='Fixed contact -4 mm'
        if is_intervention:budget='Same checkpoint'
        cols=[label,frame,history,nextgoal,','.join(r['seed'] for r in z),budget]
        ls.append(' & '.join(c.replace('_',r'\_') for c in cols)+r' \\')
    if not any('no_next' in r['method'] for r in rows):ls.append(r'No next-contract & Task & Not tested & -- & -- & -- \\')
    ls.extend([r'\bottomrule',r'\end{tabular}']);(TABLE/'Tab4_ablation_budget.tex').write_text('\n'.join(ls)+'\n')
    if rows:
        csv_write(FIG/'data/new_pilot_results.csv',rows)
        good=[r for r in rows if r.get('independent_successes') and r['family']=='P2' and r['method'] in ['B0','B1','B2','B3']]
        if good:
            fig,axes=plt.subplots(1,2,figsize=(12.4,4.9),gridspec_kw={'width_ratios':[1.05,1]});ax=axes[0];methods=sorted({r['method'] for r in good})
            for i,m in enumerate(methods):
                z=sorted([r for r in good if r['method']==m],key=lambda r:int(r['seed']));y=[100*int(r['independent_successes'])/int(r['n']) for r in z]
                if m=='B0':ax.scatter([i],y,s=50,color='#566579',marker='D')
                else:ax.scatter(i+np.linspace(-.08,.08,len(y)),y,s=45,color=['#147d92','#d47a39','#8b64a8'][:len(y)])
                ax.plot([i-.18,i+.18],[np.mean(y)]*2,color='#203544',lw=2)
                if len(y)>1:ax.errorbar(i,np.mean(y),yerr=np.std(y,ddof=1),color='#203544',capsize=4,lw=1.1)
                ax.text(i,81,' / '.join(r['independent_successes'] for r in z),ha='center',fontsize=8,color='#526676')
            ax.set(xticks=range(len(methods)),xticklabels=methods,ylabel='Independent P2 success (%)',ylim=(0,105),title='A  Fixed cases: trained policies and feedback');ax.grid(axis='y',alpha=.2)
            ax.text(.02,.035,'B0: feedback; B1: body frame; B2: task frame; B3: history',fontsize=7.5,transform=ax.transAxes,color='#526676')
            interventions=[r for r in rows if 'intervention' in r['method'] and r.get('independent_successes')]
            full=next((r for r in good if r['method']=='B3' and r['seed']=='31'),None)
            ablation=([full] if full else [])+interventions
            ax=axes[1]
            if ablation:
                for i,r in enumerate(ablation):
                    value=100*int(r['independent_successes'])/int(r['n']);ax.scatter(i,value,color='#147d92',s=45);ax.text(i,max(5,value-12),f"{r['independent_successes']}/{r['n']}",ha='center',fontsize=9)
                labels=[]
                for r in ablation:
                    m=r['method'];labels.append('Full B3' if m=='B3' else ('No history\n(8 channels)' if 'no_outcome' in m else 'No next goal\n(partial)' if 'no_next' in m else 'No measured\nhistory' if 'response' in m else m.replace('B3_','').replace('_intervention','')))
                ax.set(xticks=range(len(ablation)),xticklabels=labels,ylabel='Independent P2 success (%)',ylim=(0,105),title='B  Same checkpoint: inference input interventions')
            else:ax.text(.5,.5,'No completed inference intervention',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
            for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
            fig.suptitle('New P2 pilot: actual results, not an established method advantage',fontsize=14,fontweight='bold')
            fig.text(.01,-.055,'Development validation only. Seed dots + mean +/- sample SD (NOT confidence interval); counts are successful paired cases.\nB0 has no training seed. Input interventions reuse one seed31 checkpoint, not new trained seeds; current-state feedback remains.',fontsize=9)
            fig.tight_layout();save(fig,'Fig6_new_pilot')
            summaries=[]
            for method in methods:
                z=[r for r in good if r['method']==method];v=np.array([int(r['independent_successes'])/int(r['n']) for r in z]);summaries.append({'method':method,'training_seed_replicates':0 if method=='B0' else len(z),'validation_cases_per_model':','.join(sorted({r['n'] for r in z})),'mean_success_rate':v.mean(),'sample_seed_sd':v.std(ddof=1) if len(v)>1 else None,'model_training_budget_per_seed':','.join(sorted({r['interactions'] for r in z})),'note':'Validation rows do not add training interactions; SD is not CI'})
            csv_write(FIG/'data/new_pilot_seed_summary.csv',summaries)
    if runindex:
        eligible=[r for r in runindex if r['split']=='validation' and 'independent_success_case_ids' in r]
        pairs=[]
        for left in eligible:
            m=left['method'];seed=left['seed']
            rightmethod='B1' if m=='B2' else 'B2' if m=='B3' else 'B3' if 'intervention' in m else None
            if rightmethod is None:continue
            right=next((r for r in eligible if r['method']==rightmethod and r['seed']==seed),None)
            if not right:continue
            assert left['n']==right['n'];l=set(left['independent_success_case_ids']);r=set(right['independent_success_case_ids'])
            pairs.append({'method_a':m,'method_b':rightmethod,'seed':seed,'n':left['n'],'a_successes':len(l),'b_successes':len(r),'a_only_success':len(l-r),'b_only_success':len(r-l),'same_success_set':l==r,'a_only_case_ids':json.dumps(sorted(l-r)),'b_only_case_ids':json.dumps(sorted(r-l))})
        if pairs:csv_write(FIG/'data/new_pilot_paired_case_sets.csv',pairs)
    physicalpath=path.with_name('PHYSICAL_CAUSAL_INTERVENTIONS.json') if path else None
    if physicalpath and physicalpath.exists():
        physical=load(physicalpath);pr=[]
        for x in physical['comparisons']:
            pr.append({'comparison':x['label'],'left':x['left'],'right':x['right'],'matched_cases':x['n'],'left_successes':x['left_successes'],'right_successes':x['right_successes'],'rescued_cases':len(x['rescued_case_ids']),'lost_cases':len(x['lost_case_ids']),'initial_max_difference':x['initial_max_difference'],'mean_object_xyz_rms_difference_um':x['mean_object_xyz_rms_difference_m']*1e6,'mean_applied_reference_delta_rms_difference_um':x['mean_applied_reference_delta_rms_difference_m']*1e6,'caution':'Full closed-loop trajectories differ; not isolated force/velocity causation or calibrated significance'})
        if pr:csv_write(FIG/'data/new_pilot_physical_interventions.csv',pr)
    learningpath=path.with_name('ONLINE_TRAINING_EPISODES.csv') if path else None
    if learningpath and learningpath.exists():
        INPUTS[str(learningpath)]={'sha256':sha(learningpath),'bytes':learningpath.stat().st_size}
        with learningpath.open() as f:episodes=list(csv.DictReader(f))
        bins=[]
        for m in ['B1','B2','B3']:
            for s in SEEDS:
                ep=[r for r in episodes if r['method']==m and int(r['seed'])==s]
                if not ep:continue
                for end in [4096,8192,12288,16384]:
                    z=[r for r in ep if end-4096<int(r['interactions'])<=end]
                    bins.append({'method':m,'seed':s,'interactions_bin_end':end,'completed_episodes':len(z),'success_rate':np.mean([int(r['success']) for r in z]) if z else None,'mean_episode_return':np.mean([float(r['reward']) for r in z]) if z else None})
        if bins:
            csv_write(FIG/'data/new_pilot_online_learning.csv',bins);fig,axes=plt.subplots(1,2,figsize=(11.8,5.4));colors={'B1':'#8b64a8','B2':'#147d92','B3':'#d47a39'}
            for m in ['B1','B2','B3']:
                for s in SEEDS:
                    z=[r for r in bins if r['method']==m and r['seed']==s]
                    if not z:continue
                    x=[r['interactions_bin_end']/1000 for r in z]
                    for ax,key in zip(axes,['success_rate','mean_episode_return']):
                        y=[np.nan if r[key] is None else r[key]*(100 if key=='success_rate' else 1) for r in z]
                        ax.plot(x,y,marker={31:'o',41:'s',51:'^'}[s],color=colors[m],alpha=.65,lw=1.4,label=f'{m}, seed{s}')
            for ax in axes:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Actual online environment interactions (thousands)');ax.set(xlim=(0,17),xticks=[0,4,8,12,16])
            axes[0].set(ylabel='Training episode success (%)',ylim=(-2,102));axes[1].set_ylabel('Observed episode return')
            fig.suptitle('New P2 online pilot: observed training episodes, not frozen-test curves',fontsize=13,fontweight='bold')
            h,l=axes[1].get_legend_handles_labels();fig.legend(h,l,frameon=False,fontsize=7.5,ncol=3,loc='lower center',bbox_to_anchor=(.53,.085))
            fig.text(.01,.005,'Completed episodes binned by 4,096 actual interactions; empty bins remain gaps. Lines are individual seeds, not confidence intervals.\nA short pilot and high nominal-control success do not establish convergence or a method advantage.',fontsize=9,color='#526676');fig.tight_layout(rect=(0,.23,1,.95));save(fig,'FigS2_new_pilot_online_learning')

def new_physical_examples(pilotcsv):
    if pilotcsv is None or not pilotcsv.exists():return
    with pilotcsv.open() as f:rows=list(csv.DictReader(f))
    selected={}
    for name,method in [('full','B3'),('masked','B3_no_outcome_intervention')]:
        row=next((r for r in rows if r['method']==method and r['seed']=='31' and r['split']=='validation' and r.get('independent_successes')),None)
        if row is None:return
        d=Path(row['run']);d=d if d.is_absolute() else ROOT.parents[5]/d
        assert (d/'RUN_COMPLETE.json').exists()
        p=d/'actual_physics_trace.npz';INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
        a=np.load(p);m=load(d/'manifest.json');recs=load(d/'frozen_evaluation.json');audit=load(d/'INDEPENDENT_TASK_AND_CONTACT.json')
        byid={r['case_id']:r for r in recs};verified={r['case_id']:r['independent_task_pass'] for r in audit['cases']}
        assert verified[0] is True and verified[4] is False
        selected[name]={'a':a,'m':m,'recs':byid,'run':str(d),'cols':{str(n):i for i,n in enumerate(a['columns'])}}
    fig,axes=plt.subplots(2,2,figsize=(12,8));state_rows=[];deltas=[];series={}
    for case,color in [(0,'#147d92'),(4,'#b55e45')]:
        for mode,ls in [('full','-'),('masked','--')]:
            z=selected[mode];a=z['a'];col=z['cols'];x=a['states'][a['states'][:,:,col['case_id']]==case];r=z['recs'][case]
            phi=z['m']['scene_specs'][r['env']]['direction'];fw=np.array([np.cos(phi),np.sin(phi)]);lat=np.array([-np.sin(phi),np.cos(phi)])
            p=x[:,[col[n] for n in ['obj_x','obj_y','obj_z']]];pad=x[:,[col[n] for n in ['paddle_x','paddle_y','paddle_z']]]
            goal=np.array(r['post_step_state']['goal']);relative=p[:,:2]-goal[:2];forward=relative@fw*1000;lateral=relative@lat*1000
            t=(x[:,col['physics_tick']]-x[0,col['physics_tick']])*float(a['sample_dt_s']);omega=np.linalg.norm(x[:,[col[n] for n in ['wx','wy','wz']]],axis=1)
            separation=(pad[:,:2]-p[:,:2])@lat*1000
            label=f"Case {case} ({'accepted' if case==0 else 'timeout'}), {'full' if mode=='full' else 'history masked'}"
            axes[0,0].plot(forward,lateral,ls,color=color,lw=1.7,label=label);axes[0,1].plot(t,omega,ls,color=color,lw=1.5);axes[1,0].plot(t,separation,ls,color=color,lw=1.5)
            series[(case,mode)]=(t,p)
            for i in range(0,len(x),4):state_rows.append({'case_id':case,'mode':mode,'elapsed_since_first_sample_s':float(t[i]),'phase':int(x[i,col['stage']]),'workpiece_forward_to_final_goal_mm':float(forward[i]),'workpiece_lateral_to_final_goal_mm':float(lateral[i]),'workpiece_angular_speed_rad_s':float(omega[i]),'paddle_workpiece_lateral_separation_mm':float(separation[i]),'object_x_m':float(p[i,0]),'object_y_m':float(p[i,1]),'object_z_m':float(p[i,2]),'independent_episode_success':case==0,'source_run':z['run']})
        ta,pa=series[(case,'full')];tb,pb=series[(case,'masked')];n=min(len(ta),len(tb));assert np.allclose(ta[:n],tb[:n],atol=1e-5)
        delta=np.linalg.norm(pa[:n]-pb[:n],axis=1)*1e6;axes[1,1].plot(ta[:n],delta,color=color,lw=1.5,label=f'Case {case}')
        for i in range(0,n,4):deltas.append({'case_id':case,'matched_elapsed_s':float(ta[i]),'full_vs_masked_object_xyz_distance_um':float(delta[i]),'comparison':'Same cold start, overlapping elapsed physics samples; no time warping'})
    axes[0,0].scatter([0],[0],marker='*',color='#203544',s=90,zorder=5);axes[0,0].annotate('Final goal',(0,0),xytext=(-30,-18),textcoords='offset points',fontsize=8,color='#203544')
    axes[0,0].set(xlabel='Workpiece forward error to final goal (mm)',ylabel='Lateral error to final goal (mm)',title='A  Actual workpiece paths in task coordinates')
    axes[0,0].legend(frameon=False,fontsize=8,loc='best')
    axes[0,1].set(xlabel='Elapsed physical time (s)',ylabel='Workpiece angular speed (rad/s)',title='B  Actual angular motion')
    axes[1,0].set(xlabel='Elapsed physical time (s)',ylabel='Paddle-object lateral separation (mm)',title='C  Geometric contact-offset diagnostic, not force')
    axes[1,1].set(xlabel='Matched elapsed physical time (s)',ylabel='Full vs masked object distance (micrometers)',title='D  Small closed-loop trajectory differences')
    axes[1,1].legend(frameon=False,fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('New P2 physical diagnostics: real paths and same-policy history intervention',fontsize=14,fontweight='bold')
    fig.text(.01,.005,'Prespecified diagnostic cases 0 (accepted) and 4 (timeout); independent checks verified before plotting. Solid: full B3; dashed: 8D history masked.\n120 Hz PhysX, continuous execution, no partial contact-state restoration. Equal success sets; no isolated velocity/rotation causation or method advantage.\nThe goal star alone is not the full task-success criterion. Full traces stay non-Git; exported states are decimated 4x.',fontsize=8.5,color='#526676')
    fig.tight_layout(rect=(0,.09,1,.96));save(fig,'FigS3_new_pilot_physical_trajectories')
    csv_write(FIG/'data/new_pilot_physical_example_states.csv',state_rows);csv_write(FIG/'data/new_pilot_physical_example_paired_deltas.csv',deltas)

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=ROOT.parents[1]/'results/contact_handoff_20261016_r1');p.add_argument('--pilot-csv',type=Path);a=p.parse_args()
    FIG.mkdir(parents=True,exist_ok=True);(FIG/'provenance').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    diagrams();historical(a.evidence);physical_diagnostic(a.evidence);pilot(a.pilot_csv);new_physical_examples(a.pilot_csv)
    fig3file=FIG/'provenance/FIG3_PROVENANCE.json'
    supplementary={}
    if fig3file.exists():
        fig3=load(fig3file)
        for entry in fig3['images']:
            p=FIG/entry['file'];assert sha(p)==entry['sha256']==entry['raw_sha256']
            assert entry['appearance_identity']['equal']
        supplementary['Fig3']={'provenance_file':str(fig3file.relative_to(ROOT)),'sha256':sha(fig3file),'scope':fig3['scope'],'live_evaluation_not_training':True,'not_terminal_photographs':True,'physical_attributes_identity_verified':True,'source_script_sha256':fig3['script_sha256']}
        capture=Path(fig3['images'][0]['raw_source']).parents[1]
        audit=load(capture/'INDEPENDENT_TASK_AND_CONTACT.json');raw=load(capture/'frozen_evaluation.json');run=load(capture/'manifest.json')
        assert run['online_training_interactions']==0
        supplementary['Fig3'].update(independent_successes=audit['independent_successes'],evaluation_cases=len(raw),new_training_interactions=0)
    captions={'Fig1_system_layers':'Layered ARES research context. Implemented industrial interfaces are distinguished from synthetic PhysX evidence and proposed method increments. This is not an end-to-end multimodal-policy or real-robot claim.', 'Fig2_method_graph':'Actual v7 pilot execution graph, not a validated algorithm advantage. A 64D state-based Actor controls bounded XY residuals and a phase-0 contact-offset parameter. B3 directly receives previous observed response/angular motion plus previous action/contact-target commands; B2 has matched zero history channels. The contact-target command is not measured executed displacement; execution must be checked using applied reference and object traces. Standard PPO updates from physical reward. No learned successor-Q head or high-level switch is implemented. The dashed component marks unproven independent benefit, not an absent code path.', 'Fig4_historical_online_learning':'Historical v6b actual online training. Training bins use completed episodes, not independent test trials. Evaluation uses matched case IDs below 32; source CSVs and hashes identify all runs. Seed envelope is a range, not confidence interval.', 'Fig5_historical_frozen_tests':'Independent post-step PhysX task acceptance on previously inspected v6b frozen cases. P1 and P2 are separated to avoid masking slot performance. Each task has 16 paired cases; three learned-policy seeds do not create 48 independent scene groups.', 'Fig6_new_pilot':'New pilot actual completed audited runs only. Development split, bounded training budget, and no imputation of missing methods or seeds. See CSV/source SHA and experimental protocol.'}
    captions['Fig6_new_pilot']+=' Seed dots and error bars show mean plus/minus sample SD, not confidence intervals. B0 is a non-learning controller and has no training seed. Inference masks reuse one seed31 checkpoint; the shared current-state observations remain available, and the partial next-goal mask does not remove the known task direction from the controller. Model training steps are reported per checkpoint and are not summed again for each evaluation row.'
    (FIG/'caption_en.md').write_text('\n\n'.join(f'## {k}\n\n{v}' for k,v in captions.items())+'\n')
    with (FIG/'caption_en.md').open('a') as f:
        f.write('\n## FigS1_actual_contact_state_diagnostic\n\nHistorical validation cases 1 (failed) and 3 (accepted) from residual seed31, chosen as explicitly identified diagnostic examples rather than an effect estimate. The actual 120 Hz PhysX states show remaining target error, linear/angular velocity and yaw. Different initial cases are not a causal intervention or a population-level comparison.\n')
        if (FIG/'FigS2_new_pilot_online_learning.pdf').exists():f.write('\n## FigS2_new_pilot_online_learning\n\nActual newly completed P2 training episodes in the corrected formal v7 pilot, binned by 4,096 online environment interactions. Empty bins remain gaps because completed episodes do not necessarily occur in every rollout. Individual seed lines show episode success and return, not a frozen-test learning curve or an uncertainty band. The short 16,384-interaction runs and high nominal-controller performance do not establish convergence or an algorithmic advantage. Pre-correction diagnostic runs outside formal/ are excluded and separately charged to the total research budget.\n')
        if (FIG/'FigS3_new_pilot_physical_trajectories.pdf').exists():f.write('\n## FigS3_new_pilot_physical_trajectories\n\nNew P2 actual 120 Hz PhysX traces for prespecified diagnostic cases 0 (independently accepted) and 4 (timeout), comparing the same trained B3 seed31 checkpoint with its 8D history inputs masked. Task-frame paths, workpiece angular velocity, geometric paddle/workpiece lateral separation, and time-aligned object trajectory differences are computed from recorded states. The geometric offset is not a measured contact-force signal. Both variants have the same successful case set. Closed-loop differences combine dynamics, policy feedback and numerical sensitivity; they are not an isolated velocity/rotation intervention or evidence of superior success. Raw traces remain non-Git; a 4x-decimated CSV and source hashes are provided.\n')
        if supplementary:
            v=supplementary['Fig3']
            f.write(f"\n## Fig3_physx_scene_board\n\nLive Isaac Sim 5.1 / Isaac Lab PhysX reevaluation of the historical residual seed31 checkpoint on synthetic self-built tasks, not new v7 training or field reconstruction. Panels show case 3 at physics step 800 (eventually accepted), case 1 at step 1184 (eventually timed out), and 16 independent environments at step 800. These are nonterminal frames, not terminal success-state photographs. The rerun independently accepted {v['independent_successes']}/{v['evaluation_cases']} episodes. Raw 1920x1200 frames are byte-identical copies, arranged with external vector labels only. Camera, visual materials, lighting and antialiasing changes preserve the physical/geometric signature across all 8,336 inspected attributes. See FIG3_PROVENANCE.json for exact task, asset, checkpoint, appearance and pixel hashes.\n")
    manifest={'date':'2026-10-27','scope':'SYNTHETIC_SELF_BUILT_PHYSX_ONLY','script_sha256':sha(Path(__file__)),'inputs':INPUTS,'outputs':{str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [*FIG.glob('*'),*FIG.glob('data/*'),*TABLE.glob('*')] if p.is_file()},'supplementary_provenance':supplementary,'historical_release_commit':'79448a47ffda5c8c1887f53e33a5dbb90d3eb490'}
    if a.pilot_csv:
        index=load(a.pilot_csv.with_name('NEW_RUNS_INDEX.json'))
        manifest['new_pilot']={'completed_run_rows':len(index['runs']),'formal_online_training_interactions':index['online_training_interactions'],'evaluation_rows_do_not_add_training_budget':True,'scope':index['scope'],'final_tests_evaluated':index['final_tests_evaluated']}
    (FIG/'provenance/FIGURE_PROVENANCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
